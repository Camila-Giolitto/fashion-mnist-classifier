"""Staged hyperparameter search, the script version of notebooks/02_grilla_hiperparametros.ipynb.

Stage 1: optimizer and learning rate together (a good lr depends on the optimizer).
Stages 2-4: hidden layers -> batch size -> dropout, one hyperparameter at a time. A new value is
kept only if it beats the current configuration by more than THRESHOLD of validation accuracy
(a bit more than the seed-to-seed variation). The batch stage uses early stopping, because the
batch size changes how many weight updates happen per epoch.

All comparisons use validation. Test is used once, for the final configuration.

Usage:
    python src/grid.py          # about 17 trainings: ~10 min on a Colab T4 GPU, much longer on CPU
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train import evaluate, final, fit, load_data

SEED = 42
EPOCHS = 20
MAX_EPOCHS, PATIENCE = 50, 5          # only for the stages listed in EARLY_STOP
THRESHOLD = 0.003                     # 0.3 accuracy points

BASE = {"hidden": (128, 64), "dropout": 0.2, "batch": 100}   # the course configuration
STAGE1 = {"SGD": [0.003, 0.01, 0.03, 0.1], "Adam": [0.0003, 0.001, 0.003]}
STAGES = {
    "hidden": [(64, 32), (128, 64), (256, 128), (512, 256), (256, 128, 64)],  # always >= 2 hidden layers
    "batch": [50, 100, 200],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],                                     # never above 0.4-0.5
}
EARLY_STOP = {"batch"}


def label(cfg: dict, es: bool = False) -> str:
    hidden = "-".join(map(str, cfg["hidden"]))
    return (f"{cfg['opt']} lr={cfg['lr']:g}, hidden={hidden}, dropout={cfg['dropout']}, "
            f"batch={cfg['batch']}" + (", early stop" if es else ""))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_set, valid_set, test_set = load_data("data", SEED)

    cache = {}  # label -> history, so a configuration is never trained twice

    def run(cfg, es=False):
        key = label(cfg, es)
        if key not in cache:
            kwargs = {"epochs": MAX_EPOCHS, "patience": PATIENCE} if es else {"epochs": EPOCHS}
            _, cache[key] = fit(cfg, train_set, valid_set, device, seed=SEED, **kwargs)
            h = cache[key]
            ep = (f"best epoch {h['best_epoch']} (stopped at {len(h['valid_loss'])})" if es
                  else f"lowest valid loss at epoch {int(np.argmin(h['valid_loss'])) + 1}")
            print(f"{key:<62} valid acc {100 * final(h, 'valid_acc'):5.1f} %  "
                  f"valid loss {final(h, 'valid_loss'):.3f}  "
                  f"gap {final(h, 'valid_loss') - final(h, 'train_loss'):.3f}  {ep}")
        return cache[key]

    t0 = time.time()
    rows = []  # (stage, value, history) for the summary table

    print("--- Stage 1: optimizer and learning rate")
    stage1 = {(opt, lr): run(dict(BASE, opt=opt, lr=lr)) for opt, lrs in STAGE1.items() for lr in lrs}
    best_opt, best_lr = max(stage1, key=lambda k: stage1[k]["valid_acc"][-1])
    base = dict(BASE, opt=best_opt, lr=best_lr)
    rows += [(opt, f"lr={lr:g}", h) for (opt, lr), h in stage1.items()]
    print("=> chosen:", label(base), "\n")

    for hp, values in STAGES.items():
        es = hp in EARLY_STOP
        print(f"--- Stage: {hp}" + (" (early stopping)" if es else ""))
        results = {v: run(dict(base, **{hp: v}), es) for v in values}
        ref = results[base[hp]] if base[hp] in values else run(base, es)
        v_best = max(values, key=lambda v: final(results[v], "valid_acc"))
        gain = final(results[v_best], "valid_acc") - final(ref, "valid_acc")
        if v_best != base[hp] and gain > THRESHOLD:
            base = dict(base, **{hp: v_best})
            print(f"=> {hp} changes to {v_best} (+{100 * gain:.1f} points)\n")
        else:
            print(f"=> {hp} stays at {base[hp]} (best gain {100 * gain:+.1f} points)\n")
        rows += [(hp, "-".join(map(str, v)) if hp == "hidden" else f"{v:g}", results[v]) for v in values]

    print(f"Final configuration: {label(base)}  [{time.time() - t0:.0f} s]\n")

    print(f"{'stage':<8} {'value':<12} {'train loss':>10} {'valid loss':>10} {'gap':>7} {'valid acc':>9}")
    for stage, value, h in rows:
        tl, vl, va = final(h, "train_loss"), final(h, "valid_loss"), final(h, "valid_acc")
        print(f"{stage:<8} {value:<12} {tl:>10.3f} {vl:>10.3f} {vl - tl:>7.3f} {100 * va:>8.1f}%")

    # Final model: same seed and 20 epochs, so it is the same model trained during the search.
    model, _ = fit(base, train_set, valid_set, device, seed=SEED, epochs=EPOCHS)
    test_loader = DataLoader(test_set, batch_size=base["batch"], shuffle=False)
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"\n{label(base)} -> test loss {test_loss:.3f}, test acc {100 * test_acc:.1f} %")

    out = Path("outputs"); out.mkdir(exist_ok=True)
    json.dump({"final": {**base, "hidden": list(base["hidden"])}, "test_loss": test_loss, "test_acc": test_acc,
               "runs": cache}, open(out / "grid_results.json", "w"), indent=1)
    print(f"All curves saved to {out}/grid_results.json")


if __name__ == "__main__":
    main()

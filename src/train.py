"""Train the Fashion-MNIST classifier, evaluate it once on test, and save curves and a confusion matrix.

The defaults are the configuration chosen by the hyperparameter search (see src/grid.py):
Adam, lr = 3e-4, hidden layers 512-256, dropout 0.2, batch 100, 20 epochs, seed 42.

Usage:
    python src/train.py                                      # final configuration
    python src/train.py --hidden 128 64 --lr 1e-3            # the course baseline
    python src/train.py --optimizer SGD --lr 0.1
    python src/train.py --epochs 50 --patience 5             # early stopping on validation loss
"""

import argparse
import copy
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

from model import CLASS_NAMES, FashionMLP


def set_seed(seed: int) -> None:
    """Fix every source of randomness we use: Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor_dataset(ds) -> TensorDataset:
    """Apply the transform once and keep everything in memory (about 10x faster per epoch)."""
    xs, ys = zip(*DataLoader(ds, batch_size=10_000))
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def load_data(data_dir: str = "data", seed: int = 42):
    """Train 50,000 / validation 10,000 (from the official train set) / test 10,000 (official), pixels in [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),                 # [0, 255] -> [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
    ])
    full_train = to_tensor_dataset(datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform))
    test_set = to_tensor_dataset(datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform))
    generator = torch.Generator().manual_seed(seed)  # the 50,000 / 10,000 split is always the same
    train_set, valid_set = random_split(full_train, [50_000, 10_000], generator=generator)
    return train_set, valid_set, test_set


def make_optimizer(name: str, params, lr: float):
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"unknown optimizer: {name}")


def train_one_epoch(model, data_loader, loss_fn, optimizer, device) -> None:
    model.train()  # Dropout on
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    """Return (mean loss, accuracy) over the whole DataLoader."""
    model.eval()  # Dropout off
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, data_loader, device):
    model.eval()
    preds, labels = [], []
    for images, y in data_loader:
        preds.append(model(images.to(device)).argmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(preds), torch.cat(labels)


def fit(cfg: dict, train_set, valid_set, device, seed: int = 42, epochs: int = 20,
        patience: int | None = None, verbose: bool = False):
    """Train one configuration from scratch and return (model, history).

    cfg has the keys opt, lr, hidden, dropout and batch. Everything random (initial weights,
    mini-batch order, Dropout masks) is fixed by `seed`, so the same cfg always gives the same run.

    With `patience`, training stops when the validation loss has not improved for `patience`
    epochs in a row; the weights of the best epoch are restored and history["best_epoch"] is set.
    """
    loss_fn = nn.CrossEntropyLoss()
    valid_loader = DataLoader(valid_set, batch_size=cfg["batch"], shuffle=False)
    # Train is measured with an unshuffled loader so it does not consume the training randomness.
    train_eval_loader = DataLoader(train_set, batch_size=cfg["batch"], shuffle=False)

    set_seed(seed)
    train_loader = DataLoader(train_set, batch_size=cfg["batch"], shuffle=True)
    model = FashionMLP(cfg["hidden"], cfg["dropout"]).to(device)
    optimizer = make_optimizer(cfg["opt"], model.parameters(), cfg["lr"])

    history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}
    best_loss, best_state, waited = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        tr_loss, tr_acc = evaluate(model, train_eval_loader, loss_fn, device)
        va_loss, va_acc = evaluate(model, valid_loader, loss_fn, device)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["valid_loss"].append(va_loss); history["valid_acc"].append(va_acc)
        if verbose:
            print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"valid loss {va_loss:.4f} acc {va_acc:.4f}")
        if patience:
            if va_loss < best_loss:
                best_loss, waited = va_loss, 0
                best_state = copy.deepcopy(model.state_dict())
                history["best_epoch"] = epoch
            else:
                waited += 1
                if waited >= patience:
                    if verbose:
                        print(f"early stopping: no improvement for {patience} epochs")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def final(history: dict, key: str) -> float:
    """Metric used for comparisons: the best epoch's with early stopping, otherwise the last epoch's."""
    if "best_epoch" in history:
        return history[key][history["best_epoch"] - 1]
    return history[key][-1]


def save_curves(history: dict, title: str, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], "--", label="train")
    ax1.plot(epochs, history["valid_loss"], "-", label="validation")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("CrossEntropyLoss"); ax1.grid(alpha=.3); ax1.legend()
    ax2.plot(epochs, history["train_acc"], "--", label="train")
    ax2.plot(epochs, history["valid_acc"], "-", label="validation")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.grid(alpha=.3); ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion_matrix(preds, labels, title: str, path: Path) -> torch.Tensor:
    """Row-normalised confusion matrix (rows = true class, columns = predicted class)."""
    C = torch.zeros(10, 10, dtype=torch.long)
    for y, p in zip(labels, preds):
        C[y, p] += 1
    C_norm = C.float() / C.sum(dim=1, keepdim=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(C_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    for i in range(10):
        for j in range(10):
            v = C_norm[i, j].item()
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of the row")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return C_norm


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP classifier for Fashion-MNIST")
    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="Adam")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256],
                        help="neurons per hidden layer, e.g. --hidden 256 128 64")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs (maximum with --patience)")
    parser.add_argument("--patience", type=int, default=None,
                        help="early stopping: stop after this many epochs without a better validation loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"opt": args.optimizer, "lr": args.lr, "hidden": tuple(args.hidden),
           "dropout": args.dropout, "batch": args.batch_size}
    name = (f"{args.optimizer} lr={args.lr:g}, hidden {'-'.join(map(str, args.hidden))}, "
            f"dropout {args.dropout}, batch {args.batch_size}")
    n_params = sum(p.numel() for p in FashionMLP(cfg["hidden"], cfg["dropout"]).parameters())
    print(f"Device: {device} | {name} | {n_params:,} parameters")

    train_set, valid_set, test_set = load_data(args.data_dir, args.seed)
    model, history = fit(cfg, train_set, valid_set, device, seed=args.seed, epochs=args.epochs,
                         patience=args.patience, verbose=True)

    # Test is used once, after the configuration was chosen on validation.
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test: loss {test_loss:.4f}, accuracy {test_acc:.4f}")

    preds, labels = predict(model, test_loader, device)
    print(f"Errors: {(preds != labels).sum().item()} of {len(labels)}")
    print("Per-class accuracy (test):")
    for k, cls in enumerate(CLASS_NAMES):
        mask = labels == k
        print(f"  {cls:<12} {(preds[mask] == k).float().mean().item():.3f}")

    save_curves(history, name, out_dir / "curves.png")
    C_norm = save_confusion_matrix(preds, labels, f"Normalised confusion matrix (test)\n{name}",
                                   out_dir / "confusion_matrix.png")
    off = C_norm.clone(); off.fill_diagonal_(0)
    top = torch.topk(off.flatten(), 5)
    print("Largest confusions (true -> predicted):")
    for val, idx in zip(top.values, top.indices):
        i, j = divmod(idx.item(), 10)
        print(f"  {CLASS_NAMES[i]:>12} -> {CLASS_NAMES[j]:<12} {val:.3f}")

    torch.save(model.state_dict(), out_dir / "classifier.pt")
    print(f"Figures and weights saved to {out_dir}/")


if __name__ == "__main__":
    main()

"""Fully connected network (MLP) that classifies Fashion-MNIST images into 10 classes."""

from torch import nn

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionMLP(nn.Module):
    """(B,1,28,28) -> (B,784) -> hidden[0] -> ... -> hidden[-1] -> (B,10) logits.

    Every hidden layer is Linear -> ReLU -> Dropout. The output is raw logits (no Softmax):
    nn.CrossEntropyLoss already applies log-softmax internally.
    """

    def __init__(self, hidden=(512, 256), dropout_p: float = 0.2):
        super().__init__()
        layers, n_in = [nn.Flatten()], 28 * 28
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(dropout_p)]
            n_in = n_out
        layers.append(nn.Linear(n_in, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

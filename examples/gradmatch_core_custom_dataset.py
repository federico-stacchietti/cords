#!/usr/bin/env python3
"""Minimal example showing how to use ``gradmatch_core`` with a custom dataset.

Before running make sure ``gradmatch_core`` is importable. From the repository
root execute ``pip install -e .`` or set ``PYTHONPATH=/path/to/cords`` as
explained in ``USAGE.md``.
"""

from pathlib import Path
from dotmap import DotMap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from gradmatch_core.utils.data.dataloader import GradMatchDataLoader


def main():
    data_dir = Path("../data")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_set = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=transform)

    batch_size = 128
    num_epochs = 1
    lr = 0.05

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = models.resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    dss_args = DotMap(dict(
        type="GradMatch",
        fraction=0.1,
        select_every=5,
        lam=0.5,
        selection_type="PerClassPerGradient",
        v1=True,
        valid=False,
        eps=1e-100,
        kappa=0,
        linear_layer=True,
        model=model,
        loss=criterion_nored,
        eta=lr,
        num_classes=10,
        device=device,
    ))

    gradmatch_loader = GradMatchDataLoader(
        train_loader,
        val_loader,
        dss_args,
        logger=None,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    model.train()
    for _ in range(num_epochs):
        for inputs, targets, weights in gradmatch_loader:
            inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion_nored(outputs, targets)
            loss = torch.dot(losses, weights / weights.sum())
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Minimal example showing how to integrate CORDS GradMatch with a
custom PyTorch training loop. The script trains a ResNet18 on CIFAR-10
using a GradMatch dataloader for subset selection.
"""

from pathlib import Path
from dotmap import DotMap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader
from cords.utils.models import ResNet18


def evaluate(loader, model, device):
    """Return accuracy of ``model`` on ``loader``."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, pred = outputs.max(1)
            correct += pred.eq(targets).sum().item()
    model.train()
    return 100.0 * correct / len(loader.dataset)


def main():
    data_dir = Path("../data")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 128
    num_epochs = 10
    lr = 0.05

    # Load CIFAR-10 using the CORDS dataset builder
    train_set, val_set, test_set, num_cls = gen_dataset(str(data_dir), "cifar10", "dss")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = ResNet18(num_classes=num_cls).to(device)
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # Arguments required by GradMatch
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
        num_classes=num_cls,
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

    for epoch in range(num_epochs):
        for inputs, targets, weights in gradmatch_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion_nored(outputs, targets)
            loss = torch.dot(losses, weights / weights.sum())
            loss.backward()
            optimizer.step()

        acc = evaluate(val_loader, model, device)
        print(f"Epoch {epoch+1}: validation accuracy {acc:.2f}%")

    test_acc = evaluate(test_loader, model, device)
    print(f"Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()


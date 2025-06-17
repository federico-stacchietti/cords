#!/usr/bin/env python3
"""Example showing how to integrate CORDS GradMatch with a custom dataset.

This script mirrors a typical training loop used for a proprietary dataset
and demonstrates how to wrap the user's dataloader with ``GradMatchDataLoader``
for subset selection.  All dataset utilities are expected to be available in
the current environment and are imported at runtime.
"""

from pathlib import Path
import multiprocessing as mp

from dotmap import DotMap
import torch
import torch.nn as nn
from tabulate import tabulate

from cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader

from utils import (
    load_internal,
    load_external,
    inference_on_dataloader,
    get_scores,
    get_train_loader,
    get_valid_loader,
    compute_normalization_stats,
    apply_normalization,
)
from resnet_mod import ResNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_scores(scores: dict, floatfmt: str = ".4f"):
    rows = [(k, float(scores[k])) for k in sorted(scores)]
    print(
        tabulate(rows, headers=["Metric", "Value"], tablefmt="github", floatfmt=floatfmt)
    )


def load_datasets(dataset_folder_size: str, dataset_folder_name: str, model_path: Path):
    internal = load_internal(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
        model_path=model_path,
        regex=True,
    )
    external = load_external(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
    )
    mean, std = compute_normalization_stats(internal.train)
    train_set = apply_normalization(internal.train, mean, std)
    valid_set = apply_normalization(internal.valid, mean, std)
    test_set = apply_normalization(external.test, mean, std)
    return train_set, valid_set, test_set


def main():
    mp.set_start_method("spawn", force=True)
    device = DEVICE

    batch_size = 32768
    num_epochs = 15
    lr = 1e-3

    dataset_folder_size = "kfold_100k"
    dataset_folder_name = "14-02-25_12-56_rand_seed_42"
    model_path = Path("0_internal")

    train_set, val_set, _ = load_datasets(dataset_folder_size, dataset_folder_name, model_path)

    base_train_loader = get_train_loader(train_set, balanced=True, batch_size=batch_size, neg_batch_ratio=0.8)
    val_loader = get_valid_loader(val_set, batch_size=batch_size)

    model = ResNet(input_size=26, num_classes=2, num_groups=5,
                   blocks_in_group=3, units=128, activation_fn="ReLU",
                   dropout=0.3, batch_norm=True, focal=False).to(device)
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        num_classes=2,
        device=device,
    ))

    gradmatch_loader = GradMatchDataLoader(
        base_train_loader,
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

        scores = get_scores(*inference_on_dataloader(model, val_loader, None, device))
        print_scores(scores)

    final_scores = get_scores(*inference_on_dataloader(model, val_loader, None, device))
    print("Final validation scores:")
    print_scores(final_scores)

if __name__ == "__main__":
    main()


"""
This run file demonstrates using CORDS with the GradMatch data
selection strategy on CIFAR-10. It reuses the default supervised
learning training loop provided in CORDS and loads the predefined
configuration for GradMatch.
"""

from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data


def main():
    # Path to the predefined configuration for GradMatch on CIFAR-10
    config_file = "configs/SL/config_gradmatch_cifar10.py"
    config_data = load_config_data(config_file)

    classifier = TrainClassifier(config_data)

    # Update a few training hyperparameters. These can be tuned further
    classifier.cfg.dss_args.fraction = 0.1
    classifier.cfg.dss_args.select_every = 5
    classifier.cfg.train_args.device = 'cuda'
    classifier.cfg.train_args.print_every = 1

    classifier.train()


if __name__ == "__main__":
    main()

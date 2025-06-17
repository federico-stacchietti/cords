"""
This run file shows how to use the default SL train loop provided in CORDS and 
run it with default arguments using the configuration files provided in CORDS.
"""


from pathlib import Path
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data

# Resolve the configuration path relative to this file
BASE_DIR = Path(__file__).resolve().parents[4]
config_file = BASE_DIR / "configs" / "SL" / "config_glister-warm_cifar10.py"

#CORDS comes with some predefined configuration files that mention the format
#config_file = "configs/SL/config_glister_boston.py"
#config_file = "configs/SL/config_full_boston.py"

config_data = load_config_data(config_file)
classifier = TrainClassifier(config_data)

classifier.cfg.dss_args.fraction = 0.1
classifier.cfg.dss_args.select_every = 5
classifier.cfg.train_args.device = 'cuda'
classifier.cfg.train_args.print_every = 1
classifier.cfg.is_reg = True


#classifier.configdata.dss_args.type = "Random"
#classifier.configdata.dss_args.type = "Full"
#classifier.configdata.dataset.name = "LawSchool" #"abalone"
classifier.train()

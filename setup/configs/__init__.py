"""
Load default configs
"""
from .data import get_dataset_config, get_dataloader_config
from .optimizer import get_optimizer_config, get_scheduler_config
from .model import load_model_config


def load_main_config(args, config_dir='./configs'):
    configs = {'dataset':   get_dataset_config(args, config_dir),
               'loader':    get_dataloader_config(args, config_dir),
               'optimizer': get_optimizer_config(args, config_dir),
               'scheduler': get_scheduler_config(args, config_dir)}
    return configs

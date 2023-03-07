from os.path import join
from omegaconf import OmegaConf

from dataloaders import get_data_module


def get_dataset_config(args, config_dir='./configs'):
    get_data_module(args)  # Initialize args.dataset_type
    fpath = join(config_dir, 'datasets', args.dataset_type, f'{args.dataset}.yaml')
    config = OmegaConf.load(fpath)
    config = update_dataset_config_from_args(config, args)
    return config


def get_dataloader_config(args, config_dir='./configs'):
    get_data_module(args)  # Initialize args.dataset_type
    fpath = join(config_dir, 'loader', f'{args.dataset_type}.yaml')
    config = OmegaConf.load(fpath)
    
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.pin_memory = not args.no_pin_memory
    return config


# ---------------------------------
# Update configs from argparse args
# ---------------------------------
def update_dataset_config_from_args(config, args):
    if args.dataset_type == 'informer':
        config.size = [args.lag, args.horizon, args.horizon]
        config.features = args.features
        config.variant = args.variant
        config.scale = not args.no_scale
        config.inverse = args.inverse
    else:
        pass
    return config

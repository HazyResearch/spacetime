from os.path import join
from omegaconf import OmegaConf


def get_optimizer_config(args, config_dir='./configs'):
    config = OmegaConf.load(
        join(config_dir, 'optimizer', f'{args.optimizer}.yaml'))
    if args.lr is not None:
        config.lr = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.optimizer == 'sgd' and args.momentum is not None:
        config.momentum = args.momentum
    return config


def get_scheduler_config(args, config_dir='./configs'):
    config = OmegaConf.load(
        join(config_dir, 'scheduler', f'{args.scheduler}.yaml'))
    if (config.scheduler._name_ == 'plateau' and args.val_metric == 'acc'):
        config.scheduler.mode = 'max'
    return config

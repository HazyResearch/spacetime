import os
import copy
import torch
import numpy as np
import pandas as pd

from os.path import join
from omegaconf import OmegaConf

from dataloaders import initialize_data_functions, get_evaluation_loaders
from utils.logging import print_header, print_args, print_config
from optimizer import get_optimizer, get_scheduler
from loss import get_loss
from data_transforms import get_data_transforms
from train import train_model, evaluate_model

from setup import format_arg, seed_everything
from setup import initialize_args
from setup import load_model_config, load_main_config
from setup import initialize_experiment

from model.network import SpaceTime


def main():
    print_header('*** EXPERIMENT ARGS ***')
    args = initialize_args()
    seed_everything(args.seed)
    experiment_configs = load_main_config(args, config_dir='./configs')
    
    load_data, visualize_data = initialize_data_functions(args)
    print_header('*** DATASET ***')
    print_config(experiment_configs['dataset'])
    
    print_header('*** LOADER ***')
    print_config(experiment_configs['loader'])
    
    print_header('*** OPTIMIZER ***')
    print_config(experiment_configs['optimizer'])
    
    print_header('*** SCHEDULER ***')
    print_config(experiment_configs['scheduler'])
    
    # Loading Data
    dataloaders = load_data(experiment_configs['dataset'], 
                            experiment_configs['loader'])
    train_loader, val_loader, test_loader = dataloaders
    splits = ['train', 'val', 'test']
    dataloaders_by_split = {split: dataloaders[ix] 
                            for ix, split in enumerate(splits)}
    eval_loaders = get_evaluation_loaders(dataloaders, batch_size=args.batch_size)
    
    # Setup input_dim based on features
    x, y, *z = train_loader.dataset.__getitem__(0)
    args.input_dim = x.shape[1]  # L x D
    
    # Initialize Model
    args.device = (torch.device('cuda:0') 
                   if torch.cuda.is_available() and not args.no_cuda
                   else torch.device('cpu'))
    model_configs = {'embedding_config': args.embedding_config,
                     'encoder_config':   args.encoder_config,
                     'decoder_config':   args.decoder_config,
                     'output_config':    args.output_config}
    model_configs = OmegaConf.create(model_configs)
    model_configs = load_model_config(model_configs, config_dir='./configs/model',
                                      args=args)
    
    model_configs['inference_only'] = False
    model_configs['lag'] = args.lag
    model_configs['horizon'] = args.horizon
    
    model = SpaceTime(**model_configs)
    
    if args.verbose:
        print(model)
        print_header('*** MODEL ***')
        print_config(model_configs)
    
    
    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, experiment_configs['optimizer'])
    scheduler = get_scheduler(model, optimizer, 
                              experiment_configs['scheduler'])
    
    # Save some model stats
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    args.model_parameter_count = params
    arg_dict = print_args(args, return_dict=True)
    
    # Setup logging
    wandb = initialize_experiment(args, experiment_name_id='',
                                  best_train_metric=1e10, 
                                  best_val_metric=1e10)
    try:
        pd.DataFrame.from_dict(arg_dict).to_csv(args.log_configs_path)
    except:
        pd.DataFrame.from_dict([arg_dict]).to_csv(args.log_configs_path)
                     
    
    print_header(f'*** TRAINING ***')
    print(f'├── Lag: {args.lag}')
    print(f'├── Horizon: {args.horizon}')
    print(f'├── Criterion: {args.loss}')
    print(f'├── Dims: input={args.input_dim}, model={args.model_dim}')
    print(f'├── Number trainable parameters: {params}')  # └── 
    print(f'├── Experiment name: {args.experiment_name}')
    print(f'├── Logging to: {args.log_results_path}')
    
    # Loss objectives
    criterions = {name: get_loss(name) for name in ['rmse', 'mse', 'mae']}
    eval_criterions = criterions
    for name in ['rmse', 'mse', 'mae']:
        eval_criterions[f'informer_{name}'] = get_loss(f'informer_{name}')
    
    input_transform, output_transform = get_data_transforms(args.data_transform,
                                                            args.lag)
    
    model = train_model(model, optimizer, scheduler, dataloaders_by_split, 
                        criterions, max_epochs=args.max_epochs, config=args, 
                        input_transform=input_transform,
                        output_transform=output_transform,
                        val_metric=args.val_metric, wandb=wandb, 
                        return_best=True, early_stopping_epochs=args.early_stopping_epochs)    
    
    # Eval best val checkpoint
    eval_loaders_by_split = {split: eval_loaders[ix] for ix, split in
                             enumerate(['eval_train', 'val', 'test'])}
    evaluate_model(model, dataloaders=eval_loaders_by_split, 
                   optimizer=optimizer, scheduler=scheduler, 
                   criterions=eval_criterions, config=args,
                   epoch=args.best_val_metric_epoch, 
                   input_transform=input_transform, 
                   output_transform=output_transform,
                   val_metric=args.val_metric, wandb=wandb,
                   train=False)
                     
    
if __name__ == '__main__':
    main()


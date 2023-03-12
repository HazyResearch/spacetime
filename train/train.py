"""
Training functions and helpers
"""
import importlib
import torch
import numpy as np
import pandas as pd  # Local logging
from tqdm.auto import tqdm

from .epoch import run_epoch


def print_epoch_metrics(metrics):
    for split in metrics.keys():
        print('-'*4, f'{split}', '-'*4)
        for k, v in metrics[split].items():
            if k != 'total':
                print(f'- {k}: {v:.3f}')
            else:
                print(f'- {k}: {int(v)}')

            
def train_model(model, optimizer, scheduler, dataloaders_by_split, 
                criterions, max_epochs, config, 
                input_transform=None, output_transform=None,
                val_metric='loss', wandb=None, args=None,
                return_best=False, early_stopping_epochs=100):
    
    results_dict = config.log_results_dict
    config.best_val_metric = 0 if val_metric == 'acc' else 1e10 
    config.best_val_metric_epoch = -1
    config.best_train_metric = 1e10  # Interpolation / fitting also good to test
    config.best_train_metric_epoch = -1
    
    # Experiment with C coeffs
    config.learned_c_weights = []

    pbar = tqdm(range(max_epochs))
    
    if input_transform is None:
        input_transform = lambda x: x
        
    if output_transform is None:
        output_transform = lambda y: y
        
    early_stopping_count = 0

    for epoch in pbar:
        if epoch == 0:
            pbar.set_description(f'├── Epoch {epoch}')
        else:
            description = f'├── Epoch: {epoch}'  # Display metric * 1e3
            description += f' | Best val {val_metric}: {config.best_val_metric:.3f} (epoch = {config.best_val_metric_epoch:3d})'
            for split in metrics:
                if split != 'test':  # No look
                    for metric_name, metric in metrics[split].items():
                        if metric_name != 'total':
                            description += f' | {split}/{metric_name}: {metric:.3f}'
            pbar.set_description(description)

        _, metrics, y = run_epoch(model, dataloaders_by_split, optimizer, scheduler, 
                                  criterions, config, epoch, input_transform, output_transform,
                                  val_metric, wandb)
        
        # Reset early stopping count if epoch improved
        if config.best_val_metric_epoch == epoch:  
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            
        if (epoch + 1) % config.log_epoch == 0:
            print_epoch_metrics(metrics)
            dataset_name = config.dataset if config.variant is None else f'{config.dataset}{config.variant}'
            print(f'Dataset:    {dataset_name}')
            print(f'Experiment: {config.experiment_name}')
        
        if wandb is not None:
            log_metrics = {}
            for split in metrics.keys():
                for k, v in metrics[split].items():
                    log_metrics[f'{split}/{k}'] = v
            wandb.log(log_metrics, step=epoch)
            
        # Initialize logging dict
        for split, _metrics in metrics.items():
            for k, v in _metrics.items():
                if k not in results_dict:
                    results_dict[k] = []
            break
            
        # Actually save results
        for split in metrics.keys():
            results_dict['epoch'].append(epoch)
            results_dict['split'].append(split)
            for k, v in metrics[split].items():
                results_dict[k].append(v)
                
        # Save results locally
        pd.DataFrame.from_dict(results_dict).to_csv(config.log_results_path)
            
        if early_stopping_count == early_stopping_epochs:
            print(f'Early stopping at epoch {epoch}...')
            break  # Exit for loop and do early stopping
        
    print(f'-> Saved best val model checkpoint at epoch {config.best_val_metric_epoch}!')
    print(f'   - Saved to: {config.best_val_checkpoint_path}')
    print(f'-> Saved best train model checkpoint at epoch {config.best_train_metric_epoch}!')
    print(f'   - Saved to: {config.best_train_checkpoint_path}')    
    
    if return_best:
        best_model_dict = torch.load(config.best_val_checkpoint_path)
        best_epoch = best_model_dict['epoch']
        print(f'Returning best val model from epoch {best_epoch}')
        model.load_state_dict(best_model_dict['state_dict'])
        
    return model
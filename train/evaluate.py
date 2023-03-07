"""
Functions for evaluating trained models and plotting forecasts
"""
import torch

from loss import get_loss
from .epoch import run_epoch


def evaluate_model(model, **kwargs):
    model.eval()
    log_metrics = {}
    with torch.no_grad():
        _, metrics = run_epoch(model, **kwargs)
        if kwargs['wandb'] is not None:
            for split in metrics.keys():
                for k, v in metrics[split].items():
                    log_metrics[f'eval_{split}/{k}'] = v
            kwargs['wandb'].log(log_metrics)
        # Print out evaluation metrics
        for split in metrics.keys():
            print('-'*4, f'Eval {split}', '-'*4)
            for k, v in metrics[split].items():
                print(f'- {k}: {v}')
                log_metrics[f'eval_{split}/{k}'] = v
    return model, log_metrics


def forecast(model, epoch, feature_dim, dataloaders, splits, 
             axes, show, save, epoch_kwargs):
    # Alternative to evaluate, also save / plot predictions
    assert len(dataloaders) == len(splits)
    n_plots = 3  # hard-coded for now
    if axes is None:
        fig, axes = plt.subplots(n_plots, len(dataloaders),  # Change this subplots thing 
                                 figsize=(6.4 * len(dataloaders), 4.8 * n_plots))
    split_metrics = {split: {} for split in splits}
    for split_ix, split in enumerate(splits):
        model, metrics, total_y, loss_dict = run_epoch(model, False, epoch, 
                                                       dataloaders[split_ix],
                                                       pbar_desc=f'-> Forecasting {split} split',
                                                       **epoch_kwargs)
        informer_metrics = get_informer_metrics(total_y)
        
        # Visualization
        samples_to_plot = get_plotting_samples(total_y)
        pred_ix = 0
        for pred_type, pred_samples in samples_to_plot.items():
            if pred_type != 'true':
                axis = axes[split_ix, pred_ix] if len(splits) > 1 else axes[pred_ix]
                axis.plot(samples_to_plot['true'][..., feature_dim], 
                          label='true', color='tab:orange')
                axis.plot(pred_samples[..., feature_dim], 
                          label=pred_type, color='tab:blue', linestyle='--')
                pred_ix += 1
        # axes[split_ix].legend()
        # axes[split_ix].set_title(f'{split} split forecasts', size=15)
                axis.legend()
                axis.set_title(f'{split} forecasts', size=15)
        
        split_metrics[split] = metrics
        # split_metrics[split]['total_rmse'] = total_rmse
        for k, v in informer_metrics.items():
            split_metrics[split][k] = v
            
    return split_metrics
        
        
def get_plotting_samples(y_values):
    """
    total_y = {'true': [], 'roll_cl': [], 'roll_ol': [], 'conv': []}
    
    Assumes that samples are not shuffled, strided
    """
    samples = {k: [] for k in total_y.keys()}
    for k, y in y_values.items():
        for ix, y_by_ix in enumerate(y):
            if ix == 0:  # y_by_ix is batch x len x dim
                samples[k].append(y_by_ix[0, :, :])
                samples[k].append(y_by_ix[1:, -1, :])
            else:
                samples[k].append(y_by_ix[:, -1, :])
        samples[k] = torch.cat(samples[k]).cpu()
    return samples


def get_informer_metrics(total_y):
    """
    total_y = {'true': [], 'roll_cl': [], 'roll_ol': [], 'conv': []}
    """
    informer_metrics = {}
    informer_criterions = {'informer_mse':  get_loss['informer_mse'],
                           'informer_mae':  get_loss['informer_mae'],
                           'informer_rmse':  get_loss['informer_rmse']}
    y_true = torch.cat(total_y['true']).numpy()
    for k in total_y.keys():
        if k != 'true':
            y_pred = torch.cat(total_y[k]).numpy()
            for c, criterion in informer_criterions.items():
                try:
                    informer_metrics[k][c] = criterion(y_pred, y_true)
                except KeyError:
                    informer_metrics[k] = {c: criterion(y_pred, y_true)}

    return informer_metrics

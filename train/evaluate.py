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
        _, metrics, total_y = run_epoch(model, **kwargs)
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
                
    return model, log_metrics, total_y


def plot_forecasts(y_by_splits, splits, feature_dim=0, axes=None):
    # Save / plot predictions

    n_plots = len(splits)  # hard-coded for now
    if axes is None:
        fig, axes = plt.subplots(1, n_plots, 
                                 figsize=(6.4 * n_plots, 4.8 * n_plots))

    for split_ix, split in enumerate(splits):
        y = y_by_splits[split]
        
        # Visualization
        samples_to_plot = plot_samples(y)
        pred_ix = 0
        for pred_type, pred_samples in samples_to_plot.items():
            if pred_type != 'true':
                # axis = axes[split_ix, pred_ix] if len(splits) > 1 else axes[pred_ix]
                axis = axes[split_ix]
                axis.plot(samples_to_plot['true'][..., feature_dim], 
                          label='true', color='tab:orange')
                axis.plot(pred_samples[..., feature_dim], 
                          label=pred_type, color='tab:blue', linestyle='--')
                pred_ix += 1
                axis.legend()
                axis.set_title(f'{split} forecasts', size=15)
        
        
def plot_samples(y):
    """
    y = {'true': torch.stack(total_y_true)
         'pred': torch.stack(total_y_pred),
         'true_informer': total_y_true_informer
         'pred_informer': total_y_pred_informer}
    
    Assumes that samples are not shuffled, strided
    """
    samples = {}
    for k, _y in y.items():
        if 'informer' not in k and 'true' not in k:  # Only plot raw-scale samples
            samples[k] = average_horizons(_y)
        elif k == 'true':
            samples[k] = average_horizons(_y)
    return samples


def average_horizons(y):
    """
    y.shape is B x L x D
    """
    b, l, d = y.shape
    
    total_len = b + l - 1
    total_pred = torch.zeros(b, total_len, d)
    total_pred[total_pred == 0] = float('nan')
    for ix, y_preds in enumerate(y):
        total_pred[ix][ix:ix+len(y_preds)] = y_preds
    return torch.nanmean(total_pred, dim=0)

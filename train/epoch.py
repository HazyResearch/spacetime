"""
Shared functions called during each epoch
"""
import importlib
import torch


def initialize_shared_step(config):
    step_module = importlib.import_module(f'train.step.{config.dataset_type}')
    return getattr(step_module, 'shared_step')

        
def run_epoch(model, dataloaders, optimizer, scheduler, criterions, 
              config, epoch, input_transform=None, output_transform=None,
              val_metric='loss', wandb=None, train=True):
    # dataloaders is {'train': train_loader, 'val': val_loader, 'test': test_loader}
    metrics = {split: None for split in dataloaders.keys()}
    total_y = {split: None for split in dataloaders.keys()}
    shared_step = initialize_shared_step(config)
    
    
    for split, dataloader in dataloaders.items():
        try:
            mean = dataloader.dataset.standardization['means'][0]
            std  = dataloader.dataset.standardization['stds'][0]
        except AttributeError:
            mean = 0.
            std = 1.
        
        model, _metrics, y = shared_step(model, dataloader, optimizer, scheduler, 
                                         criterions, epoch, config, split, 
                                         input_transform=input_transform,
                                         output_transform=output_transform)
        metrics[split] = _metrics
        total_y[split] = y
        
    if train:
        # Save checkpoints if metric better than before
        save_checkpoint(model, optimizer, config, epoch, 'val', val_metric,
                        metrics['val'][val_metric], config.best_val_metric)
        save_checkpoint(model, optimizer, config, epoch, 'train', val_metric,
                        metrics['train'][val_metric], config.best_train_metric)

        # Update optimizer
        if config.scheduler == 'plateau':
            scheduler.step(metrics['val'][val_metric])
        elif config.scheduler == 'timm_cosine':
            scheduler.step(epoch)

    return model, metrics, total_y


def better_metric(metric_a, metric_b, metric_name):
    if metric_name == 'acc':
        return metric_a > metric_b
    else:
        return metric_a < metric_b


def save_checkpoint(model, optimizer, config, epoch, split,
                    val_metric, run_val_metric, best_val_metric): 
    checkpoint_path = getattr(config, f'best_{split}_checkpoint_path')
    try:  # try-except here because checkpoint fname could be too long
        if (better_metric(run_val_metric, best_val_metric, val_metric) or epoch == 0):
            setattr(config, f'best_{split}_metric', run_val_metric)
            setattr(config, f'best_{split}_metric_epoch', epoch)
            config.best_val_metric = run_val_metric
            config.best_val_metric_epoch = epoch
            torch.save({'epoch': epoch,
                        'val_metric': run_val_metric,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                       }, checkpoint_path)
    except Exception as e:
        print(e)

"""
Model optimizer and scheduler
"""
import torch


def get_optimizer(model, configs):
    optim_configs = {k: v for k, v in configs.items() if k != '_name_'}
    if configs['_name_'] == 'adamw':
        return torch.optim.AdamW(model.parameters(), **optim_configs)
    elif configs['_name_'] == 'sgd':
        return torch.optim.SGD(model.parameters(), **optim_configs)
    elif configs['_name_'] == 'adam':
        return torch.optim.Adam(model.parameters(), **optim_configs)
    
    
def get_scheduler(model, optimizer, configs):
    if 'scheduler' in configs:
        configs = configs['scheduler']
    scheduler_configs = {k: v for k, v in configs.items() if k != '_name_'}
    if configs['_name_'] == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        print(scheduler_configs)
        try:
            return ReduceLROnPlateau(optimizer=optimizer, **scheduler_configs)
        except:
            return ReduceLROnPlateau(optimizer=optimizer)
    else:
        return None
    

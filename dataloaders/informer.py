import numpy as np
import matplotlib.pyplot as plt

from dataloaders.datasets.informer import ETTHour, ETTMinute, ECL, Exchange, ILI, Traffic, Weather


def get_dataset(name):
    if name == 'etth':
        return ETTHour
    elif name == 'ettm':
        return ETTMinute
    elif name == 'ecl':
        return ECL
    elif name == 'exchange':
        return Exchange
    elif name == 'ili':
        return ILI
    elif name == 'traffic':
        return Traffic
    elif name == 'weather':
        return Weather
    else:
        supported = ['etth', 'ettm', 'ecl', 'exchange', 'ili', 'traffic', 'weather']
        raise NotImplementedError(f"Please check that name is in {supported}")
    

def load_data(config_dataset, config_loader):
    dataset = get_dataset(config_dataset['_name_'])(**config_dataset)
    dataset.setup()
    
    train_loader = dataset.train_dataloader(**config_loader)
    # Eval loaders are dictionaries where key is resolution, value is dataloader
    # - Borrowed from S4 dataloaders. For now just set resolution to 1
    val_loader   = dataset.val_dataloader(**config_loader)[None]
    test_loader  = dataset.test_dataloader(**config_loader)[None]
    return (train_loader, val_loader, test_loader), dataset


def visualize_data(dataloaders, splits=['train', 'val', 'test'],
                   save=False, args=None, title=None):
    assert len(splits) == len(dataloaders)
    start_idx = 0
    for idx, split in enumerate(splits):
        y = dataloaders[idx].dataset.data_x
        x = np.arange(len(y)) + start_idx
        plt.plot(x, y, label=split)
        start_idx += len(x)
    plt.title(title)
    plt.legend()
    plt.show()
"""
Model loss functions and objectives
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import L1Loss as MAE
from torch.nn import MSELoss as MSE
from torch.nn import CrossEntropyLoss


def get_loss(loss, reduction='none', ignore_index=-100):
    """
    Different loss functions depending on the dataset / task
    """
    if loss == 'mse':
        return nn.MSELoss(reduction=reduction)
    elif loss == 'mae':
        return nn.L1Loss(reduction=reduction)
    elif loss == 'rmse':
        return multivariate_RMSE(reduction=reduction)
    elif loss == 'cross_entropy':
        return nn.CrossEntropyLoss(reduction=reduction,
                                   ignore_index=ignore_index)
    elif loss == 'informer_mse':
        return informer_MSE
    elif loss == 'informer_mae':
        return informer_MAE
    elif loss == 'informer_rmse':
        return informer_RMSE
        
    
def multivariate_RMSE(reduction='none'):
    criterion = torch.nn.MSELoss(reduction=reduction)
    def loss(y_pred, y_true, ):
        return torch.sqrt(
            criterion(y_pred, y_true).mean(dim=0))  # .mean()
    return loss


# Losses from Informer code
def informer_MAE(y_pred, y_true):
    return torch.mean(torch.abs(y_pred-y_true))


def informer_MSE(y_pred, y_true):
    return torch.mean((y_pred-y_true)**2)


def informer_RMSE(y_pred, y_true):
    return torch.sqrt(informer_MSE(y_pred, y_true))


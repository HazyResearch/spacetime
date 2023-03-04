"""
Basic neural net components

OurModule from: https://github.com/HazyResearch/state-spaces/blob/main/src/models/sequence/ss/kernel.py (OptimModule)
Activation from: https://github.com/HazyResearch/state-spaces/blob/main/src/models/nn/components.py
"""

import torch.nn as nn


class OurModule(nn.Module):
    def __init__(self): 
        super().__init__()

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""
        if trainable:
            try:
                self.register_parameter(name, nn.Parameter(tensor))
            except KeyError:
                delattr(self, name)
                self.register_parameter(name, nn.Parameter(tensor))
        else:
            
            try:
                self.register_buffer(name, tensor)
            except KeyError:
                delattr(self, name)
                self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None: optim["lr"] = lr
        if trainable and wd is not None: optim["weight_decay"] = wd
        if len(optim) > 0: setattr(getattr(self, name), "_optim", optim)
        

def Activation(activation=None, size=None, dim=-1, inplace=False):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity(inplace)
    elif activation == 'tanh':
        return nn.Tanh(inplace)
    elif activation == 'relu':
        return nn.ReLU(inplace)
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU(inplace)
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid(inplace)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))
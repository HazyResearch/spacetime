"""
Basic neural net components

OurModule from: https://github.com/HazyResearch/state-spaces/blob/main/src/models/sequence/ss/kernel.py (OptimModule)
Activation and DropoutND from: https://github.com/HazyResearch/state-spaces/blob/main/src/models/nn/components.py
"""
import torch
import torch.nn as nn

from einops import rearrange


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
        
        
class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, x):
        """ x: (batch, lengths..., dim) """
        if self.training:
            if self.transposed: x = rearrange(x, 'b ... d -> b d ...')
            mask_shape = x.shape[:2] + (1,)*(x.ndim-2) if self.tie else x.shape
            mask = torch.rand(*mask_shape, device=x.device) < 1.-self.p
            x = x * mask * (1.0/(1-self.p))
            if self.transposed: x = rearrange(x, 'b d ... -> b ... d')
            return x
        return x

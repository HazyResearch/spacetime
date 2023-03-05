import torch.nn as nn

from einops import rearrange
from model.components import Activation


def init_mlp(config):
    if config['method'] == 'mlp':
        return MLP(**config['kwargs'])
    else:
        return nn.Identity()


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,                 
                 output_dim: int,
                 activation: str=None,
                 dropout: float=0.,
                 layernorm: bool=False,
                 n_layers: int=1,
                 n_activations: int=0,
                 pre_activation: bool=False,
                 input_shape: str='bld',
                 hidden_dim: int=None,
                 skip_connection: bool=False,
                 average_pool: str=None):
        """
        Fully-connected network 
        """
        super().__init__()
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim
        self.input_shape   = input_shape
        
        self.activation      = Activation(activation, inplace=True)
        self.dropout         = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layernorm       = nn.LayerNorm(input_dim) if layernorm else nn.Identity()
        self.n_layers        = n_layers
        self.n_activations   = n_activations
        self.pre_activation  = pre_activation
        self.skip_connection = skip_connection
        self.average_pool    = average_pool
        
        self.initialize_layers()
        
    def initialize_layers(self):
        n_layers_to_init = self.n_layers
        n_activations_to_init = self.n_activations
        
        if self.hidden_dim is None:  # Probs not great, but implicitly handle
            self.hidden_dim = self.output_dim
            
        # Add layers
        if self.n_activations > self.n_layers or self.pre_activation:
            layers = [self.activation]
            n_activations_to_init -= 1
        else:
            layers = []
            
        while n_layers_to_init > 0 or n_activations_to_init > 0:
            if n_layers_to_init == self.n_layers:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif n_layers_to_init > 1:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            elif n_layers_to_init == 1:
                layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            
            if n_activations_to_init > 0:
                layers.append(self.activation)
            
            n_layers_to_init -= 1
            n_activations_to_init -= 1
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layernorm(x)
        
        if self.input_shape == 'bdl':
            x = rearrange(x, 'b d l -> b l d')
        
        if self.skip_connection:
            # Layernorm with skip connection
            x = self.layers(x) + x  
        else: 
            x = self.layers(x)
        
        x = self.dropout(x)
        
        if self.average_pool == 'l':
            x = x.mean(dim=1, keepdim=True)
        return x
import torch
import torch.nn as nn
import opt_einsum as oe
from einops import rearrange, repeat

from model.components import OurModule


class SSM(OurModule):
    def __init__(self, 
                 model_dim: int, 
                 n_kernels: int,  # Number of kernels / scales
                 kernel_dim: int,
                 kernel_repeat: int,
                 n_heads: int=None,  # Number of heads per kernel
                 head_dim: int=1, # Dimension of each head
                 kernel_weights: torch.float=None,
                 kernel_init: str='normal',
                 kernel_train: bool=True,
                 skip_connection: bool=False):
        super().__init__()
        # At least one of these should be int
        assert not (n_heads is None and head_dim is None)
                 
        self.model_dim = model_dim
        self.n_kernels = n_kernels
        self.kernel_dim = kernel_dim
        self.kernel_repeat = kernel_repeat
        self.head_dim, self.n_heads = self.init_heads(n_heads, head_dim)
        self.kernel_weights  = kernel_weights
        self.kernel_init     = kernel_init
        self.kernel_train    = kernel_train
        self.skip_connection = skip_connection
        
        self.init_weights()
        
    def init_heads(self, n_heads: int, head_dim: int):
        if head_dim is None:
            self.head_dim = self.model_dim // (self.kernel_repeat * 
                                               self.n_kernels * n_heads)
            self.n_heads  = n_heads
        else:
            self.head_dim = head_dim
            self.n_heads  = self.model_dim // (self.kernel_repeat * 
                                               self.n_kernels * head_dim)
        return self.head_dim, self.n_heads
        
    def fft_conv(self, u_input: torch.tensor, v_kernel: torch.tensor):
        # Convolve u with v in O(n log n) time with FFT (n = len(u))
        L   = u_input.shape[-1]  # Assume u is input
        u_f = torch.fft.rfft(u_input, n=2*L) # (B H L)
        v_f = torch.fft.rfft(v_kernel[:, :L], n=2*L) # (H L)

        y_f = oe.contract('b h l, h l -> b h l', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2*L)[..., :L]  # (B H L)
        return y
    
    def init_weights(self):
        if self.kernel_weights is not None:  
            # lr and wd as None sets them to be same as model lr and weight_decay
            register('k', self.kernel_weights, trainable=True, lr=None, wd=None)
        
        skip = torch.ones(self.model_dim)
        self.register('skip', skip, trainable=True, lr=None, wd=None)
    
    def get_kernel(self):
        raise NotImplementedError
        
    def forward(self, u):
        u = rearrange(u, 'b l d -> b d l')  # Assume u is B x L x D
        # Repeat kernels across heads
        if self.kernel_weights is None:
            k = self.get_kernel(u)
            k = repeat(k, 'nk kd -> (kr nk nh hd) kd', 
                   kr=self.kernel_repeat, nh=self.n_heads, hd=self.head_dim)
        else:
            k = self.k
        try:
            y = self.fft_conv(u, k)
        except Exception as e:
            print(e)
            breakpoint()
        if self.skip_connection:
            y = y + oe.contract('b d l, d -> b d l', u, self.skip)
        y = rearrange(y, 'b d l -> b l d')
        return y
        
        
class IdentitySSM(SSM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_weights(self):
        self.register('kernel', None, trainable=False)
        
    def forward(self, u):
        return u

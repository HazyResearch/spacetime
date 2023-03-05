import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
from einops import repeat, rearrange
from model.functional.krylov import krylov

from .base import Kernel


class ShiftKernel(Kernel):
    """
    Open-loop implementation of Shift SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} S^k B u_i
    """
    def __init__(self, **kwargs):
    #     model_dim: int, 
    #     n_kernels: int,  # Number of kernels / scales
    #     kernel_dim: int,
    #     kernel_repeat: int=1,
    #     n_heads: int=None,  # Number of heads per kernel
    #     head_dim: int=1, # Dimension of each head
    #     kernel_weights: torch.float=None,
    #     kernel_init: str='normal',
    #     kernel_train: bool=True,
    #     skip_connection: bool=False
        kwargs['kernel_repeat'] = 1
        # Set kwargs['n_heads'] as n_kernels for preprocessing kernels
        super().__init__(**kwargs)
        
    def init_weights(self):
        if self.kernel_weights is None:
            if self.kernel_init == 'normal':
                kernel = torch.randn(self.n_heads, self.kernel_dim)
            elif self.kernel_init == 'xavier':
                # Xavier-ish initialization
                stdv = 1. / math.sqrt(self.kernel_dim)
                kernel = torch.FloatTensor(
                    self.n_kernels, self.kernel_dim).uniform_(-stdv, stdv)
            else:
                raise NotImplementedError
            self.register("kernel", kernel, trainable=self.kernel_train, 
                          lr=None, wd=None)
        else:
            self.register("k", self.kernel_weights, trainable=self.kernel_train, 
                          lr=None, wd=None)
            
        if self.skip_connection:
            skip = torch.randn(self.input_dim)
            self.register("skip", skip, trainable=True, lr=None, wd=None)
    
    def get_kernel(self, u):
        """
        From kernel weights, instantiate the kernel to convolve the input with
        Args:
        - u: sample, assume shape is B x D x L
        """
        kernel = torch.zeros(self.n_heads, 
                             (self.max_dilation + 1) * self.kernel_dim).float().to(u.device)
        # Could be slow
        for ix in range(self.n_heads):
            kernel_i = rearrange(
                [self.kernel[ix]] + self.dilations[ix] * [self.kernel_pad[ix]], 
                'r d -> (d r)')
            kernel[ix, :len(kernel_i)] = kernel_i
        kernel = kernel[..., :self.end_padding]
        return kernel.to(u.device)  
        
    def forward(self, u, u_horizon=None):
        u = rearrange(u, 'b l d -> b d l')  # Assume u is B x L x D
        k = self.get_kernel(u)  if self.kernel_weights is None else self.k
        y = self.fft_conv(u, k)
        if self.skip_connection:
            y = y + oe.contract('b d l, d -> b d l', u, self.skip)
        y = rearrange(y, 'b d l -> b l d')
        return y
    
    
class ClosedLoopShiftKernel(ShiftKernel):
    """
    Closed-loop implementation of Shift SSM:
    - Instantiate A, B, C; so we can compute both:
    - Open-loop inference:   y_{n + h - 1} = \sum_{i = 0}^{n + h - 1} CA^{n + h - 1 - i} B u_i
    - Closed-loop inference: y_{n + h - 1} = \sum_{i = n}^{h - 1} C(A + BC)^{n - 1 - i} B u_i + C(A + BC)^{h - 1}A x_n
    
    Just do strict trainable shift kernel in this case
    """
    def __init__(self, n_heads, input_dim, kernel_dim,
                 skip_connection=False, kernel_weights=None,
                 kernel_train=True, dilations=[0], kernel_init='normal',
                 closed_loop=True, use_initial=True):  #, target_len=None):
        # TODO: see if we don't need these assertions 
        assert dilations == [0]
        assert kernel_weights is None
        assert kernel_train is True
        assert skip_connection is False
        self.kernel_shape = (n_heads, kernel_dim)
        self.closed_loop  = closed_loop
        self.horizon_len  = None  # How many samples to rollout
        self.use_initial  = use_initial
        super().__init__(n_heads=n_heads, 
                         input_dim=input_dim,
                         kernel_dim=kernel_dim,
                         skip_connection=skip_connection, 
                         kernel_weights=kernel_weights,
                         kernel_train=kernel_train, 
                         dilations=dilations,
                         kernel_init=kernel_init)
        
    def init_weights(self):
        # A matrix ix fixed shift matrix
        A = torch.zeros(self.n_heads, self.kernel_dim, self.kernel_dim)
        A[:, 1:, :-1] = torch.eye(self.kernel_dim - 1)
        self.register("A", A, trainable=False, lr=None, wd=None)
        
        # B matrix is fixed
        b    =  torch.zeros(self.kernel_dim).float()
        b[0] = 1.
        b    = repeat(b, 'd -> h d', h=self.n_heads).clone().contiguous()
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C matrix is trainable
        if self.kernel_init == 'normal':
            c = torch.randn(self.n_heads, self.kernel_dim)
        elif self.kernel_init == 'xavier':
            # Xavier-ish initialization
            stdv = 1. / math.sqrt(self.kernel_dim)
            # Initialize same weights across input_dims
            c = torch.FloatTensor(self.n_heads, self.kernel_dim).uniform_(-stdv, stdv)
        else:
            raise NotImplementedError
        self.register("c", c, trainable=self.kernel_train, 
                      lr=None, wd=None)
        
    def norm(self, x, ord=1, dim=2):
        # x.shape = C x H x D
        x_norm = torch.linalg.norm(x, ord=ord, dim=dim, keepdim=True)
        x = x / x_norm
        return x
    
    def convolve(self, u, A, l, b, c):
        """
        if self.closed_loop:
        - compute: \sum_{i=0}^{t - 1} c(A + bc)^{t - 1 - i}b u_i
        else:
        - compute: \sum_{i=0}^{t - 1} cA^{t - 1 - i}b u_i
        """
        k = krylov(l, A, b, c)
        y = self.fft_conv(u, k)
        return y
        
    def get_kernel(self, u, A, l, b, c):
        """
        If self.closed_loop, compute kernel such that:
        - y_t = \sum_{i=0}^{t - 1} c(A + bc)^{t - 1 - i}b u_i
        
        Else, compute kernel such that:
        - y_t = \sum_{i=0}^{t - 1} cA^{t - 1 - i}b u_i
        """
        return krylov(l, A, b, c)
        
    def forward(self, u, u_horizon=None):
        """
        Assume u is B x L x D, u_0, ..., u_{L - 1}
        u_horizon is future values, could be [u_n, 0, ... 0]
        where u_n is the last output of the layer before
        """
        u = rearrange(u, 'b l d -> b d l')
        b, d, l = u.shape
        c = F.normalize(self.c, p=1, dim=1)
        # Closed-loop Inference
        if self.closed_loop:
            # BC = oe.contract('h i, h j -> h i j', self.b, self.c)
            BC = oe.contract('h i, h j -> h i j', self.b, c)
            # A_BC = self.norm(self.A + BC)
            A_BC = self.A + BC
            if self.use_initial:  # Compute hidden-state term first
                # Inefficient with the passes
                hidden_c = krylov(l, A_BC, self.A, self.c)[:, :, -1]
                hidden_y  = self.convolve(u, self.A, l, self.b, hidden_c)[:, :, -1:]
            else:
                hidden_y = 0
            # Now compute future horizon term
            # if u_horizon is None:
            #     u_horizon = torch.zeros(b, self.horizon_len, d)
            #     u_horizon[:, 0, :] = u[:, :, -1]  # No noise for now
    
            u_horizon = rearrange(u_horizon, 'b l d -> b d l').to(u.device)
            l_horizon = u_horizon.shape[2]
            # y = self.convolve(u_horizon, A_BC, l_horizon, self.b, self.c)
            y = self.convolve(u_horizon, A_BC, l_horizon, self.b, c)
            # Add up for final y
            y = y + hidden_y
            # Add original input terms too
            # y = torch.cat([u[:, :, :-l_horizon], y], dim=2)
            y = torch.cat([u[:, :, :], y], dim=2)
        else:  # Currently not super principled with use_initial
            # y = self.convolve(u, self.A, l, self.b, self.c)
            y = self.convolve(u, self.A, l, self.b, c)
        return rearrange(y, 'b d l -> b l d')

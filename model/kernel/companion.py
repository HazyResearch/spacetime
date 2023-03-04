import torch
import opt_einsum as oe
from einops import repeat, rearrange

from model.functional.krylov import krylov
from model.kernel.base import Kernel


class CompanionKernel(Kernel):
    """
    Open-loop implementation of Companion SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} A^k B u_i
       where A is companion matrix
    """
    def __init__(self, **kwargs):
        kwargs['kernel_repeat'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = True
        # Set kwargs['n_heads'] as n_kernels for preprocessing kernels
        # Set kwargs['head_dim'] to be original sample input dim
        super().__init__(**kwargs)
        
    def init_kernel_weights(self, kernel_init):
        if kernel_init == 'normal':
            kernel = torch.randn(self.n_kernels, self.kernel_dim)
        elif kernel_init == 'xavier':
            # Xavier-ish initialization
            stdv = 1. / math.sqrt(self.kernel_dim)
            # Initialize same weights across input_dims
            kernel = torch.FloatTensor(
                self.n_kernels, self.kernel_dim).uniform_(-stdv, stdv)
        else:
            raise NotImplementedError
        return kernel
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection
        self._fp = (self.n_kernels, self.kernel_dim)
        
        # Shift matrix initialization
        self.shift_matrix = torch.zeros(self.n_kernels, 
                                        self.kernel_dim, 
                                        self.kernel_dim)
        self.shift_matrix[:, 1:, :-1] = torch.eye(self.kernel_dim - 1)
        self.p_padding = torch.zeros(*self._fp)
        self.p_padding[:, -1] = 1.
        
        # A matrix
        a = self.init_kernel_weights(self.kernel_init)
        self.register("a", a, trainable=True, lr=None, wd=None)
        
        # B matrix
        b = self.init_kernel_weights(self.kernel_init) 
        self.register("b", b, trainable=True, lr=None, wd=None)
        
        # C matrix
        c = self.init_kernel_weights(self.kernel_init)
        self.register("c", c, trainable=True, lr=None, wd=None)
    
    def norm(self, x, ord=1):
        # x.shape = C x H x D
        x_norm = torch.linalg.norm(x, ord=ord, dim=2, keepdim=True)
        x = x / x_norm if x_norm[:, 0].item() != 0 else x 
        return x
    
    def matrix_power(self, l, c, b, p):
        ch, h, d = b.shape 
        # Construct companion matrix
        A = self.shift_matrix.to(p.device) + (
            oe.contract('h i, h j -> h j i', 
                        self.p_padding.to(p.device), p)
        )
        # Use repeated squares to power A
        g = krylov(l, A, b, c)
        return g
    
    def get_kernel(self, u, c=None, l=None):
        l = u.shape[-1] if l is None else l
        c = self.c if c is None else c
        a = self.norm(self.a, ord=1)
        f = self.matrix_power(l, c, self.b, a).to(u.device)
        return f
    
    def forward(self, u):
        return super().forward(u)
        
        
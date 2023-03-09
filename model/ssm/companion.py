import torch
import torch.nn.functional as F
import opt_einsum as oe
from einops import repeat, rearrange

from model.functional.krylov import krylov
from model.ssm.base import SSM


class CompanionSSM(SSM):
    """
    Open-loop implementation of Companion SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} A^k B u_i
       where A is companion matrix
    """
    def __init__(self, norm_order, **kwargs):
        self.norm_order = norm_order
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
            kernel = torch.FloatTensor(self.n_kernels, 
                                       self.kernel_dim).uniform_(-stdv, stdv)
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
        # x.shape is either (H x D) or (H x D x D)
        x_norm = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        # If norm(x) in batch close to 0, don't normalize 
        # (heuristicky, but we norm for stability)
        try:
            x = x / x_norm if torch.abs(x_norm).mean().item() > 1e-4 else x  
        except Exception as e:
            print(e)
            breakpoint()
        # x = F.normalize(x, dim=1, p=ord, eps=1)
        return x
    
    def matrix_power(self, l, c, b, p):
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
        a = (self.norm(self.a, ord=self.norm_order) 
             if self.norm_order > 0 else self.a)
        f = self.matrix_power(l, c, self.b, a).to(u.device)
        return f
    
    def forward(self, u):
        return super().forward(u)
        
        
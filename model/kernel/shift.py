import torch
import opt_einsum as oe
from einops import repeat, rearrange

from model.functional.krylov import krylov
from model.kernel.companion import CompanionKernel


class ShiftKernel(CompanionKernel):
    """
    Open-loop implementation of Shift SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} S^k B u_i
       where S is shift matrix
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection, B, C matrices
        
        # A matrix (technically initialized above, but now we zero-out)
        a = torch.zeros(self.n_kernels, self.kernel_dim)
        self.register("a", a, trainable=False, lr=None, wd=None)
    
    def forward(self, u):
        return super().forward(u)
import torch
import torch.nn.functional as F
import opt_einsum as oe
from einops import repeat, rearrange

from model.functional.krylov import krylov
from model.ssm.closed_loop.companion import ClosedLoopCompanionSSM


class ClosedLoopShiftSSM(ClosedLoopCompanionSSM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection, A, B, C
        # A Matrix
        a = torch.zeros(self.n_kernels, self.kernel_dim)
        self.register("a", a, trainable=False, lr=None, wd=None)
        
        # B Matrix - make it not learnable
        b = torch.zeros(self.n_kernels, self.kernel_dim)
        b[:, 0] = 1
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C matrix
        c = self.init_kernel_weights(self.kernel_init)
        self.register("c", c, trainable=True, lr=None, wd=None)
        
        # K matrix
        k = self.init_kernel_weights(self.kernel_init)
        self.register("k", k, trainable=True, lr=None, wd=None)
        
    def get_companion_matrix(self, p):
        # Construct "companion" matrix
        return self.shift_matrix.to(p.device)

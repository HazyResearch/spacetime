import torch
import opt_einsum as oe
from einops import repeat, rearrange

from model.functional.krylov import krylov
from model.ssm.companion import CompanionSSM


class ClosedLoopCompanionSSM(CompanionSSM):
    """
    Closed-loop implementation of Companion SSM:
    - Instantiate A, B, C; so we can compute both:
    - Open-loop inference:   
      -> y_{n + h} = \sum_{i = 0}^{n + h - 1} CA^{n + h - 1 - i} B u_i
    - Closed-loop inference: 
      -> y_{n + h} = C(A + BK)^{h} x_n
                   = C(A + BK)^{h} \sum_{j = 0}^{n - 1} A^{n - 1 - j} B u_j
                   = C(A + BK)^{n + h - 1} x_1
                   = C(A + BK)^{n + h - 1} B u_0
                   = \sum_{i = 0}^{n + h - 1} C(A + BK)^{n + h - 1 - i} B u_i, u_j = 0 for j > 0
    """
    def __init__(self, 
                 lag: int=1,
                 horizon: int=1,
                 use_initial: bool=False,
                 **kwargs):
        self.lag     = lag
        self.horizon = horizon
        self.use_initial = use_initial  # When False, assumes initial hidden_state x_0 = 0. True not implemented
        self.closed_loop = True         # Toggle closed or open-loop forward pass, see self.forward
        self.inference_only = False     # Toggle different behavior during training and test, see self.get_kernel
        kwargs['kernel_repeat'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = True
        kwargs['skip_connection'] = False
        # Set kwargs['n_heads'] as n_kernels for preprocessing kernels if tie_weights else arbitrary?
        # Set kwargs['head_dim'] to be original sample input dim
        super().__init__(**kwargs)
        
    def init_kernel_weights(self, kernel_init):
        if kernel_init == 'normal':
            kernel = torch.randn(self.n_kernels, self.kernel_dim)
        elif kernel_init == 'xavier':
            # Xavier-ish initialization
            stdv = 1. / math.sqrt(self.kernel_dim)
            kernel = torch.FloatTensor(
                self.n_kernels, self.kernel_dim).uniform_(-stdv, stdv)
        else:
            raise NotImplementedError
        return kernel
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection, A, B, C
        # K matrix
        k = self.init_kernel_weights(self.kernel_init)
        self.register("k", k, trainable=True, lr=None, wd=None)
    
    def norm(self, x, ord=1):
        # x.shape = C x H x D
        x_norm = torch.linalg.norm(x, ord=ord, dim=2, keepdim=True)
        x = x / x_norm if x_norm[:, 0].item() != 0 else x 
        return x
    
    def get_companion_matrix(self, p):
        # Construct companion matrix
        return self.shift_matrix.to(p.device) + (
            oe.contract('h i, h j -> h j i', 
                        self.p_padding.to(p.device), p)
        )
    
    def get_kernel(self, u, a, l):
        # Handles several cases:
        # 1. Inference only (i.e., not training) and closed-loop,
        # 2. Not inference only (i.e., during training) and closed-loop
        # 3. Not inference only and open-loop
        a = self.norm(a, ord=1)
        if self.inference_only and self.closed_loop:
            # Only return kernel for final model output predictions, i.e.,
            # Krylov matrix (CB, CAB, CA^2B, ... CA^{l + h - 1}B) 
            # where C is self.c, A is a = A + BK (should be calculated before method call)
            k_y = krylov(l, a, self.b, self.c).to(u.device)  # Use repeated squares to power A
            k_u = None
            
        elif not self.inference_only and self.closed_loop:
            # Return both final output kernel (k_y) and next-time-step input kernel (k_u), i.e.,
            # both C(A + BK)^{n}B and K(A + BK)^{n}B
            k = krylov(l, a, self.b, c=None).to(u.device)
            k_y = torch.einsum('...nl, ...n -> ...l', k, self.c).continuous()
            k_u = torch.einsum('...nl, ...n -> ...l', k, self.k).continuous()
            
        elif self.inference_only and not self.closed_loop:
            raise NotImplementedError  # Never happens
        
        elif not self.inference_only and not self.closed_loop:
            # Return CA^{n}B where A = a is computed companion matrix from self.a
            k_y = krylov(l, a, self.b, self.c).to(u.device)
            k_u = None
            
        return k_y, k_u
    
    def forward(self, u):
        """
        During training, call this function twice to compute closed-loop and open-loop
        -> minimize the closed-loop?
        """
        u = rearrange(u, 'b l d -> b d l')
        b, d, l = u.shape
        l_horizon = self.horizon
        
        A = self.get_companion_matrix(self.a)
        if self.closed_loop:  # Compute closed-loop forecast
            # Get input
            u_horizon = torch.zeros(b, d, l + l_horizon)
            u_horizon[:, :, 0] = u[:, :, 0]  # supervised with y[1], ... , y[L + H]
            
            # Get A + BK matrix
            A_BK = A + oe.contract('h i, h j -> h i j', self.b, self.k)
            # Get kernels for output and next-step input
            k_horizons = self.get_kernel(u_horizon, a=A_BK, l=l+l_horizon)
            k_horizons = [repeat(k_horizon, 'nk kd -> (kr nk nh hd) kd', 
                                 kr=self.kernel_repeat, nh=self.n_heads,
                                 hd=self.head_dim)
                          for k_horizon in k_horizons]
            y = [None, None]  # layer outputs, and next-time-step layer inputs
            for kix, k in enumerate(k_horizons):
                if k is not None:
                    y[kix] = rearrange(self.fft_conv(u_horizon, k),
                                       'b d l -> b l d')
            return y[0], y[1]  # shape is B x (L + H) x D if not None
        else:
            # Compute open-loop forecast up to L
            k = self.get_kernel(u, a=A, l=l)
            k = repeat(k, 'nk kd -> (kr nk nh hd) kd', 
                       kr=self.kernel_repeat, nh=self.n_heads, hd=self.head_dim)
            y = rearrange(self.fft_conv(u, k), 'b d l -> b l d')
            return y, None  # shape is B x L x D
        
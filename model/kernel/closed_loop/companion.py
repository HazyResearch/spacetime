import torch
import opt_einsum as oe
from einops import repeat, rearrange

from model.functional.krylov import krylov
from model.kernel.companion import CompanionKernel


class ClosedLoopCompanionKernel(CompanionKernel):
    """
    Closed-loop implementation of Companion SSM:
    - Instantiate A, B, C; so we can compute both:
    - Open-loop inference:   
      -> y_{n + h - 1} = \sum_{i = 0}^{n + h - 1} CA^{n + h - 1 - i} B u_i
    - Closed-loop inference: 
      -> y_{n + h - 1} = \sum_{i = n}^{h - 1} C(A + BC)^{n - 1 - i} B u_i + C(A + BC)^{h - 1}A x_n
    """
    def __init__(self, 
                 closed_loop: bool=True,
                 use_initial: bool=False,
                 **kwargs):
        self.closed_loop = closed_loop
        self.use_initial = use_initial
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
            # Initialize same weights across input_dims
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
    
    def get_kernel(self, u, c, a, b, l):
        # l = u.shape[-1] if l is None else l
        # c = self.c if c is None else c
        a = self.norm(a, ord=1)
        f = self.matrix_power(l, c, b, a).to(u.device)
        return f
    
    def forward(self, u, l_horizon):
        """
        During training, call this function twice to compute closed-loop and open-loop
        -> minimize the closed-loop?
        """
        u = rearrange(u, 'b l d -> b d l')
        b, d, l = u.shape
        
        # krylov(L, A, b, c=None)
        # Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.
        
        if self.closed_loop:
            # Compute closed-loop forecast
            # Get input
            u_horizon = torch.zeros(b, d, l + l_horizon)
            u_horizon[:, :, 0] = u[:, :, 0]  # supervise y[1], ... , y[L]
            
            # Get matrix
            A = self.shift_matrix.to(self.a.device) + (
                oe.contract('h i, h j -> h j i', 
                            self.p_padding.to(self.a.device), self.a)
            )
            BC = oe.contract('h i, h j -> h i j', self.b, self.c)
            A_BC = A + BC 
            k_horizon = self.get_kernel(u_horizon, c=self.k, 
                                        a=A_BC, b=self.b, l=l+l_horizon)
            k_horizon = repeat(k_horizon, 'nk kd -> (kr nk nh hd) kd', 
                               kr=self.kernel_repeat, nh=self.n_heads,
                               hd=self.head_dim)
            # B x (L + H) x D
            y_horizon = self.fft_conv(u_horizon, k_horizon)
            y_horizon = rearrange(y_horizon, 'b d l -> b l d')
            return y_horizon
        else:
            # Compute open-loop forecast up to L
            k = self.get_kernel(u, c=self.k, a=self.a, b=self.n, l=l)
            k = repeat(k, 'nk kd -> (kr nk nh hd) kd', 
                       kr=self.kernel_repeat, nh=self.n_heads, hd=self.head_dim)
            # B x L x D
            y = self.fft_conv(u, k)
            y = rearrange(y, 'b d l -> b l d')
            return y
        
        
        
    # Alternatively do stuff by first computing hidden-state, then rolling out from there
        
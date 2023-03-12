import torch
import torch.nn.functional as F

from einops import rearrange, repeat

from model.ssm.base import SSM
from model.ssm.preprocess.differencing import get_pascal


class ResidualSSM(SSM):
    """
    Computes both order-N differencing and moving average residuals over input sequence
    """
    def __init__(self, 
                 max_diff_order: int=4, 
                 min_avg_window: int=4, 
                 max_avg_window: int=720,
                 n_kernels: int=8,
                 kernel_repeat: int=16,
                 **kwargs):
        self.max_diff_order = max_diff_order
        self.min_avg_window = min_avg_window
        self.max_avg_window = max_avg_window
        self.n_ma_kernels = (n_kernels - self.max_diff_order) * kernel_repeat
        kwargs['n_heads'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = False
        kwargs['skip_connection'] = False
        # Set kwargs['kernel_repeat'] to number of model n_kernels
        super().__init__(n_kernels=n_kernels, kernel_repeat=kernel_repeat, **kwargs)
        
    def init_weights(self):
        diff_kernel    = repeat(self.init_differencing_weights(), 'nk kd -> (kr nk) kd',
                                kr=self.kernel_repeat)
        ma_r_kernel = self.init_moving_average_weights()  # Shape: (kr x nk) x hd
        self.register('diff_kernel', diff_kernel, trainable=False, lr=None, wd=None)
        self.register('ma_r_kernel', ma_r_kernel, trainable=False, lr=None, wd=None)
        
    def init_differencing_weights(self):
        kernel = torch.zeros(self.max_diff_order, self.max_diff_order).float()
        diff_coeffs = get_pascal(self.max_diff_order, self.max_diff_order).float()
        kernel[:, :self.max_diff_order] += diff_coeffs
        return kernel
    
    def init_moving_average_weights(self):
        ma_window = torch.randint(low=self.min_avg_window,
                                  high=self.max_avg_window,
                                  size=(1, self.n_ma_kernels))
        # Compute moving average kernel 
        max_window = self.max_avg_window
        kernel = torch.zeros(self.n_ma_kernels, max_window)
        kernel[:, 0] = 1.
        
        moving_avg = (1. / torch.clamp(ma_window, min=self.min_avg_window, max=max_window))
        for ix, window in enumerate(ma_window[0]):
            kernel[ix, :window] -= moving_avg[:1, ix]
        return kernel

    def get_kernel(self, u):
        """
        Initialize weights for differencing kernel
        - Assume u is shape B x D x L
        """
        b, d, l = u.shape
        l = max(l, self.diff_kernel.shape[1])
        # Pad kernels to input length
        diff_kernel = F.pad(self.diff_kernel, (0, l - self.diff_kernel.shape[1]), 'constant', 0)
        ma_r_kernel = F.pad(self.ma_r_kernel, (0, l - self.ma_r_kernel.shape[1]), 'constant', 0)
        
        # Combine kernels
        diff_kernel = rearrange(diff_kernel, '(kr nk) kd -> kr nk kd', 
                                kr=self.kernel_repeat)
        ma_r_kernel = rearrange(ma_r_kernel, '(kr nk) kd -> kr nk kd', 
                                kr=self.kernel_repeat)
        
        kernel = torch.cat([diff_kernel, ma_r_kernel], dim=1)
        kernel = repeat(kernel, 'kr nk kd -> (kr nk hd) kd', hd=self.head_dim)
        return kernel
    
    def forward(self, u):
        # Same as base SSM forward, but kernel repeating already taken care of
        u = rearrange(u, 'b l d -> b d l')
        k = self.get_kernel(u)
        y = self.fft_conv(u, k)
        return rearrange(y, 'b d l -> b l d')

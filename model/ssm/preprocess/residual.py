import torch
import torch.nn.functional as F
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
                 **kwargs):
        self.max_diff_order = max_diff_order
        self.min_avg_window = min_avg_window
        self.max_avg_window = max_avg_window
        self.n_ma_kernels = kwargs['n_kernels'] - self.max_diff_order
        kwargs['n_heads'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = False
        kwargs['skip_connection'] = False
        # Set kwargs['kernel_repeat'] to number of model n_kernels
        super().__init__(**kwargs)
        
    def init_weights(self):
        kernel    = self.init_differencing_weights()
        ma_window = self.init_moving_average_weights()
        self.register('kernel', kernel, trainable=False, lr=None, wd=None)
        self.register('ma_window', ma_window, trainable=True, lr=None, wd=None)
        
    def init_differencing_weights(self):
        kernel = torch.zeros(self.n_kernels, self.kernel_dim).float()
        diff_coeffs = get_pascal(self.max_diff_order, self.n_kernels).float()
        kernel[:, :self.max_diff_order] += diff_coeffs
        return kernel
    
    def init_moving_average_weights(self):
        ma_window = torch.randint(low=self.min_avg_window, 
                                  high=self.max_avg_window,  # self.kernel_dim
                                  size=(1, self.n_ma_kernels)).float()
        return ma_window

    def get_kernel(self, u):
        """
        Initialize weights for differencing kernel
        - Assume u is shape B x D x L
        """
        b, d, l = u.shape
        max_window = min(self.max_avg_window, l)
        moving_avg = (1. / torch.clamp(torch.round(self.ma_window), 
                                       min=self.min_avg_window, 
                                       max=max_window).T)
        kernel = F.pad(self.kernel, (0, l - self.kernel_dim, 0, 0), 'constant', 0)
        kernel[self.max_diff_order:self.max_diff_order + self.n_ma_kernels, :max_window] -= moving_avg
        return kernel
    
    def forward(self, u):
        return super().forward(u)
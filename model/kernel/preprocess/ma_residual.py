import torch
import torch.nn.functional as F
from model.kernel.base import Kernel


class MovingAvgResidualKernel(Kernel):
    """
    Computes moving average residuals over input sequence
    """
    def __init__(self, min_avg_window=4, max_avg_window=720, **kwargs):
        self.min_avg_window = min_avg_window
        self.max_avg_window = max_avg_window
        kwargs['n_heads'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = False
        kwargs['skip_connection'] = False
        # Set kwargs['kernel_repeat'] to number of model n_kernels
        super().__init__(**kwargs)
        
    def init_weights(self):
        # Moving average window kernels
        kernel = torch.zeros(self.n_kernels, self.kernel_dim).float()
        kernel[:, 0] = 1.
        
        # Low is a heuristic for now
        ma_window = torch.randint(low=self.min_avg_window, 
                                  high=self.max_avg_window,  # self.kernel_dim
                                  size=(1, self.n_kernels)).float()
        
        self.register('ma_window', ma_window, trainable=True, lr=None, wd=None)
        self.register('kernel', kernel, trainable=False, lr=None, wd=None)
        
    def get_kernel(self, u):
        """
        Initialize weights for differencing kernel
        - Assume u is shape B x D x L
        """
        b, d, l = u.shape
        # Set kernel values s.t. convolution computes residuals
        # from moving average, i.e., y[t] - mean(y[t:t - m])
        kernel = self.kernel - (1. / torch.clamp(torch.round(self.ma_window), 
                                                 min=self.min_avg_window, 
                                                 max=self.kernel_dim).T)
        return F.pad(self.kernel, (0, l-self.kernel_dim, 0, 0), 'constant', 0)
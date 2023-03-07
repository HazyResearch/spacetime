import torch
from .affine import AffineTransform


class MeanTransform(AffineTransform):
    """
    Zero-center values
    """
    def __init__(self, lag):
        super().__init__(lag=lag)
        
    def forward(self, x):
        self.a = torch.ones(1)
        self.b = x[:, :self.lag, :].mean(dim=1)[:, None, :]
        return self.a * x - self.b
    
    
class MeanInputTransform(AffineTransform):
    """
    Same as mean, but compute mean over entire input
    """
    def __init__(self, lag):  # ignore lag here
        super().__init__(lag=None)
        
    def forward(self, x):
        self.a = torch.ones(1)
        self.b = x.mean(dim=1)[:, None, :]  # Same as x[:, :None, :].mean(dim=1)
        return self.a * x - self.b
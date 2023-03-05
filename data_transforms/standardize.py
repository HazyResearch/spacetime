import torch

from .affine import AffineTransform


class StandardizeTransform(AffineTransform):
    """
    Standardize lag terms, i.e., z = (x - mean(x)) / std(x)
    - Computed as (1 / std(x)) * x - mean(x) * (1 / std(x)) to fit with inverse call,
      which does (z + (mean(x) / std(x))) * std(x) = z * std(x) + mean(x)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, x):
        self.a = 1. / torch.std(x[:, :self.lag, :], dim=1)[:, None, :]
        self.b = torch.mean(x[:, :self.lag, :], dim=1)[:, None, :] * self.a
        return self.a * x - self.b

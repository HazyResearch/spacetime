from .affine import AffineTransform


class MeanTransform(AffineTransform):
    """
    Zero-center values
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, x):
        self.a = 1.
        self.b = x[:, :self.lag, :].mean(dim=1)[:, None, :]
        return self.a * x - self.b
    
    
class MeanInputTransform(AffineTransform):
    """
    Same as mean, but compute mean over entire input
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, x):
        self.a = 1.
        self.b = x.mean(dim=1)[:, None, :]  # Same as x[:, :None, :].mean(dim=1)
        return self.a * x - self.b
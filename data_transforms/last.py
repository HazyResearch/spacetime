from .affine import AffineTransform


class LastAffineTransform(AffineTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, x):
        self.a = 1.
        self.b = x[:, self.lag - 1, :][:, None, :]
        return self.a * x - self.b
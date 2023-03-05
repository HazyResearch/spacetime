import torch.nn as nn


class AffineTransform(nn.Module):
    def __init__(self, lag=None):
        """
        Transform data: f(x) = ax - b  
        """
        super().__init__()
        self.lag = lag
        
    def forward(self, x):
        # Assume x.shape is B x L x D
        raise NotImplementedError
    
    
class InverseAffineTransform(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform  # AffineTransform object
        
    def forward(self, x):
        try:  # Hacky, but catches different device tensors. Should clean up.
            return (x + self.transform.b.to(x.device)) / self.transform.a
        except:
            try:
                return (x + self.transform.b) / self.transform.a.to(x.device)
            except:
                breakpoint()
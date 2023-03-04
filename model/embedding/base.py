import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int):
        """
        Generic class for encoding 
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.initialize_layers()
        
    def initialize_layers(self):
        self.layers = nn.Identity()
        
    def forward(self, x):
        return self.layers(x)
    
    

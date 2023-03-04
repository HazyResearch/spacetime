import torch.nn as nn
from .base import Embedding


class LinearEmbedding(Embedding):
    def __init__(self, input_dim, embedding_dim):
        super().__init__(input_dim, embedding_dim)
        
    def initialize_layers(self):  
        self.layers = nn.Linear(self.input_dim, self.embedding_dim) 
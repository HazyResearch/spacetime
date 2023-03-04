from einops import repeat
from .base import Embedding
    

class RepeatEmbedding(Embedding):
    def __init__(self, 
                 input_dim: int, 
                 embedding_dim: int=None, 
                 n_heads: int=None,
                 n_kernels: int=None):

        if embedding_dim is None:
            try:
                embedding_dim = input_dim * n_heads * n_kernels
            except Exception as e:
                raise e('If embedding_dim not specified, must specify n_kernels and n_heads')
        else:
            assert embedding_dim % input_dim == 0, 'Embedding_dim should be multiple of input_dim'
        
        super().__init__(input_dim, embedding_dim)
        
    def repeat(self, x):
        return repeat(x, 'b l d -> b l (r d)', 
                      r=self.embedding_dim // self.input_dim)
        
    def initialize_layers(self):  
        self.layers = self.repeat
        
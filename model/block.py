"""
SpaceTime blocks, stacked into encoder and decoder of architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components import OurModule
from model.mlp import init_mlp
from model.ssm import init_ssm
from model.ssm.preprocess import init_preprocess_ssm as init_pre


class Block(OurModule):
    """
    Standard encoder block
    """
    def __init__(self, 
                 input_dim: int,
                 pre_config: str=None,
                 ssm_config: str=None,
                 mlp_config: str=None,
                 skip_connection: bool=False,
                 skip_preprocess: bool=False):
        super().__init__()
        self.input_dim = input_dim
        self.skip_connection = skip_connection
        self.skip_preprocess = skip_preprocess
        
        self.pre = init_pre(pre_config)
        self.ssm = init_ssm(ssm_config)
        self.mlp = init_mlp(mlp_config)
            
    def forward(self, u):
        """
        Input shape: B x L x D
        """
        z = self.pre(u)
        y = self.ssm(z)
        y = self.mlp(y)
        if self.skip_connection and self.skip_preprocess:
            return y + u  # Also skip preprocessing step
        elif self.skip_connection:
            return y + z
        else:
            return y
    
    
class ClosedLoopBlock(Block):
    """
    Block with a closed-loop SSM. 
    
    In SpaceTime, we only consider using one ClosedLoopBlock 
    as the last-layer in a single-layer decoder. 
    
    However, other architectures can also be explored, e.g., 
    having more "open" blocks on top of the ClosedLoopBlock 
    in a multi-layer decoder.
    """
    def __init__(self, **kwargs):
        kwargs['skip_connection'] = False
        super().__init__(**kwargs)
        
    def forward(self, u):
        z = self.pre(u)
        # Computes layer outputs and next-time-step layer inputs
        y, u_next = self.ssm(z)  
        # Return both layer outputs and prediction + "ground-truth"
        # for next-time-step layer inputs
        return y, (u_next, u)    
    
    
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = self.init_blocks(config)
        
    def init_blocks(self, config):
        blocks = []
        for block in config['blocks']:
            blocks.append(Block(**block))
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)
    
    
class Decoder(nn.Module):
    """
    In SpaceTime, we only consider using one ClosedLoopBlock 
    as the last-layer in a single-layer decoder. 
    
    However, other architectures can also be explored, e.g., 
    having more "open" blocks on top of the ClosedLoopBlock 
    in a multi-layer decoder.
    
    In future, can refactor this class to be more general 
    and support multiple layers. (p easy, just weirdness with
    nn.Sequential and multiple outputs)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = self.init_blocks(config)
        
    def init_blocks(self, config):
        return ClosedLoopBlock(**config['blocks'][0])
    
    def forward(self, x):
        return self.blocks(x)  # y, (u_next, u)
 
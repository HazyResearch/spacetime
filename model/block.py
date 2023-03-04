"""
SpaceTime blocks, backbone behind the architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp import init_mlp
from model.module import OurModule
from model.attention import HedgehogCrossAttention, HedgehogSelfAttention


class Block(OurModule):
    def __init__(self, 
                 input_dim: int,
                 conv_attn: str, 
                 conv_pool: str=None,
                 conv_config: str=None,
                 mlp_config: str=None,
                 conv: dict=None,
                 mlp: dict=None):
        super().__init__()
        self.input_dim = input_dim
        self.conv_attn = conv_attn
        self.conv_pool = conv_pool  # Whether to pool conv output
        
        self.conv = self.init_conv(conv_attn, conv)
        self.mlp  = init_mlp(mlp)
        
    def init_conv(self, conv_attn, conv):
        # Pooling
        if self.conv_pool == 'cls':  # As in BERT, summarize sequence into a "token" embedding
            cls = torch.randn(1, 1, self.input_dim)
            self.register("cls", cls, trainable=True, lr=None, wd=None)
            self.pool = lambda x: x[:, -1:, :]  # cls token is last
        elif self.conv_pool == 'mean':  # Alternatively do length-wise avg pooling
            self.cls = None
            self.pool = lambda x: x.mean(dim=1, keepdim=True)
        else:
            self.cls = None
            self.pool = lambda x: x
            
        if conv_attn == 'identity' or conv is None:
            return nn.Identity()
        elif conv_attn == 'cross':
            return HedgehogCrossAttention(**conv)
        elif conv_attn == 'self':
            return HedgehogSelfAttention(**conv)
        else:
            raise NotImplementedError(f'conv_attn {conv_attn} not implemented')
            
    def forward(self, x_tuple):
        """
        Input shape: tuple of B x L x D
        Output shape:
        - if self.conv_pool == 'cls' or 'mean', outputs B x 1 x D
        - else, outputs B x L x D
        """
        b, l, d = x_tuple[-1].shape

        if self.conv_attn == 'self' and self.conv_pool == 'cls':
            x = torch.cat([x_tuple[0], self.cls.expand(b, -1, -1)], dim=1)
            x = self.conv((x,))
            
        elif self.conv_attn == 'self':  # and self.conv_pool == 'mean':
            x = x_tuple[0]
            # print(f'(block) --> x.shape:', x.shape)
            x = self.conv((x,))
            
        elif self.conv_attn == 'cross' and self.conv_pool == 'cls':
            x = x_tuple[0]
            x = self.conv((self.cls.expand(b, -1, -1), x, x))
            
        elif self.conv_attn == 'cross':  # and self.conv_pool == 'mean':
            x = self.conv(x_tuple)
            
        else:  # identity
            x = self.conv(x_tuple[0])
            
        x = self.pool(x)
        x = self.mlp(x)
        return (x,)  # cross-attention input flexibility
    

class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = self.init_blocks(config)
        self.input_pad = config['input_pad']
        
    def init_blocks(self, config):
        blocks = []
        for block in config['blocks']:
            blocks.append(Block(**block))
        return nn.Sequential(*blocks)
    
    def forward(self, x_tuple):
        p3d = (0, 0, self.input_pad, 0, 0, 0)
        x_tuple = [F.pad(x, p3d, mode='constant', value=0) for x in x_tuple]
        # print('(backbone) --> len(x_tuple):', len(x_tuple))
        return self.blocks(x_tuple)
    
class Encoder(Backbone):
    def __init__(self, config):
        super().__init__(config=config)
        
class Decoder(Backbone):
    def __init__(self, config):
        super().__init__(config=config) 
 
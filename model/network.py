"""
SpaceTime Network
"""
import torch.nn as nn

from model.embedding import init_embedding
from model.block import Encoder, Decoder
from model.mlp import init_mlp


class SpaceTime(nn.Module):
    def __init__(self,
                 embedding_config: dict,
                 encoder_config: dict,
                 decoder_config: dict,
                 output_config: dict,
                 inference_only: bool=False,
                 lag: int=1,
                 horizon: int=1):
        super().__init__()
        
        self.embedding_config  = embedding_config
        self.encoder_config    = encoder_config
        self.decoder_config    = decoder_config
        self.output_config     = output_config
        
        self.inference_only = inference_only
        self.lag     = lag
        self.horizon = horizon
        
        self.init_weights(embedding_config, encoder_config,
                          decoder_config, output_config)
        
    # -----------------
    # Initialize things
    # -----------------
    def init_weights(self, 
                     embedding_config: dict, 
                     encoder_config: dict, 
                     decoder_config: dict, 
                     output_config: dict):
        self.embedding  = self.init_embedding(embedding_config)
        self.encoder    = self.init_encoder(encoder_config)
        self.decoder    = self.init_decoder(decoder_config)
        self.output     = self.init_output(output_config)
        
    def init_embedding(self, config):
        return init_embedding(config)
    
    def init_encoder(self, config):
        self.encoder = Encoder(config)
        # Allow access to first encoder SSM kernel_dim
        self.kernel_dim = self.encoder.blocks[0].ssm.kernel_dim
        return self.encoder
    
    def init_decoder(self, config):
        self.decoder = Decoder(config)
        self.decoder.blocks.ssm.lag = self.lag
        self.decoder.blocks.ssm.horizon = self.horizon
        return self.decoder
    
    def init_output(self, config):
        return init_mlp(config)
    
    # -------------
    # Toggle things
    # -------------
    def set_inference_only(self, mode=False):
        self.inference_only = mode
        self.decoder.blocks.ssm.inference_only = mode
        
    def set_closed_loop(self, mode=True):
        self.decoder.blocks.ssm.closed_loop = mode
        
    def set_train(self):
        self.train()
        
    def set_eval(self):
        self.eval()
        self.set_inference_only(mode=True)
        
    def set_lag(self, lag: int):
        decoder.blocks.ssm.lag = lag
        
    def set_horizon(self, horizon: int):
        decoder.blocks.ssm.horizon = horizon
        
    # ------------
    # Forward pass
    # ------------
    def forward(self, u):
        self.set_closed_loop(True)
        # Assume u.shape is (batch x len x dim), 
        # where len = lag + horizon
        z = self.embedding(u)
        # print('z = self.embedding(u)')
        # breakpoint()
        z = self.encoder(z)
        # print('z = self.encoder(z)')
        # breakpoint()
        y_c, z_u = self.decoder(z)  # closed-loop
        # print('y_c, z_u = self.decoder(z)')
        # breakpoint()
        y_c = self.output(y_c)
        # print('y_c = self.output(y_c)')
        # breakpoint()
        
        z_u_pred, z_u_true = z_u
        if not self.inference_only:  
            # Also compute outputs via open-loop
            self.set_closed_loop(False)
            y_o, _ = self.decoder(z)
            y_o = self.output(y_o)
        else:
            y_o = None
        # Return (model outputs), (model last-layer next-step inputs)
        return (y_c, y_o), (z_u_pred, z_u_true)

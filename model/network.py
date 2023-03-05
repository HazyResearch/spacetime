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
                          decoder_config, output_config,
                          position_config)
        
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
        return Encoder(config)
    
    def init_decoder(self, config):
        decoder = Decoder(config)
        decoder.blocks.ssm.lag = self.lag
        decoder.blocks.ssm.horizon = self.horizon
    
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
        z = self.encoder(z)
        y_c, z_u = self.decoder(z)  # closed-loop
        z_u_pred, z_u_true = z_u
        if not self.inference_only:  
            # Also compute outputs via open-loop
            self.set_closed_loop(False)
            y_o, _ = self.decoder(z) 
        # Return (model outputs), (model last-layer next-step inputs)
        return (y_c, y_o), (z_u_pred, z_u_true)

"""
SpaceTime Network
"""
import torch.nn as nn


from embedding import get_embedding


class SpaceTime(nn.Module):
    def __init__(self,
                 embedding_config: dict,
                 preprocess_config: dict,
                 encoder_config: dict,
                 decoder_config: dict,
                 output_config: dict,
                 lag=1,
                 horizon=1,
                 noise_variance=0.):
        super().__init__()
        
        self.embedding_config  = embedding_config
        self.preprocess_config = preprocess_config
        self.encoder_config    = encoder_config
        self.decoder_config    = decoder_config
        self.output_config     = output_config
        
        self.init_weights(embedding_config, encoder_config,
                          decoder_config, output_config,
                          position_config)
        
    # -----------------
    # Initialize things
    # -----------------
    def init_weights(self, 
                     embedding_config: dict, 
                     preprocess_config: dict,
                     encoder_config: dict, 
                     decoder_config: dict, 
                     output_config: dict):
        self.embedding  = self.init_embedding(embedding_config)
        self.preprocess = self.init_preprocess(preprocess_config)
        self.encoder    = self.init_encoder(encoder_config)
        self.decoder    = self.init_decoder(decoder_config)
        self.output     = self.init_output(output_config)
        
    def init_embedding(self, config):
        return get_embedding(config)

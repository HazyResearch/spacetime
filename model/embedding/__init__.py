from base import Embedding
from linear import LinearEmbedding
from repeat import RepeatEmbedding


def get_embedding(config):
    methods = ['linear', 'identity', 'repeat']
    if config['method'] == 'linear':
        return LinearEmbedding(**config['kwargs'])
    elif config['method'] == 'repeat':
        return RepeatEmbedding(**config['kwargs'])
    elif config['method'] == 'identity' or method is None:
        return Embedding(**config['kwargs'])
    else:
        raise NotImplementedError(f'Embedding method {method} not implemented. Please select among {methods}')
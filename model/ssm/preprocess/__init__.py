import torch.nn as nn

from .differencing import DifferencingSSM
from .ma_residual import MovingAvgResidualSSM
from .residual import ResidualSSM


def init_preprocess_ssm(config):
    if config['method'] == 'differencing':
        ssm = DifferencingSSM
    elif config['method'] == 'ma_residual':
        ssm = MovingAvgResidualSSM
    elif config['method'] == 'residual':
        ssm = ResidualSSM
    elif config['method'] in ['identity', None]:
        return nn.Identity()
    else:
        raise NotImplementedError(f"Preprocessing config method {config['method']} not implemented!")
    return ssm(**config['kwargs'])
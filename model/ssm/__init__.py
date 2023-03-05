from .companion import CompanionSSM
from .shift import ShiftSSM
from .closed_loop import ClosedLoopCompanionSSM


def init_ssm(config):
    supported_methods = ['shift', 'companion', 'closed_loop_companion']
    if config['method'] == 'shift':
        ssm = ShiftSSM
    elif config['method'] == 'companion':
        ssm = CompanionSSM
    elif config['method'] == 'closed_loop_companion':
        ssm = ClosedLoopCompanionSSM
    else:
        raise NotImplementedError(f"SSM config method {config['method']} not implemented! \
                                    Please choose from {supported_methods}")
    return ssm(**config['kwargs'])
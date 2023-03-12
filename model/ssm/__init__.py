from .companion import CompanionSSM
from .shift import ShiftSSM
from .closed_loop import ClosedLoopCompanionSSM, ClosedLoopShiftSSM


def init_ssm(config):
    supported_methods = ['companion', 'closed_loop_companion',
                         'shift', 'closed_loop_shift']
    if config['method'] == 'companion':
        ssm = CompanionSSM
    elif config['method'] == 'closed_loop_companion':
        ssm = ClosedLoopCompanionSSM
    elif config['method'] == 'shift':
        ssm = ShiftSSM
    elif config['method'] == 'closed_loop_shift':
        ssm = ClosedLoopShiftSSM
    else:
        raise NotImplementedError(
            f"SSM config method {config['method']} not implemented! Please choose from {supported_methods}")
    return ssm(**config['kwargs'])
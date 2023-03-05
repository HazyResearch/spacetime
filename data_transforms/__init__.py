from .mean import MeanTransform, MeanInputTransform
from .standardisze import StandardizeTransform
from .affine import InverseAffineTransform
from .last import LastAffineTransform


def get_data_transforms(lag):
    supported_methods = ['mean', 'mean_input', 'last', 
                         'standardize', 'none']
    if method == 'mean':
        input_transform = MeanTransform(lag)
        output_transform = InverseAffineTransform(input_transform)
    elif method == 'mean_input':
        input_transform = MeanInputTransform(lag)
        output_transform = InverseAffineTransform(input_transform)
    elif method == 'last':
        input_transform = LastAffineTransform(lag)
        output_transform = InverseAffineTransform(input_transform)   
    elif method == 'standardize':
        input_transform = StandardizeTransform(lag)
        output_transform = InverseAffineTransform(input_transform)   
    elif method == 'none':
        input_transform = lambda x: x
        output_transform = lambda x: x
    else:
        raise NotImplementedError(f"Data transform method '{method}' not supported. Please choose from {supported_methods}."
        
    return input_transform, output_transform
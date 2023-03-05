import torch
import torch.nn.functional as F
from model.ssm.base import SSM


class DifferencingSSM(SSM):
    """
    Computes order-N differencing over input sequence
    """
    def __init__(self, max_diff_order=4, **kwargs):
        self.max_diff_order = max_diff_order
        kwargs['n_heads'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = False
        kwargs['skip_connection'] = False
        # Set kwargs['kernel_repeat'] to number of model n_kernels
        super().__init__(**kwargs)
        
    def init_weights(self):
        kernel = torch.zeros(self.n_kernels, self.kernel_dim).float()
        # Hard-coded up to 4 orders, but just the binomial coeffs / Pascal's triangle (with negatives)
        diff_coeffs = get_pascal(self.max_diff_order)
        # Could be slow, but just done once at initialization
        for ix in range(self.n_kernels):
            try:
                kernel[ix, :self.max_diff_order] += diff_coeffs[ix % len(diff_coeffs)].float()
            except:
                breakpoint()
        self.register('kernel', kernel, trainable=False, lr=None, wd=None)
    
    def get_kernel(self, u):
        """
        Initialize weights for differencing kernel
        - Assume u is shape B x D x L
        """
        b, d, l = u.shape
        return F.pad(self.kernel, (0, l-self.kernel_dim, 0, 0), 'constant', 0)
    
    
def get_pascal(n, total_rows=None):
    total_rows = n if total_rows is None else total_rows
    # Compute binomial coeffs for all rows up to n
    line = torch.zeros(total_rows, n).float()
    line[:, 0] = 1.
    for j in range(1, n):      # For all rows,
        for k in range(0, j):  # Compute C(j, k)
            # Coefficients are binomial coeffs, 
            # C(n, k + 1) = C(n, k) * (n - k) / (k + 1)
            negate = 2 * k % 2 - 1  # Negate even elements
            line[j][k+1] += (line[j][k] * (j - k) / (k + 1)) * negate
    return line
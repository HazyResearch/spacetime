import math

import torch
import torch.nn.functional as F

from einops import rearrange, reduce


def companion_from_p(p):
    """
    Arguments:
        p: (..., d)
    Return:
        A: (..., d, d)
    """
    batch_size, d = p.shape[:-1], p.shape[-1]
    A = torch.zeros(*batch_size, d, d, dtype=p.dtype, device=p.device)
    A[..., 1:, :-1] = torch.eye(d - 1, dtype=p.dtype, device=p.device)
    A[..., -1] = p
    return A


def companion_krylov(L, p, b, c=None, c_tilde=None):
    """
    Compute the Krylov matrix (c^T b, c^T A b, c^T A^2 b, ...), where A = shift + p e_d^T.
    Arguments:
        p: (..., d), real
        b: (..., d), real
        c: (..., d), real. One can instead supply c_tilde (below).
        c_tilde: (..., d), real, where c_tilde = c^T (I - A^L)
    At least c or c_tilde must be supplied.
    """
    d = p.shape[-1]
    batch_size = p.shape[:-1]
    e_d = torch.zeros(*batch_size, d, device=p.device, dtype=p.dtype)
    e_d[..., -1] = 1.0
    assert e_d.shape == p.shape
    assert b.shape == p.shape
    if c_tilde is None:
        assert c is not None, 'at least c or c_tilde must be supplied'
        assert c.shape == p.shape
        A = companion_from_p(p)
        c_tilde = c - torch.einsum('...m,...mn->...n', c, torch.linalg.matrix_power(A, L).to(dtype=c.dtype))
    else:
        assert c_tilde.shape == p.shape

    def fft_conv(u, v):  # This is actually convolution and not cross-correlation
        d = u.shape[-1]
        u_f = torch.fft.rfft(u, n=2 * d)
        v_f = torch.fft.rfft(v, n=2 * d)
        return torch.fft.irfft(u_f * v_f.conj(), n=2 * d)[..., :d]

    def quadratic_form(u, v):
        d_rounded = math.ceil(d / L) * L
        # The reduce is to deal with the case where d > L
        return torch.fft.rfft(reduce(F.pad(fft_conv(u, v), (0, d_rounded - d)),
                                     '... (m L) -> ... L', L=L, reduction='sum'), n=L)


    Zconj = torch.exp(1j * 2 * math.pi * torch.arange(L // 2 + 1, dtype=torch.float32, device=p.device) / L)
    # woodbury = quadratic_form(c_tilde, b) + quadratic_form(c_tilde, p) * quadratic_form(e_d, b) / (Zconj - quadratic_form(e_d, p))
    quad = quadratic_form(rearrange(torch.stack([c_tilde, e_d], dim=-2), '... two d -> ... two 1 d'),
                          rearrange(torch.stack([b, p], dim=-2), '... two d -> ... 1 two d'))
    woodbury = quad[..., 0, 0, :] + quad[..., 0, 1, :] * quad[..., 1, 0, :] / (Zconj - quad[..., 1, 1, :])
    woodbury_irfft = torch.fft.irfft(woodbury, n=L)
    return woodbury_irfft


if __name__ == '__main__':
    torch.manual_seed(0)
    d = 25
    L = 9
    H = 2
    p = torch.randn(H, d)
    p /= torch.linalg.norm(p, ord=1, dim=-1, keepdim=True)
    b = torch.randn(H, d)
    c = torch.randn(H, d)

    A = companion_from_p(p)

    from src.ops.krylov import krylov
    K = krylov(L, A, b, c)
    K_fast = companion_krylov(L, p, b, c=c)
    print((K - K_fast).abs().max())

    from benchmarks.utils import benchmark_all

    torch.manual_seed(0)
    d = 512
    L = 1024
    H = 256
    p = torch.randn(H, d, device='cuda', requires_grad=True)
    p = p / torch.linalg.norm(p, ord=1, dim=-1, keepdim=True)
    b = torch.randn(H, d, device='cuda', requires_grad=True)
    c = torch.randn(H, d, device='cuda', requires_grad=True)
    A = companion_from_p(p)

    benchmark_all(krylov, L, A, b, c, desc='krylov')
    benchmark_all(companion_krylov, L, p, b, c, desc='companion fast krylov')
    benchmark_all(companion_krylov, L, p, b, c_tilde=c, desc='companion fast krylov c_tilde')
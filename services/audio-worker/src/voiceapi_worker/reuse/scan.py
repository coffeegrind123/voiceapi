"""Selective scan (Mamba S6) backends.

``selective_scan_cuda`` is the compiled kernel shipped in the prebuilt
``mamba_ssm`` wheel (imported directly, so the package's Mamba2/Mamba3 imports
and their extra dependencies never load). When it is missing (CPU tests, a
mismatched wheel) the reference recurrence from
``mamba_ssm.ops.selective_scan_interface.selective_scan_ref`` runs instead,
batched in slices to bound memory. Both compute, per channel d and state n::

    delta = softplus(delta + delta_bias)
    x_t   = exp(delta_t * A) * x_{t-1} + delta_t * B_t * u_t
    y_t   = <x_t, C_t> + D * u_t
    out   = y * silu(z)
"""

from __future__ import annotations

import logging
from enum import StrEnum

import torch
import torch.nn.functional as F

log = logging.getLogger("voiceapi.worker.scan")

try:  # pragma: no cover - depends on the GPU image
    import selective_scan_cuda  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    selective_scan_cuda = None

# Elements of the [batch, dim, length, state] intermediates per reference slice (~256 MiB fp32 each).
_REFERENCE_SLICE_ELEMENTS = 64 * 1024 * 1024


class ScanBackend(StrEnum):
    CUDA = "cuda_kernel"
    REFERENCE = "reference"


def available_backend(device: torch.device) -> ScanBackend:
    if device.type == "cuda" and selective_scan_cuda is not None:
        return ScanBackend.CUDA
    return ScanBackend.REFERENCE


def _reference(u, delta, A, B, C, D, z, delta_bias) -> torch.Tensor:  # noqa: PLR0917 - selective_scan_fn order
    dtype_in = u.dtype
    u = u.float()
    delta = F.softplus(delta.float() + delta_bias[..., None].float())
    B = B.float()
    C = C.float()
    batch, dim, length = u.shape
    state = A.shape[1]
    x = A.new_zeros((batch, dim, state))
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    ys = torch.empty((batch, dim, length), dtype=torch.float32, device=u.device)
    for index in range(length):
        x = delta_a[:, :, index] * x + delta_b_u[:, :, index]
        ys[:, :, index] = torch.einsum("bdn,bn->bd", x, C[:, :, index])
    out = ys + u * D[:, None]
    return (out * F.silu(z.float())).to(dtype_in)


def selective_scan(  # noqa: PLR0917 - mirrors mamba_ssm's selective_scan_fn argument order
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    delta_bias: torch.Tensor,
) -> torch.Tensor:
    """u, delta, z: [b, d, l]; A: [d, n] fp32; B, C: [b, n, l]; D, delta_bias: [d]."""
    if available_backend(u.device) is ScanBackend.CUDA:
        out, _x, *rest = selective_scan_cuda.fwd(
            u.contiguous(), delta.contiguous(), A.contiguous(), B.unsqueeze(1).contiguous(),
            C.unsqueeze(1).contiguous(), D.contiguous(), z.contiguous(), delta_bias.contiguous(), True,
        )
        return rest[0]
    batch, dim, length = u.shape
    per_row = dim * length * A.shape[1]
    step = max(1, _REFERENCE_SLICE_ELEMENTS // max(per_row, 1))
    if step >= batch:
        return _reference(u, delta, A, B, C, D, z, delta_bias)
    return torch.cat(
        [
            _reference(u[i : i + step], delta[i : i + step], A, B[i : i + step], C[i : i + step], D,
                       z[i : i + step], delta_bias)
            for i in range(0, batch, step)
        ],
        dim=0,
    )


__all__ = ["ScanBackend", "available_backend", "selective_scan"]

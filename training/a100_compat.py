"""A100 runtime compatibility patches for Unsloth training.

Import this module BEFORE importing unsloth in any training script.
On A100 GPUs, PyTorch autocast defaults to fp16 but Unsloth assumes bf16.
Unsloth's custom kernels use in-place addmm_() which bypasses autocast,
causing dtype mismatches in both forward and backward passes.

This module monkey-patches torch.Tensor.addmm_ to auto-cast operands
to the target tensor's dtype. The .to(dt) call is a no-op when dtypes
already match, so there is zero overhead on H100/other GPUs.

Usage:
    import a100_compat  # noqa: F401  - must be before unsloth imports
"""

import torch

_orig_addmm_ = torch.Tensor.addmm_


def _safe_addmm_(self, mat1, mat2, *, beta=1, alpha=1):
    """addmm_ with automatic dtype alignment of all operands."""
    dt = self.dtype
    return _orig_addmm_(self, mat1.to(dt), mat2.to(dt), beta=beta, alpha=alpha)


torch.Tensor.addmm_ = _safe_addmm_

print("[a100_compat] Patched torch.Tensor.addmm_ for A100 dtype safety")

#!/usr/bin/env python3
"""A100 GPU compatibility patches for Unsloth and transformers.

Patches installed package source files on disk to fix:
1. Unsloth matmul_lora dtype mismatch (fp16 vs bf16 in autocast)
2. Qwen3.5 compute_3d_position_ids empty delta tensor (transformers 5.2.0 bug)

Usage: python training/patch_a100.py
"""
import os
import re
import sys


def _clear_pycache(fpath, name_fragment):
    """Remove .pyc files so Python picks up the patched source."""
    pycache_dir = os.path.join(os.path.dirname(fpath), "__pycache__")
    if os.path.isdir(pycache_dir):
        for f in os.listdir(pycache_dir):
            if name_fragment in f and f.endswith((".pyc", ".pyo")):
                os.remove(os.path.join(pycache_dir, f))


def patch_matmul_lora():
    """Fix Unsloth matmul_lora: cast all addmm_ operands to out.dtype.

    On A100, PyTorch autocast defaults to fp16, but Unsloth's addmm_ call
    mixes fp16 (XA from autocast) with bf16 (B.to(dtype) from model config).
    The in-place addmm_ bypasses autocast, so we must align dtypes manually.
    """
    try:
        import unsloth.kernels.utils as _mod
        fpath = _mod.__file__
    except ImportError:
        print("[patch_a100] unsloth not installed — skipping matmul_lora")
        return False

    with open(fpath, "r") as f:
        content = f.read()

    patched_marker = "out.addmm_(XA.to(_dt), B.to(_dt)"
    if patched_marker in content:
        print("[patch_a100] matmul_lora — already patched")
        return True

    # Match with flexible whitespace
    pattern = r"out\.addmm_\(XA,\s*B\.to\(dtype\),\s*alpha\s*=\s*s\)"
    replacement = "_dt = out.dtype; out.addmm_(XA.to(_dt), B.to(_dt), alpha = s)"

    new_content, n = re.subn(pattern, replacement, content)
    if n == 0:
        print(f"[patch_a100] WARNING: matmul_lora pattern not found in {fpath}")
        # Dump context around line 1043 for debugging
        lines = content.split("\n")
        for i in range(max(0, 1038), min(len(lines), 1048)):
            print(f"  L{i+1}: {lines[i]}")
        return False

    with open(fpath, "w") as f:
        f.write(new_content)

    _clear_pycache(fpath, "utils")
    print(f"[patch_a100] Patched matmul_lora ({n} occurrence(s)) in {fpath}")
    return True


def patch_compute_3d_position_ids():
    """Fix Qwen3.5 compute_3d_position_ids for text-only inputs.

    transformers 5.2.0 bug: when no vision tokens exist, the delta tensor
    has shape (batch, 0) but position_ids has shape (batch, seq_len).
    Broadcasting fails. Fix: skip delta addition when delta is empty.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as _mod
        fpath = _mod.__file__
    except ImportError:
        print("[patch_a100] transformers qwen3_5 not available — skipping")
        return False

    with open(fpath, "r") as f:
        content = f.read()

    patched_marker = "if delta.numel() > 0 else position_ids"
    if patched_marker in content:
        print("[patch_a100] compute_3d_position_ids — already patched")
        return True

    old = "position_ids = position_ids + delta.to(device=position_ids.device)"
    new = "position_ids = position_ids + delta.to(device=position_ids.device) if delta.numel() > 0 else position_ids"

    if old not in content:
        # Try with flexible whitespace
        pattern = r"position_ids\s*=\s*position_ids\s*\+\s*delta\.to\(device\s*=\s*position_ids\.device\)"
        match = re.search(pattern, content)
        if match:
            content = content[:match.start()] + new + content[match.end():]
        else:
            print(f"[patch_a100] WARNING: compute_3d_position_ids pattern not found in {fpath}")
            return False
    else:
        content = content.replace(old, new)

    with open(fpath, "w") as f:
        f.write(content)

    _clear_pycache(fpath, "modeling_qwen3_5")
    print(f"[patch_a100] Patched compute_3d_position_ids in {fpath}")
    return True


def main():
    print("=" * 50)
    print("A100 Compatibility Patches")
    print("=" * 50)

    results = []
    results.append(("matmul_lora", patch_matmul_lora()))
    results.append(("compute_3d_position_ids", patch_compute_3d_position_ids()))

    print("-" * 50)
    for name, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
    print("=" * 50)

    if not all(ok for _, ok in results):
        print("WARNING: Some patches failed — check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()

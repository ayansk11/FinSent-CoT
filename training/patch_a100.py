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


def _find_package_file(package_name, subpath):
    """Find a file inside an installed package without importing it.

    Uses importlib.util.find_spec on the top-level package (safe even when
    submodules have syntax errors) to locate site-packages, then joins subpath.
    """
    import importlib.util
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        return None
    pkg_dir = spec.submodule_search_locations[0]
    fpath = os.path.join(pkg_dir, *subpath.split("/"))
    return fpath if os.path.isfile(fpath) else None


def patch_matmul_lora():
    """Fix Unsloth matmul_lora: cast all addmm_ operands to out.dtype.

    On A100, PyTorch autocast defaults to fp16, but Unsloth's addmm_ call
    mixes fp16 (XA from autocast) with bf16 (B.to(dtype) from model config).
    The in-place addmm_ bypasses autocast, so we must align dtypes manually.
    """
    fpath = _find_package_file("unsloth", "kernels/utils.py")
    if fpath is None:
        print("[patch_a100] unsloth not installed - skipping matmul_lora")
        return False

    with open(fpath, "r") as f:
        content = f.read()

    patched_marker = "out.addmm_(XA.to(_dt), B.to(_dt)"
    if patched_marker in content:
        print("[patch_a100] matmul_lora - already patched")
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
        print("[patch_a100] transformers qwen3_5 not available - skipping")
        return False

    with open(fpath, "r") as f:
        content = f.read()

    patched_marker = "if delta.numel() > 0 else position_ids"
    if patched_marker in content:
        print("[patch_a100] compute_3d_position_ids - already patched")
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


def patch_fast_lora_backward():
    """Fix Unsloth fast_lora backward pass: torch.matmul(..., out=X) dtype mismatch.

    On A100, autocast produces fp16 intermediates but saved tensors from forward
    pass are float32. The backward pass uses torch.matmul(..., out=X) where X is
    a saved float32 tensor but the matmul result is fp16. PyTorch refuses to write
    fp16 into a float32 out tensor.

    Fix: replace all `out = X if ctx.inplace else None` with `out = None` to
    disable the inplace optimization. Performance impact is negligible (one extra
    tensor allocation per backward call).

    NOTE: We find fast_lora.py by path (not import) because a previous corrupted
    patch may have left the file with a SyntaxError, making import impossible.
    """
    fpath = _find_package_file("unsloth", "kernels/fast_lora.py")
    if fpath is None:
        print("[patch_a100] unsloth.kernels.fast_lora not found - skipping")
        return False

    with open(fpath, "r") as f:
        content = f.read()

    # Repair damage from Round 4 (inline comment inside function call args
    # turned: torch.matmul(..., out = None  # comment)  ← ) is commented out)
    broken_marker = "# A100_PATCH: disable inplace matmul in backward"
    if broken_marker in content:
        content = content.replace("out = None  " + broken_marker, "out = None")
        with open(fpath, "w") as f:
            f.write(content)
        _clear_pycache(fpath, "fast_lora")
        print("[patch_a100] Repaired corrupted fast_lora.py from previous patch attempt")

    # Check if already patched (original inplace pattern no longer present)
    inplace_pattern = r"out\s*=\s*\w+\s+if\s+ctx\.inplace\s+else\s+None"
    if not re.search(inplace_pattern, content):
        print("[patch_a100] fast_lora backward - already patched (no inplace pattern found)")
        return True

    # Replace: out = X if ctx.inplace else None  →  out = None
    # No inline comment - the pattern appears inside function calls like
    # torch.matmul(..., out = X if ctx.inplace else None) and a # comment
    # would swallow the closing paren.
    new_content, n = re.subn(inplace_pattern, "out = None", content)
    if n == 0:
        print(f"[patch_a100] WARNING: fast_lora inplace pattern not found in {fpath}")
        return False

    with open(fpath, "w") as f:
        f.write(new_content)

    _clear_pycache(fpath, "fast_lora")
    print(f"[patch_a100] Patched fast_lora backward ({n} occurrence(s)) in {fpath}")
    return True


def patch_llama_cpp_install():
    """No longer needed - GGUF export now uses llama.cpp directly via subprocess."""
    print("[patch_a100] install_llama_cpp - skipped (GGUF export uses llama.cpp directly)")
    return True


def main():
    print("=" * 50)
    print("A100 Compatibility Patches")
    print("=" * 50)

    results = []
    # fast_lora_backward MUST run first - it repairs corrupted fast_lora.py
    # which would block all subsequent unsloth imports via SyntaxError
    results.append(("fast_lora_backward", patch_fast_lora_backward()))
    results.append(("matmul_lora", patch_matmul_lora()))
    results.append(("compute_3d_position_ids", patch_compute_3d_position_ids()))
    results.append(("install_llama_cpp", patch_llama_cpp_install()))

    # install_llama_cpp is optional - SLURM scripts build llama.cpp separately
    # All patches are optional - Unsloth patches fail harmlessly for MobileLLM (PEFT+BnB)
    optional = {"install_llama_cpp", "fast_lora_backward", "matmul_lora", "compute_3d_position_ids"}

    print("-" * 50)
    for name, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
    print("=" * 50)

    critical_failed = [name for name, ok in results if not ok and name not in optional]
    if critical_failed:
        print(f"CRITICAL patches failed: {', '.join(critical_failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Patch Unsloth's compiled GRPOTrainer cache to fix tensor mismatch bug.

Bug: masked_batch_mean(x) assumes x and completion_mask have the same
sequence length, but they can differ by ~22 tokens causing:
  RuntimeError: The size of tensor a (534) must match the size of tensor b (512)

Fix: truncate both tensors to the shorter length before multiplication.

The function is NESTED inside compute_loss in the compiled cache file, so
standard monkey-patching via sys.modules/globals doesn't work. We must
patch the source file directly.

Usage (run ONCE after installing Unsloth, or after any Unsloth upgrade):
    python training/patch_unsloth_cache.py

    # Or with explicit cache generation:
    python training/patch_unsloth_cache.py --generate

This modifies: unsloth_compiled_cache/UnslothGRPOTrainer.py
"""

import argparse
import glob
import os
import sys


def find_cache_files():
    """Find all Unsloth compiled cache files that may contain the bug."""
    patterns = [
        "unsloth_compiled_cache/UnslothGRPOTrainer.py",
        "unsloth_compiled_cache/*GRPOTrainer*.py",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return list(set(files))


def generate_cache():
    """Import unsloth + trl to generate the compiled cache if needed."""
    print("Generating Unsloth compiled cache (importing unsloth + trl)...")
    os.environ.setdefault("WANDB_DISABLED", "true")
    import unsloth  # noqa: F401 — generates the cache
    from trl import GRPOConfig, GRPOTrainer  # noqa: F401 — triggers patching
    print("Cache generated.")


def patch_file(filepath):
    """Patch a single compiled cache file."""
    print(f"\nPatching: {filepath}")

    with open(filepath, "r") as f:
        content = f.read()

    # The buggy pattern (inside nested def masked_batch_mean(x):)
    old = "return (x * completion_mask).sum() / completion_token_count"

    count = content.count(old)
    if count == 0:
        print("  Pattern not found (already patched or different Unsloth version)")
        return False

    # Fixed: truncate x and completion_mask to the shorter sequence length
    new = (
        "_n = min(x.shape[-1], completion_mask.shape[-1]); "
        "return (x[..., :_n] * completion_mask[..., :_n]).sum() / "
        "completion_mask[..., :_n].sum().clamp(min=1)"
    )

    patched = content.replace(old, new)

    with open(filepath, "w") as f:
        f.write(patched)

    # Remove .pyc bytecode cache so Python re-compiles from patched source
    pycache_dir = os.path.join(os.path.dirname(filepath), "__pycache__")
    if os.path.exists(pycache_dir):
        removed = 0
        for pyc in glob.glob(os.path.join(pycache_dir, "*.pyc")):
            os.remove(pyc)
            removed += 1
        if removed:
            print(f"  Removed {removed} bytecode cache file(s)")

    print(f"  Patched {count} occurrence(s)")
    return True


def verify_patch(filepath):
    """Verify the patch was applied correctly."""
    with open(filepath, "r") as f:
        content = f.read()

    old = "return (x * completion_mask).sum() / completion_token_count"
    if old in content:
        print(f"  VERIFICATION FAILED: buggy pattern still present in {filepath}")
        return False

    if "min(x.shape[-1], completion_mask.shape[-1])" in content:
        print(f"  VERIFIED: patch is active in {filepath}")
        return True

    print(f"  WARNING: could not verify patch in {filepath}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Patch Unsloth compiled cache to fix masked_batch_mean tensor mismatch"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate cache by importing unsloth (requires GPU)",
    )
    args = parser.parse_args()

    files = find_cache_files()

    if not files:
        if args.generate:
            generate_cache()
            files = find_cache_files()
        else:
            print("No compiled cache found.")
            print("Run with --generate on a GPU node, or run 'python -c \"import unsloth\"' first.")
            sys.exit(1)

    if not files:
        print("ERROR: Could not find Unsloth compiled cache after generation.")
        sys.exit(1)

    print(f"Found {len(files)} cache file(s)")

    patched_any = False
    for f in files:
        if patch_file(f):
            patched_any = True
        verify_patch(f)

    if patched_any:
        print("\nPatch applied successfully!")
    else:
        print("\nNo patches needed.")


if __name__ == "__main__":
    main()

#!/bin/bash
#SBATCH --job-name=finsent-gemma4-e4b
#SBATCH --account=r01510
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=14:00:00
#SBATCH --output=logs/gemma4_e4b_%j.out
#SBATCH --error=logs/gemma4_e4b_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# FinSent - Gemma4-E4B full pipeline (SFT -> GRPO -> Export)

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== Job $SLURM_JOB_ID - Gemma4-E4B ==="
echo "Node: $SLURM_NODELIST | GPUs: $SLURM_GPUS_ON_NODE | Start: $(date)"

# Stagger start to avoid HF Hub rate limiting when multiple jobs launch together
sleep $(( SLURM_JOB_ID % 120 ))

# Cache redirect
export HF_HOME=/N/scratch/ayshaikh/.cache/huggingface
export HF_HUB_CACHE=/N/scratch/ayshaikh/.cache/huggingface/hub
export XDG_CACHE_HOME=/N/scratch/ayshaikh/.cache
export TORCH_HOME=/N/scratch/ayshaikh/.cache/torch
export TMPDIR=/N/scratch/ayshaikh/tmp/$SLURM_JOB_ID
export CUDA_CACHE_PATH=/N/scratch/ayshaikh/.cache/nv
export TRITON_CACHE_DIR=/N/scratch/ayshaikh/.cache/triton
export NUMBA_CACHE_DIR=/N/scratch/ayshaikh/.cache/numba
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" \
         "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR"

module load python/gpu/3.12.5
module load cudatoolkit/12.6

cd /N/scratch/ayshaikh/FinSent-CoT
source venv/bin/activate
mkdir -p logs

# Auth tokens
if [ -f /N/scratch/ayshaikh/.tokens ]; then
    source /N/scratch/ayshaikh/.tokens
fi
export WANDB_PROJECT=FinSent
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p "$WANDB_DIR"

# Clear stale Unsloth compiled cache (may be from different TRL version)
rm -rf "$TMPDIR/unsloth_compiled_cache" "./unsloth_compiled_cache"

# A100 compatibility patches - must run BEFORE patch_unsloth_cache (which
# imports unsloth). If fast_lora.py is corrupted, this repairs it first.
python training/patch_a100.py

# Patch Unsloth compiled cache (fix masked_batch_mean tensor mismatch)
python training/patch_unsloth_cache.py --generate
ln -sfn "$TMPDIR/unsloth_compiled_cache" ./unsloth_compiled_cache 2>/dev/null || true

# Build llama.cpp if not present (needed for GGUF export)
# NOTE: runs in subshell (...) so cd cannot leak into the parent script
if [ ! -f llama.cpp/build/bin/llama-quantize ]; then
    if [ ! -d llama.cpp ]; then
        echo "Cloning llama.cpp..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp
    fi
    echo "Building llama.cpp (cmake)..."
    (
        cd llama.cpp
        cmake -B build 2>&1 | tail -5
        cmake --build build --target llama-quantize -j16 2>&1 | tail -5
    )
fi

# Install gguf module (needed by convert_hf_to_gguf.py)
python -m pip install gguf -q 2>/dev/null || true

# Ensure llama.cpp tools are in PATH for GGUF export
export PATH="$PWD/llama.cpp/build/bin:$PATH"

# Run full pipeline
python training/gemma4_e4b.py --phase all --upload

echo "End: $(date)"

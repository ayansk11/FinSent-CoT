#!/bin/bash
#SBATCH --job-name=finsent-qwen3.5-0.8b
#SBATCH --account=r01510
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3_5_0_8b_%j.out
#SBATCH --error=logs/qwen3_5_0_8b_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# FinSent-CoT — Qwen3.5-0.8B full pipeline (SFT -> GRPO -> Export)

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== Job $SLURM_JOB_ID — Qwen3.5-0.8B ==="
echo "Node: $SLURM_NODELIST | GPUs: $SLURM_GPUS_ON_NODE | Start: $(date)"

# Stagger start to avoid HF Hub rate limiting when multiple jobs launch together
sleep $(( SLURM_JOB_ID % 120 ))

export HF_HOME=/N/scratch/ayshaikh/.cache/huggingface
export HF_HUB_CACHE=/N/scratch/ayshaikh/.cache/huggingface/hub
export XDG_CACHE_HOME=/N/scratch/ayshaikh/.cache
export TORCH_HOME=/N/scratch/ayshaikh/.cache/torch
export TMPDIR=/N/scratch/ayshaikh/tmp
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

if [ -f /N/scratch/ayshaikh/.tokens ]; then
    source /N/scratch/ayshaikh/.tokens
fi
export WANDB_PROJECT=FinSent-CoT
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p "$WANDB_DIR"

# Clear stale Unsloth compiled cache (may be from different TRL version)
rm -rf "$TMPDIR/unsloth_compiled_cache"

# Patch Unsloth compiled cache (fix masked_batch_mean tensor mismatch)
python training/patch_unsloth_cache.py --generate

# A100 compatibility patches (matmul_lora dtype fix + Qwen3.5 position_ids fix)
python training/patch_a100.py

# A100 fix: patch Unsloth matmul_lora dtype mismatch (fp16 vs bf16 in autocast)
_UNSLOTH_UTILS="venv/lib/python3.12/site-packages/unsloth/kernels/utils.py"
if [ -f "$_UNSLOTH_UTILS" ] && grep -q 'out\.addmm_(XA, B\.to(dtype), alpha = s)' "$_UNSLOTH_UTILS"; then
    sed -i 's/out\.addmm_(XA, B\.to(dtype), alpha = s)/_dt = out.dtype; out.addmm_(XA.to(_dt), B.to(_dt), alpha = s)/g' "$_UNSLOTH_UTILS"
    echo "Patched Unsloth matmul_lora for A100 dtype compat"
fi

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
        cmake -B build -DGGML_CUDA=ON 2>&1 | tail -5
        cmake --build build --target llama-quantize -j16 2>&1 | tail -5
    )
fi

# Ensure Unsloth can find pre-built llama.cpp for GGUF export
export PATH="$PWD/llama.cpp/build/bin:$PATH"

# Run full pipeline
python training/qwen3_5_0_8b.py --phase all

echo "End: $(date)"

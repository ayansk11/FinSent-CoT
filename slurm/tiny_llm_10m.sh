#!/bin/bash
#SBATCH --job-name=finsenti-tiny-llm
#SBATCH --account=r01510
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --output=logs/tiny_llm_10m_%j.out
#SBATCH --error=logs/tiny_llm_10m_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# FinSenti - Tiny-LLM-10M full pipeline (SFT -> GRPO -> Export)
# Uses PEFT + bitsandbytes (no Unsloth)

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== Job $SLURM_JOB_ID - Tiny-LLM-10M ==="
echo "Node: $SLURM_NODELIST | GPUs: $SLURM_GPUS_ON_NODE | Start: $(date)"

# Stagger start to avoid HF Hub rate limiting when multiple jobs launch together
sleep $(( SLURM_JOB_ID % 120 ))

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

if [ -f /N/scratch/ayshaikh/.tokens ]; then
    source /N/scratch/ayshaikh/.tokens
fi
export WANDB_PROJECT=FinSenti
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p "$WANDB_DIR"

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

# Install/repair Tiny-LLM deps. Pin trl<0.25 to stay compatible with the
# shared unsloth 2026.4.6 install other Qwen jobs depend on.
python -m pip install wandb datasets peft bitsandbytes 'trl<0.25' accelerate gguf transformers==5.2.0 -q 2>&1 | tail -3 || true

# Fail fast if the environment is still broken (same check as mobilellm).
python -c "
import torch, transformers, trl
print(f'env: torch={torch.__version__} transformers={transformers.__version__} trl={trl.__version__}')
from trl import GRPOConfig, GRPOTrainer
" || { echo 'Environment sanity check failed'; exit 1; }

# Ensure llama.cpp tools are in PATH for GGUF export
export PATH="$PWD/llama.cpp/build/bin:$PATH"

# A100 compatibility patches
python training/patch_a100.py

# Resume from saved SFT checkpoint if present. SFT for Tiny-LLM is fast
# (~6 min) so we just rerun it if the checkpoint is missing; the 8-hour
# TIMEOUT in job 6896744 was in GRPO, not SFT.
SFT_CKPT=./checkpoints/sft/tiny-llm-10m
if [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
    echo "SFT checkpoint not found at $SFT_CKPT - running SFT phase first"
    python training/tiny_llm_10m.py --phase sft
fi
python training/tiny_llm_10m.py --phase grpo
python training/tiny_llm_10m.py --phase export --upload

echo "End: $(date)"

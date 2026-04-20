#!/bin/bash
#SBATCH --job-name=finsenti-mobilellm
#SBATCH --account=r01510
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=36:00:00
#SBATCH --output=logs/mobilellm_r1_950m_%j.out
#SBATCH --error=logs/mobilellm_r1_950m_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# FinSenti - MobileLLM-R1-950M full pipeline (SFT -> GRPO -> Export)
# Uses PEFT + bitsandbytes (no Unsloth)

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== Job $SLURM_JOB_ID - MobileLLM-R1-950M ==="
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

# Install/repair MobileLLM deps. Use trl<0.25 to stay compatible with unsloth 2026.4.6
# (other Qwen jobs share the same ~/.local install).
python -m pip install wandb datasets peft bitsandbytes 'trl<0.25' accelerate gguf transformers==5.2.0 -q 2>&1 | tail -3 || true

# Fail fast if the environment is still in the broken state that caused job 6893085
python -c "
import torch, triton, transformers, trl
print(f'env: torch={torch.__version__} triton={triton.__version__} transformers={transformers.__version__} trl={trl.__version__}')
from trl import GRPOConfig, GRPOTrainer
" || { echo 'Environment sanity check failed. Run the pip install in README before resubmitting.'; exit 1; }

# Ensure llama.cpp tools are in PATH for GGUF export
export PATH="$PWD/llama.cpp/build/bin:$PATH"

# A100 compatibility patches
python training/patch_a100.py

# Resume from saved SFT checkpoint. SFT already completed in job 6893085
# (1.64h, train_loss 1.33, mean_token_accuracy 0.757) and the adapter is at
# ./checkpoints/sft/mobilellm-r1-950m/. If that directory is missing, add a
# `python training/mobilellm_r1_950m.py --phase sft` call back before the
# grpo one below.
SFT_CKPT=./checkpoints/sft/mobilellm-r1-950m
if [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
    echo "SFT checkpoint not found at $SFT_CKPT - running SFT phase first"
    python training/mobilellm_r1_950m.py --phase sft
fi
python training/mobilellm_r1_950m.py --phase grpo
python training/mobilellm_r1_950m.py --phase export --upload

echo "End: $(date)"

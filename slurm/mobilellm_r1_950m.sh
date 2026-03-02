#!/bin/bash
#SBATCH --job-name=finsent-mobilellm
#SBATCH --account=r01510
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=logs/mobilellm_r1_950m_%j.out
#SBATCH --error=logs/mobilellm_r1_950m_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# FinSent-CoT — MobileLLM-R1-950M full pipeline (SFT -> GRPO -> Export)
# Uses PEFT + bitsandbytes (no Unsloth)

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== Job $SLURM_JOB_ID — MobileLLM-R1-950M ==="
echo "Node: $SLURM_NODELIST | GPUs: $SLURM_GPUS_ON_NODE | Start: $(date)"

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

module load python/gpu/3.11.5
module load cudatoolkit/12.1

cd /N/scratch/ayshaikh/FinSent-CoT
source venv/bin/activate
mkdir -p logs

if [ -f /N/scratch/ayshaikh/.tokens ]; then
    source /N/scratch/ayshaikh/.tokens
fi
export WANDB_PROJECT=FinSent-CoT
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p "$WANDB_DIR"

# Build llama.cpp if not present (needed for GGUF export)
if [ ! -f llama.cpp/llama-quantize ] && [ ! -f llama.cpp/quantize ]; then
    if [ ! -d llama.cpp ]; then
        echo "Cloning llama.cpp..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp
    fi
    echo "Building llama.cpp..."
    cd llama.cpp && make -j16 llama-quantize 2>&1 | tail -3 && cd ..
fi

# Run full pipeline
python training/mobilellm_r1_950m.py --phase all

echo "End: $(date)"

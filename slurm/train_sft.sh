#!/bin/bash
#SBATCH --job-name=finsent-sft
#SBATCH --account=r01510
#SBATCH --partition=hopper
#SBATCH --qos=hopper
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ─── Cache redirect ─────────────────────────────────────────────────────────
export HF_HOME=/N/scratch/ayshaikh/.cache/huggingface
export HF_HUB_CACHE=/N/scratch/ayshaikh/.cache/huggingface/hub
export XDG_CACHE_HOME=/N/scratch/ayshaikh/.cache
export TORCH_HOME=/N/scratch/ayshaikh/.cache/torch
export TMPDIR=/N/scratch/ayshaikh/tmp
mkdir -p $TMPDIR

# ─── Load modules ───────────────────────────────────────────────────────────
module load python/gpu

# ─── Activate venv & cd ─────────────────────────────────────────────────────
source /N/scratch/ayshaikh/FinSent-CoT/venv/bin/activate
cd /N/scratch/ayshaikh/FinSent-CoT
mkdir -p logs

# ─── Ensure training deps are installed (needs GPU node) ──────────────────
pip install unsloth trl --quiet 2>/dev/null

# ─── Load auth tokens (HF_TOKEN + WANDB_API_KEY) ─────────────────────────
source /N/scratch/ayshaikh/.tokens
export WANDB_PROJECT=FinSent-CoT
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p $WANDB_DIR

# ─── Run SFT training ──────────────────────────────────────────────────────
echo "[$(date)] Starting SFT warm-up..."
python training/train_sft.py \
    --base-model "unsloth/Qwen3-4B" \
    --dataset-dir ./validated \
    --output-dir ./checkpoints/sft \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4

echo "[$(date)] SFT training complete! Exit code: $?"

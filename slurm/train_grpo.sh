#!/bin/bash
#SBATCH --job-name=finsent-grpo
#SBATCH --account=r01510
#SBATCH --partition=hopper
#SBATCH --qos=hopper
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
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
module load python
module load cuda

# ─── Activate venv & cd ─────────────────────────────────────────────────────
source /N/scratch/ayshaikh/FinSent-CoT/venv/bin/activate
cd /N/scratch/ayshaikh/FinSent-CoT
mkdir -p logs

# ─── Load auth tokens (HF_TOKEN + WANDB_API_KEY) ─────────────────────────
source /N/scratch/ayshaikh/.tokens
export WANDB_PROJECT=FinSent-CoT
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p $WANDB_DIR

# ─── Run GRPO training ─────────────────────────────────────────────────────
echo "[$(date)] Starting GRPO training..."
python training/train_grpo.py \
    --sft-checkpoint ./checkpoints/sft \
    --dataset-dir ./validated \
    --output-dir ./checkpoints/grpo \
    --max-steps 3000 \
    --batch-size 4 \
    --lr 5e-5 \
    --num-generations 6 \
    --max-completion-length 512 \
    --eval-steps 50 \
    --early-stop-patience 10 \
    --early-stop-min-delta 0.01 \
    --early-stop-warmup 200

echo "[$(date)] GRPO training complete! Exit code: $?"

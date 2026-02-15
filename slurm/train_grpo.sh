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
#SBATCH --exclusive

# ============================================================
# GRPO Training: up to 3000 steps with early stopping
# 4 equal-weight reward functions, patience-based convergence
# ============================================================

set -euo pipefail

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"
echo "================"

# ─── Cache redirect (CRITICAL — home dir has 5GB quota) ─────────────────────
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

# ─── Load modules ───────────────────────────────────────────────────────────
module load python/gpu/3.11.5
module load cudatoolkit/12.1

# ─── Activate venv & cd ─────────────────────────────────────────────────────
cd /N/scratch/ayshaikh/FinSent-CoT
source venv/bin/activate
mkdir -p logs

# ─── Ensure training deps are installed (needs GPU node) ──────────────────
pip install unsloth trl --quiet 2>/dev/null || true

# ─── Load auth tokens (HF_TOKEN + WANDB_API_KEY) ─────────────────────────
if [ -f /N/scratch/ayshaikh/.tokens ]; then
    source /N/scratch/ayshaikh/.tokens
    echo "HF Token: $([ -n "${HF_TOKEN:-}" ] && echo 'SET' || echo 'NOT SET')"
    echo "WANDB Key: $([ -n "${WANDB_API_KEY:-}" ] && echo 'SET' || echo 'NOT SET')"
fi
export WANDB_PROJECT=FinSent-CoT
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p "$WANDB_DIR"

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

GRPO_EXIT=$?
echo ""
echo "End: $(date)"
echo "GRPO training exit code: $GRPO_EXIT"
exit $GRPO_EXIT

#!/bin/bash
#SBATCH --job-name=finsent-export
#SBATCH --account=r01510
#SBATCH --partition=hopper
#SBATCH --qos=hopper
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/export_%j.out
#SBATCH --error=logs/export_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# ============================================================
# Export: 3 GGUF quantizations (Q4_K_M, Q5_K_M, Q8_0)
# Each quant goes to its own HuggingFace repo
#
# Usage:
#   sbatch slurm/export_gguf.sh qwen3-4b
#   sbatch slurm/export_gguf.sh mobilellm-r1-950m
#
# Available model keys:
#   qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-8b,
#   deepseek-r1-1.5b, mobilellm-r1-950m
# ============================================================

set -euo pipefail

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# ─── Get model key from argument ────────────────────────────────────────────
MODEL_KEY="${1:-qwen3-4b}"
echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Model: $MODEL_KEY"
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

# ─── Ensure export deps are installed (needs GPU node) ────────────────────
pip install unsloth --quiet 2>/dev/null || true

# ─── Load auth tokens (HF_TOKEN + WANDB_API_KEY) ─────────────────────────
if [ -f /N/scratch/ayshaikh/.tokens ]; then
    source /N/scratch/ayshaikh/.tokens
    echo "HF Token: $([ -n "${HF_TOKEN:-}" ] && echo 'SET' || echo 'NOT SET')"
    echo "WANDB Key: $([ -n "${WANDB_API_KEY:-}" ] && echo 'SET' || echo 'NOT SET')"
fi
export WANDB_PROJECT=FinSent-CoT
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p "$WANDB_DIR"

# ─── Export to GGUF (Q4_K_M + Q5_K_M + Q8_0) + HF upload ──────────────
echo "[$(date)] Starting GGUF export for $MODEL_KEY..."
python training/export_gguf.py \
    --model-key "$MODEL_KEY" \
    --grpo-checkpoint "./checkpoints/grpo/$MODEL_KEY" \
    --output-dir "./export/$MODEL_KEY" \
    --upload

EXPORT_EXIT=$?
echo ""
echo "End: $(date)"
echo "Model: $MODEL_KEY"
echo "Export exit code: $EXPORT_EXIT"
exit $EXPORT_EXIT

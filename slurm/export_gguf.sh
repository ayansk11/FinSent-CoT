#!/bin/bash
#SBATCH --job-name=finsent-export
#SBATCH --account=r01510
#SBATCH --partition=hopper
#SBATCH --qos=hopper
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/export_%j.out
#SBATCH --error=logs/export_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# ============================================================
# Export: Dual GGUF (Q5_K_M + Q4_K_M) + HuggingFace upload
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

# ─── Export to GGUF (both Q5_K_M and Q4_K_M) ─────────────────────────────
echo "[$(date)] Starting dual GGUF export..."
python training/export_gguf.py \
    --grpo-checkpoint ./checkpoints/grpo \
    --output-dir ./export \
    --model-name FinSent-CoT-Qwen3-4B

EXPORT_EXIT=$?
echo "[$(date)] Export complete! Exit code: $EXPORT_EXIT"

# ─── Upload to HuggingFace ─────────────────────────────────────────────────
if [ $EXPORT_EXIT -eq 0 ]; then
    echo "[$(date)] Uploading to HuggingFace..."
    python -c "
from huggingface_hub import HfApi
api = HfApi()

# Upload both GGUF quantizations
for quant in ['Q5_K_M', 'Q4_K_M']:
    fname = f'FinSent-CoT-Qwen3-4B.{quant}.gguf'
    api.upload_file(
        path_or_fileobj=f'./export/{fname}',
        path_in_repo=fname,
        repo_id='Ayansk11/FinSent-CoT-Qwen3-4B',
        repo_type='model',
    )
    print(f'  Uploaded {fname}')

# Upload Modelfile (points to Q5_K_M by default)
api.upload_file(
    path_or_fileobj='./export/Modelfile',
    path_in_repo='Modelfile',
    repo_id='Ayansk11/FinSent-CoT-Qwen3-4B',
    repo_type='model',
)

# Upload merged HF weights
api.upload_folder(
    folder_path='./export/merged_hf',
    repo_id='Ayansk11/FinSent-CoT-Qwen3-4B',
    repo_type='model',
)

print('HuggingFace upload complete!')
"
    echo "[$(date)] Upload complete!"
fi

echo ""
echo "End: $(date)"
echo "Exit code: $EXPORT_EXIT"
exit $EXPORT_EXIT

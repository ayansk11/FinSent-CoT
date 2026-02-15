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

# ─── Export to GGUF (both Q5_K_M and Q4_K_M) ─────────────────────────────
echo "[$(date)] Starting dual GGUF export..."
python training/export_gguf.py \
    --grpo-checkpoint ./checkpoints/grpo \
    --output-dir ./export \
    --model-name FinSent-CoT-Qwen3-4B

echo "[$(date)] Export complete! Exit code: $?"

# ─── Upload to HuggingFace ─────────────────────────────────────────────────
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

echo "[$(date)] All done!"

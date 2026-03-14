#!/bin/bash
# ============================================================
# One-time environment setup for FinSent-CoT training.
#
# Run this ONCE on a GPU node BEFORE submitting training jobs.
# It creates/updates the venv with all required dependencies.
#
# Target: Big Red 200 — A100 GPUs (gpu partition)
#
# Usage (interactive GPU session):
#   srun --account=r01510 --partition=gpu \
#        --gpus-per-node=1 --mem=32G --time=01:00:00 --pty bash
#   bash slurm/setup_env.sh
#
# Usage (SLURM batch — if interactive isn't available):
#   sbatch slurm/setup_env.sh
# ============================================================
#SBATCH --job-name=finsent-setup
#SBATCH --account=r01510
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=== FinSent-CoT Environment Setup ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# ─── Cache redirect ──────────────────────────────────────────────────────
export HF_HOME=/N/scratch/ayshaikh/.cache/huggingface
export HF_HUB_CACHE=/N/scratch/ayshaikh/.cache/huggingface/hub
export XDG_CACHE_HOME=/N/scratch/ayshaikh/.cache
export TORCH_HOME=/N/scratch/ayshaikh/.cache/torch
export TMPDIR=/N/scratch/ayshaikh/tmp
export PIP_CACHE_DIR=/N/scratch/ayshaikh/.cache/pip
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" \
         "$TMPDIR" "$PIP_CACHE_DIR"

# ─── Load modules (must match training scripts: python/gpu/3.12.5 + cudatoolkit/12.6)
module load python/gpu/3.12.5
module load cudatoolkit/12.6

cd /N/scratch/ayshaikh/FinSent-CoT
mkdir -p logs

# ─── Create or activate venv ─────────────────────────────────────────────
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
echo "Venv: $(which python)"
echo ""

# ─── Upgrade pip first ───────────────────────────────────────────────────
pip install --upgrade pip setuptools wheel

# ─── Install PyTorch (CUDA 12.6 compatible — cu124 wheels work with 12.6)
echo ""
echo "=== Installing PyTorch ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: CUDA not available! Training will fail.')
    exit(1)
"

# ─── Install training dependencies ───────────────────────────────────────
echo ""
echo "=== Installing training dependencies ==="
pip install \
    transformers \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    trl \
    wandb \
    huggingface_hub \
    sentencepiece \
    protobuf

# ─── Install Unsloth (AFTER torch + deps are stable) ────────────────────
echo ""
echo "=== Installing Unsloth ==="
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"

# ─── Verify ALL dependencies ─────────────────────────────────────────────
echo ""
echo "=== Verifying installations ==="
python << 'PYEOF'
packages = {
    'torch': 'import torch',
    'transformers': 'import transformers',
    'datasets': 'import datasets',
    'accelerate': 'import accelerate',
    'peft': 'import peft',
    'bitsandbytes': 'import bitsandbytes',
    'trl': 'from trl import SFTConfig',
    'wandb': 'import wandb',
    'huggingface_hub': 'import huggingface_hub',
    'unsloth': 'from unsloth import FastLanguageModel',
}

all_ok = True
for name, stmt in packages.items():
    try:
        exec(stmt)
        print(f'  OK: {name}')
    except Exception as e:
        print(f'  FAIL: {name}: {e}')
        all_ok = False

if all_ok:
    print()
    print('All dependencies installed successfully!')
else:
    print()
    print('WARNING: Some dependencies failed. Fix before submitting jobs.')
    exit(1)
PYEOF

# ─── Download training data from HuggingFace if missing ──────────────────
echo ""
echo "=== Checking training data ==="
if [ ! -f validated/sft_train.jsonl ] || [ ! -f validated/grpo_train.jsonl ]; then
    echo "  Downloading dataset from Ayansk11/FinSent-CoT-Dataset..."
    python data_generation/download_dataset.py
else
    for f in validated/sft_train.jsonl validated/grpo_train.jsonl; do
        lines=$(wc -l < "$f")
        echo "  OK: $f ($lines lines)"
    done
fi

# ─── Verify training scripts exist ────────────────────────────────────
echo ""
echo "=== Checking training scripts ==="
for f in training/qwen3_0_6b.py training/qwen3_4b.py training/qwen3_8b.py \
         training/deepseek_r1_1_5b.py training/mobilellm_r1_950m.py \
         training/rewards.py training/callbacks.py; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo "Date: $(date)"
echo ""
echo "You can now submit training jobs:"
echo "  bash slurm/run_all.sh --all"

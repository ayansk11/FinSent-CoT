#!/bin/bash
#SBATCH --job-name=finsent-datagen
#SBATCH --account=r01510
#SBATCH --partition=hopper
#SBATCH --qos=hopper
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/datagen_%j.out
#SBATCH --error=logs/datagen_%j.err
#SBATCH --mail-user=ayshaikh@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# ============================================================
# Data Generation: Qwen3-235B-A22B-FP8 via vLLM on 4x H100
# Generates ~50K CoT financial sentiment samples
#
# NOTE: Uses a SEPARATE venv for vLLM server (0.8.5 + torch 2.6.0)
# because vLLM >=0.9 has broken MoE kernels (topk_softmax bug).
# The generation Python scripts run in the main project venv.
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

# ─── Project dir ──────────────────────────────────────────────────────────
cd /N/scratch/ayshaikh/FinSent-CoT
mkdir -p logs

# ─── Load auth tokens (HF_TOKEN + WANDB_API_KEY) ─────────────────────────
if [ -f /N/scratch/ayshaikh/.tokens ]; then
    source /N/scratch/ayshaikh/.tokens
    echo "HF Token: $([ -n "${HF_TOKEN:-}" ] && echo 'SET' || echo 'NOT SET')"
    echo "WANDB Key: $([ -n "${WANDB_API_KEY:-}" ] && echo 'SET' || echo 'NOT SET')"
fi
export WANDB_PROJECT=FinSent-CoT
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p "$WANDB_DIR"

# ─── Setup vLLM venv (separate from main venv, needs torch 2.6.0) ────────
# vLLM >= 0.9 has broken MoE kernels (_moe_C topk_softmax bug).
# Qwen team recommends vLLM >= 0.8.5 for Qwen3-235B-A22B.
# vLLM 0.8.5 requires torch 2.6.0 — incompatible with our main venv (torch 2.9.0).
VLLM_VENV=/N/scratch/ayshaikh/vllm_venv
if [ ! -f "$VLLM_VENV/bin/activate" ]; then
    echo "[$(date)] Creating vLLM venv (one-time setup)..."
    python -m venv "$VLLM_VENV"
    source "$VLLM_VENV/bin/activate"
    pip install --upgrade pip
    echo "[$(date)] Installing vLLM 0.8.5 (this may take a few minutes)..."
    pip install vllm==0.8.5
    echo "[$(date)] vLLM venv ready!"
else
    echo "[$(date)] Using existing vLLM venv..."
    source "$VLLM_VENV/bin/activate"
    # Verify vLLM is installed
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
fi

# ─── Start vLLM server with Qwen3-235B-A22B-FP8 (4x H100 tensor parallel) ─
echo "[$(date)] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-235B-A22B-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-log-requests \
    &

VLLM_PID=$!
echo "[$(date)] vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready (large model, may take 30-60 min to load)
VLLM_TIMEOUT=3600
echo "[$(date)] Waiting for vLLM to be ready (timeout=${VLLM_TIMEOUT}s)..."
VLLM_READY=false
for i in $(seq 1 $VLLM_TIMEOUT); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "[$(date)] vLLM server ready after ${i}s"
        VLLM_READY=true
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM server process died"
        exit 1
    fi
    sleep 1
done

if [ "$VLLM_READY" = false ]; then
    echo "[$(date)] ERROR: vLLM failed to start within ${VLLM_TIMEOUT}s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Verify model is loaded
curl -s http://localhost:8000/v1/models | python -m json.tool || echo "WARNING: Could not parse /v1/models response"
echo ""

# ─── Switch to main project venv for generation scripts ──────────────────
# (vLLM server continues running in background from the vllm_venv)
deactivate
source /N/scratch/ayshaikh/FinSent-CoT/venv/bin/activate
echo "[$(date)] Switched to project venv for generation scripts"

# ─── Run data generation ───────────────────────────────────────────────────
echo "[$(date)] Starting CoT data generation..."
python data_generation/generate_cot_v2.py \
    --model "Qwen/Qwen3-235B-A22B-FP8" \
    --api-base "http://localhost:8000/v1" \
    --target-samples 50000 \
    --batch-size 32 \
    --checkpoint-dir ./checkpoints/datagen \
    --temperature 0.4

DATAGEN_EXIT=$?
echo "[$(date)] Data generation finished with exit code: $DATAGEN_EXIT"

# ─── Validate dataset ──────────────────────────────────────────────────────
if [ $DATAGEN_EXIT -eq 0 ]; then
    echo "[$(date)] Running validation..."
    python data_generation/validate_dataset.py \
        --input ./checkpoints/datagen/generated_cot.jsonl \
        --output-dir ./validated \
        --strict
    echo "[$(date)] Validation complete!"
fi

# ─── Cleanup ────────────────────────────────────────────────────────────────
echo "[$(date)] Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "End: $(date)"
echo "Exit code: $DATAGEN_EXIT"
exit $DATAGEN_EXIT

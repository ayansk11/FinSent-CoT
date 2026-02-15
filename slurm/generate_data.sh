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

# ─── Activate venv ──────────────────────────────────────────────────────────
cd /N/scratch/ayshaikh/FinSent-CoT
source venv/bin/activate
mkdir -p logs

# ─── Ensure vLLM is installed (needs GPU node) ─────────────────────────────
pip install vllm openai --quiet 2>/dev/null || true

# ─── Load auth tokens (HF_TOKEN + WANDB_API_KEY) ─────────────────────────
if [ -f /N/scratch/ayshaikh/.tokens ]; then
    source /N/scratch/ayshaikh/.tokens
    echo "HF Token: $([ -n "${HF_TOKEN:-}" ] && echo 'SET' || echo 'NOT SET')"
    echo "WANDB Key: $([ -n "${WANDB_API_KEY:-}" ] && echo 'SET' || echo 'NOT SET')"
fi
export WANDB_PROJECT=FinSent-CoT
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p "$WANDB_DIR"

# ─── Start vLLM server with Qwen3-235B-A22B-FP8 (4x H100 tensor parallel) ─
echo "[$(date)] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-235B-A22B-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --dtype float16 \
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

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

# ─── Cache redirect (CRITICAL — home dir has 5GB quota) ─────────────────────
export HF_HOME=/N/scratch/ayshaikh/.cache/huggingface
export HF_HUB_CACHE=/N/scratch/ayshaikh/.cache/huggingface/hub
export XDG_CACHE_HOME=/N/scratch/ayshaikh/.cache
export TORCH_HOME=/N/scratch/ayshaikh/.cache/torch
export TMPDIR=/N/scratch/ayshaikh/tmp
mkdir -p $TMPDIR $HF_HOME $TORCH_HOME

# ─── Load modules ───────────────────────────────────────────────────────────
module load python
module load cuda

# ─── Activate venv ──────────────────────────────────────────────────────────
source /N/scratch/ayshaikh/FinSent-CoT/venv/bin/activate

# ─── Load auth tokens (HF_TOKEN + WANDB_API_KEY) ─────────────────────────
source /N/scratch/ayshaikh/.tokens
export WANDB_PROJECT=FinSent-CoT
export WANDB_DIR=/N/scratch/ayshaikh/FinSent-CoT/wandb
mkdir -p $WANDB_DIR

# ─── Project dir ────────────────────────────────────────────────────────────
cd /N/scratch/ayshaikh/FinSent-CoT
mkdir -p logs

# ─── Start vLLM server with Qwen3-235B-A22B-FP8 (4x H100 tensor parallel) ─
echo "[$(date)] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-235B-A22B-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --dtype float16 \
    --port 8000 \
    --trust-remote-code \
    &

VLLM_PID=$!
echo "[$(date)] vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready (large model, may take 10-15 min to load)
echo "[$(date)] Waiting for vLLM to be ready..."
MAX_WAIT=1200  # 20 minutes
WAITED=0
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[$(date)] ERROR: vLLM failed to start within ${MAX_WAIT}s"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo "[$(date)] Waiting... (${WAITED}s / ${MAX_WAIT}s)"
done
echo "[$(date)] vLLM is ready!"

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
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "[$(date)] Job complete!"

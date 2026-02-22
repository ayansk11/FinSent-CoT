#!/bin/bash
# ============================================================
# Master orchestration: Train all 6 models end-to-end
#
# Submits SFT -> GRPO -> Export for each model.
# GRPO jobs depend on SFT completion (--dependency=afterok).
# Export jobs depend on GRPO completion.
#
# Usage:
#   bash slurm/train_all.sh           # Submit all 6 models
#   bash slurm/train_all.sh sft       # SFT only for all 6
#   bash slurm/train_all.sh grpo      # GRPO only (assumes SFT done)
#   bash slurm/train_all.sh export    # Export only (assumes GRPO done)
# ============================================================

set -euo pipefail

cd /N/scratch/ayshaikh/FinSent-CoT
mkdir -p logs

PHASE="${1:-all}"

MODELS=(
    "qwen3-0.6b"
    "mobilellm-r1-950m"
    "deepseek-r1-1.5b"
    "qwen3-1.7b"
    "qwen3-4b"
    "qwen3-8b"
)

echo "============================================"
echo "FinSent-CoT Multi-Model Training Pipeline"
echo "============================================"
echo "Phase: $PHASE"
echo "Models: ${MODELS[*]}"
echo "Date: $(date)"
echo ""

if [ "$PHASE" = "sft" ] || [ "$PHASE" = "all" ]; then
    echo "--- Submitting SFT jobs ---"
    declare -A SFT_JOBS
    for model in "${MODELS[@]}"; do
        JOB_ID=$(sbatch --parsable slurm/train_sft.sh "$model")
        SFT_JOBS[$model]=$JOB_ID
        echo "  $model -> SFT Job $JOB_ID"
    done
    echo ""
fi

if [ "$PHASE" = "grpo" ] || [ "$PHASE" = "all" ]; then
    echo "--- Submitting GRPO jobs ---"
    declare -A GRPO_JOBS
    for model in "${MODELS[@]}"; do
        if [ "$PHASE" = "all" ]; then
            # Chain after SFT completion
            SFT_JOB=${SFT_JOBS[$model]}
            JOB_ID=$(sbatch --parsable --dependency=afterok:$SFT_JOB slurm/train_grpo.sh "$model")
            echo "  $model -> GRPO Job $JOB_ID (after SFT $SFT_JOB)"
        else
            # Submit immediately (SFT assumed done)
            JOB_ID=$(sbatch --parsable slurm/train_grpo.sh "$model")
            echo "  $model -> GRPO Job $JOB_ID"
        fi
        GRPO_JOBS[$model]=$JOB_ID
    done
    echo ""
fi

if [ "$PHASE" = "export" ] || [ "$PHASE" = "all" ]; then
    echo "--- Submitting Export jobs ---"
    for model in "${MODELS[@]}"; do
        if [ "$PHASE" = "all" ]; then
            # Chain after GRPO completion
            GRPO_JOB=${GRPO_JOBS[$model]}
            JOB_ID=$(sbatch --parsable --dependency=afterok:$GRPO_JOB slurm/export_gguf.sh "$model")
            echo "  $model -> Export Job $JOB_ID (after GRPO $GRPO_JOB)"
        else
            # Submit immediately (GRPO assumed done)
            JOB_ID=$(sbatch --parsable slurm/export_gguf.sh "$model")
            echo "  $model -> Export Job $JOB_ID"
        fi
    done
    echo ""
fi

echo "============================================"
echo "All jobs submitted! Monitor with: squeue -u \$USER"
echo "============================================"

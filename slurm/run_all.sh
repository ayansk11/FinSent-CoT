#!/bin/bash
# Submit FinSent-CoT training jobs (SFT -> GRPO -> Export per model)
#
# Usage:
#   bash slurm/run_all.sh              # Submit all 6 original models
#   bash slurm/run_all.sh --small      # Submit only small models (0.6B, 1.5B, 950M)
#   bash slurm/run_all.sh --large      # Submit only large models (1.7B, 4B, 8B)
#   bash slurm/run_all.sh --qwen3.5    # Submit all 4 Qwen3.5 models
#   bash slurm/run_all.sh --all        # Submit all 10 models

set -euo pipefail

cd /N/scratch/ayshaikh/FinSent-CoT
mkdir -p logs

echo "============================================"
echo "FinSent-CoT — Submitting Training Jobs"
echo "============================================"
echo ""

MODE="${1:-all}"

submit() {
    local script="$1"
    local name="$2"
    local job_id
    job_id=$(sbatch "$script" | awk '{print $4}')
    echo "  $name -> Job $job_id"
}

if [ "$MODE" = "--small" ] || [ "$MODE" = "all" ] || [ "$MODE" = "--all" ]; then
    echo "Small models:"
    submit slurm/qwen3_0_6b.sh       "Qwen3-0.6B      (64G, 12h)"
    submit slurm/deepseek_r1_1_5b.sh  "DeepSeek-R1-1.5B (64G, 14h)"
    submit slurm/mobilellm_r1_950m.sh "MobileLLM-R1-950M (64G, 14h)"
    echo ""
fi

if [ "$MODE" = "--large" ] || [ "$MODE" = "all" ] || [ "$MODE" = "--all" ]; then
    echo "Large models:"
    submit slurm/qwen3_1_7b.sh "Qwen3-1.7B (64G, 14h)"
    submit slurm/qwen3_4b.sh   "Qwen3-4B   (80G, 16h)"
    submit slurm/qwen3_8b.sh   "Qwen3-8B   (80G, 20h)"
    echo ""
fi

if [ "$MODE" = "--qwen3.5" ] || [ "$MODE" = "--all" ]; then
    echo "Qwen3.5 models:"
    submit slurm/qwen3_5_0_8b.sh "Qwen3.5-0.8B (64G, 12h)"
    submit slurm/qwen3_5_2b.sh   "Qwen3.5-2B   (64G, 14h)"
    submit slurm/qwen3_5_4b.sh   "Qwen3.5-4B   (80G, 16h)"
    submit slurm/qwen3_5_9b.sh   "Qwen3.5-9B   (80G, 20h)"
    echo ""
fi

echo "All jobs submitted! Monitor with: squeue -u ayshaikh"
echo "Logs: logs/<model>_<jobid>.out/.err"

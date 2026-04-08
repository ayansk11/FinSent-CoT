#!/bin/bash
# Submit FinSenti training jobs (SFT -> GRPO -> Export per model)
#
# Usage:
#   bash slurm/run_all.sh              # Submit all 6 original models
#   bash slurm/run_all.sh --small      # Submit only small models (0.6B, 1.5B, 950M)
#   bash slurm/run_all.sh --large      # Submit only large models (1.7B, 4B, 8B)
#   bash slurm/run_all.sh --qwen3.5    # Submit all 4 Qwen3.5 models
#   bash slurm/run_all.sh --gemma4     # Submit all 3 Gemma 4 models
#   bash slurm/run_all.sh --all        # Submit all 13 models

set -euo pipefail

cd /N/scratch/ayshaikh/FinSent-CoT
mkdir -p logs

echo "============================================"
echo "FinSenti — Submitting Training Jobs"
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
    submit slurm/qwen3_0_6b.sh       "Qwen3-0.6B       (12h)"
    submit slurm/deepseek_r1_1_5b.sh  "DeepSeek-R1-1.5B  (14h)"
    submit slurm/mobilellm_r1_950m.sh "MobileLLM-R1-950M  (14h)"
    echo ""
fi

if [ "$MODE" = "--large" ] || [ "$MODE" = "all" ] || [ "$MODE" = "--all" ]; then
    echo "Large models:"
    submit slurm/qwen3_1_7b.sh "Qwen3-1.7B (14h)"
    submit slurm/qwen3_4b.sh   "Qwen3-4B   (16h)"
    submit slurm/qwen3_8b.sh   "Qwen3-8B   (20h)"
    echo ""
fi

if [ "$MODE" = "--qwen3.5" ] || [ "$MODE" = "--all" ]; then
    echo "Qwen3.5 models:"
    submit slurm/qwen3_5_0_8b.sh "Qwen3.5-0.8B (12h)"
    submit slurm/qwen3_5_2b.sh   "Qwen3.5-2B   (14h)"
    submit slurm/qwen3_5_4b.sh   "Qwen3.5-4B   (16h)"
    submit slurm/qwen3_5_9b.sh   "Qwen3.5-9B   (20h)"
    echo ""
fi

if [ "$MODE" = "--gemma4" ] || [ "$MODE" = "--all" ]; then
    echo "Gemma 4 models:"
    submit slurm/gemma4_e2b.sh      "Gemma4-E2B     (12h, A100)"
    submit slurm/gemma4_e4b.sh      "Gemma4-E4B     (14h, A100)"
    echo "  Gemma4-26B-A4B -> SKIP (submit from Quartz: sbatch slurm/gemma4_26b_a4b.sh)"
    echo ""
fi

echo "All jobs submitted! Monitor with: squeue -u ayshaikh"
echo "Logs: logs/<model>_<jobid>.out/.err"

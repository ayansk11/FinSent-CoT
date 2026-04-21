"""
Submit SLURM jobs that evaluate every FinSenti model + baselines on every
benchmark. Each (model, benchmark) pair is its own job so failures are
isolated and the sweep parallelizes across the cluster.

Output layout:
    evaluation/results/
        finsenti/
            qwen3-0.6b/
                fpb.json
                fiqa.json
                twitterfin.json
                finsenti.json
            qwen3-1.7b/
                ...
        baselines/
            finbert/
                fpb.json
                ...

Usage:
    # Full sweep (all 22 FinSenti + 3 baselines * 4 benchmarks = 100 jobs)
    python evaluation/run_all.py

    # Just one family
    python evaluation/run_all.py --models qwen3-0.6b qwen3-1.7b

    # Skip baselines
    python evaluation/run_all.py --no-baselines

    # Smoke test (50 samples each, local transformers, no SLURM)
    python evaluation/run_all.py --dry-run --max-samples 50 --local
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# FinSenti models to evaluate. The HF repo is the source of truth.
# Keep this list in sync with training/generate_model_cards.py and
# training/create_hf_collection.py.
FINSENTI_MODELS = {
    # key        (hf repo id,                                 short display name)
    "qwen3-0.6b":       ("Ayansk11/FinSenti-Qwen3-0.6B",          "Qwen3-0.6B"),
    "qwen3-1.7b":       ("Ayansk11/FinSenti-Qwen3-1.7B",          "Qwen3-1.7B"),
    "qwen3-4b":         ("Ayansk11/FinSenti-Qwen3-4B",            "Qwen3-4B"),
    "qwen3-8b":         ("Ayansk11/FinSenti-Qwen3-8B",            "Qwen3-8B"),
    "qwen3.5-0.8b":     ("Ayansk11/FinSenti-Qwen3.5-0.8B",        "Qwen3.5-0.8B"),
    "qwen3.5-2b":       ("Ayansk11/FinSenti-Qwen3.5-2B",          "Qwen3.5-2B"),
    "qwen3.5-4b":       ("Ayansk11/FinSenti-Qwen3.5-4B",          "Qwen3.5-4B"),
    "qwen3.5-9b":       ("Ayansk11/FinSenti-Qwen3.5-9B",          "Qwen3.5-9B"),
    "deepseek-r1-1.5b": ("Ayansk11/FinSenti-DeepSeek-R1-1.5B",    "DeepSeek-R1-1.5B"),
    # Added to the list once these repos land:
    "mobilellm-r1-950m":("Ayansk11/FinSenti-MobileLLM-R1-950M",   "MobileLLM-R1-950M"),
    "tiny-llm-10m":     ("Ayansk11/FinSenti-Tiny-LLM-10M",        "Tiny-LLM-10M"),
    "llama-3.2-1b":     ("Ayansk11/FinSenti-Llama-3.2-1B",        "Llama-3.2-1B"),
    "smollm-1.7b":      ("Ayansk11/FinSenti-SmolLM-1.7B",         "SmolLM-1.7B"),
    # Gemma 4 (hopper partition jobs, add when they train)
    "gemma4-e2b":       ("Ayansk11/FinSenti-Gemma4-E2B",          "Gemma4-E2B"),
    "gemma4-e4b":       ("Ayansk11/FinSenti-Gemma4-E4B",          "Gemma4-E4B"),
    "gemma4-26b-a4b":   ("Ayansk11/FinSenti-Gemma4-26B-A4B",      "Gemma4-26B-A4B"),
}

BASELINES = ["finbert", "finbert-tone", "finbert-sentiment"]

BENCHMARKS = ["fpb", "fiqa", "twitterfin", "finsenti"]


# SLURM sbatch script template for a single eval run. Uses --ntasks-per-node=1
# and one GPU per task. Small models finish in <10 min so we don't need big
# walltime.
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=fs-eval-{short_name}-{benchmark}
#SBATCH --account=r01510
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval_{short_name}_{benchmark}_%j.out
#SBATCH --error=logs/eval_{short_name}_{benchmark}_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1

module load python/gpu/3.12.5
module load cudatoolkit/12.6

cd /N/scratch/ayshaikh/FinSent-CoT
mkdir -p logs evaluation/results/{subdir}/{short_name}

if [ -f /N/scratch/ayshaikh/.tokens ]; then
    source /N/scratch/ayshaikh/.tokens
fi

export HF_HOME=/N/scratch/ayshaikh/.cache/huggingface
export TRITON_CACHE_DIR=/N/scratch/ayshaikh/.cache/triton

python {runner} \\
    {runner_args} \\
    --benchmark {benchmark} \\
    --max-samples {max_samples} \\
    --output-json evaluation/results/{subdir}/{short_name}/{benchmark}.json \\
    --no-wandb
"""


def sbatch_for_finsenti(
    key: str, repo_id: str, benchmark: str, max_samples: int | None
) -> str:
    short = key.replace(".", "-")
    return SBATCH_TEMPLATE.format(
        short_name=short,
        benchmark=benchmark,
        subdir="finsenti",
        runner="evaluation/benchmark.py",
        runner_args=f"--backend transformers --model {repo_id}",
        max_samples=max_samples or 10000,
    )


def sbatch_for_baseline(
    name: str, benchmark: str, max_samples: int | None
) -> str:
    return SBATCH_TEMPLATE.format(
        short_name=name,
        benchmark=benchmark,
        subdir="baselines",
        runner="evaluation/baselines.py",
        runner_args=f"--baseline {name}",
        max_samples=max_samples or 10000,
    )


def submit(script: str, dry_run: bool) -> str | None:
    """Write the script to a temp file and sbatch it. Returns the job id."""
    if dry_run:
        print(script)
        print("-" * 70)
        return None
    slurm_dir = Path("slurm/eval_tmp")
    slurm_dir.mkdir(parents=True, exist_ok=True)
    # Hash the first 80 chars of the script name line for a stable filename
    name_line = next(
        (l for l in script.splitlines() if "--job-name=" in l), ""
    ).split("=")[-1].strip()
    path = slurm_dir / f"{name_line}.sh"
    path.write_text(script)
    result = subprocess.run(
        ["sbatch", str(path)], capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr}", file=sys.stderr)
        return None
    # Output is like "Submitted batch job 1234567"
    parts = result.stdout.strip().split()
    return parts[-1] if parts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific FinSenti model keys. Default: all.",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=None,
        choices=BENCHMARKS,
        help="Specific benchmarks. Default: all.",
    )
    parser.add_argument("--no-finsenti", action="store_true",
                        help="Skip FinSenti models (baselines only).")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baselines (FinSenti only).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap per benchmark (for smoke tests).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print SLURM scripts without submitting.")
    args = parser.parse_args()

    models = args.models or list(FINSENTI_MODELS.keys())
    benchmarks = args.benchmarks or BENCHMARKS

    jobs: list[tuple[str, str | None]] = []

    if not args.no_finsenti:
        for key in models:
            if key not in FINSENTI_MODELS:
                print(f"[skip] unknown model key: {key}")
                continue
            repo_id, _ = FINSENTI_MODELS[key]
            for b in benchmarks:
                script = sbatch_for_finsenti(key, repo_id, b, args.max_samples)
                job_id = submit(script, args.dry_run)
                jobs.append((f"{key}/{b}", job_id))

    if not args.no_baselines:
        for name in BASELINES:
            for b in benchmarks:
                script = sbatch_for_baseline(name, b, args.max_samples)
                job_id = submit(script, args.dry_run)
                jobs.append((f"{name}/{b}", job_id))

    print()
    print(f"{'Dry-run' if args.dry_run else 'Submitted'} {len(jobs)} eval jobs:")
    for label, job_id in jobs:
        print(f"  {label:>40s}  {job_id or '(dry-run)'}")


if __name__ == "__main__":
    main()

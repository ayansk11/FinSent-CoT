# FinSent-CoT — Financial Sentiment with Chain-of-Thought

## What This Is
A high-quality financial sentiment analysis model (Qwen3-4B) trained with GRPO using
chain-of-thought reasoning. The model outputs structured `<reasoning>` and `<answer>` tags
so users can understand *why* it classified a given sentiment.

## Author
Ayan Shaikh (ayansk11) — Indiana University

## Repository Structure
```
FinSent-CoT/
├── data_generation/
│   ├── generate_cot_v2.py          # Main CoT data generation (vLLM on H100)
│   ├── validate_dataset.py         # Quality validation & deduplication
│   ├── merge_and_upload.py         # Merge shards + upload to HuggingFace
│   └── sources.py                  # Multi-source dataset loading
├── training/
│   ├── train_sft.py                # SFT warm-up training (W&B tracked)
│   ├── train_grpo.py               # GRPO training with early stopping (W&B tracked)
│   ├── rewards.py                  # 4 reward functions (correctness, format, reasoning, consistency)
│   └── export_gguf.py              # Dual export: Q5_K_M + Q4_K_M (W&B tracked)
├── evaluation/
│   └── benchmark.py                # Full benchmark suite (W&B tracked)
├── slurm/
│   ├── generate_data.sh            # SLURM: Data generation (4x H100, Qwen3-235B-A22B-FP8)
│   ├── train_sft.sh                # SLURM: SFT warm-up (1x H100)
│   ├── train_grpo.sh               # SLURM: GRPO training with early stopping (1x H100)
│   └── export_gguf.sh              # SLURM: Dual GGUF export + HF upload
├── configs/
│   └── tokens_template.sh          # Template for .tokens file (HF + W&B keys)
├── modelfile/
│   └── Modelfile                   # Ollama Modelfile for local deployment
├── CLAUDE.md                       # This file
└── README.md                       # Project documentation
```

## Platform: IU Big Red 200
- **SLURM accounts**: `r01510` (hopper/H100 partition, `--qos=hopper`)
- **User**: `ayshaikh`
- **Project path**: `/N/scratch/ayshaikh/FinSent-CoT`
- **GPUs**: NVIDIA H100 (hopper partition, 4 per node, 80GB each)
- **Home**: `/N/u/ayshaikh/Quartz` (small quota)
- **Scratch**: `/N/scratch/ayshaikh` (large, use for everything)

## Critical: Cache Directory Redirect
All SLURM scripts MUST redirect caches to scratch:
```bash
export HF_HOME=/N/scratch/ayshaikh/.cache/huggingface
export HF_HUB_CACHE=/N/scratch/ayshaikh/.cache/huggingface/hub
export XDG_CACHE_HOME=/N/scratch/ayshaikh/.cache
export TORCH_HOME=/N/scratch/ayshaikh/.cache/torch
export TMPDIR=/N/scratch/ayshaikh/tmp
```

## W&B Tracking
- **Project**: `FinSent-CoT` on wandb.ai/ayansk11/FinSent-CoT
- **All phases log to W&B**: data generation, SFT, GRPO, export, evaluation
- SLURM scripts source WANDB_API_KEY from `/N/scratch/ayshaikh/.tokens`
- GRPO training has early stopping with full evidence logging (reward trajectory, convergence analysis)
- Export logs Q5_K_M vs Q4_K_M comparison table
- Evaluation logs confusion matrix, per-label accuracy, format compliance, sample outputs

## Pipeline
1. **Data Generation**: Qwen3-235B-A22B-FP8 via vLLM on 4x H100 (tensor parallel)
   - Sources: FinGPT + FiQA + Financial PhraseBank
   - Target: 50K unique, validated, balanced CoT samples
   - W&B: tracks valid rate, per-label quality, reasoning length, generation speed
2. **SFT Warm-up**: 3 epochs on full dataset (Qwen3-4B base)
   - W&B: tracks training loss, learning rate
3. **GRPO Training**: up to 3000 steps with early stopping at reward convergence
   - 4 equal-weight reward functions (correctness, format, reasoning, consistency)
   - Early stopping: patience=10, min_delta=0.01, warmup=200 steps
   - W&B: tracks all 4 rewards individually, mean reward, early stop evidence
4. **Export**: Dual GGUF (Q5_K_M recommended + Q4_K_M for comparison)
   - W&B: logs quantization comparison table with sizes
5. **Upload**: HuggingFace (model + dataset) + GitHub

## Key Commands
```bash
# Data generation (4x H100, ~6-8 hours)
sbatch slurm/generate_data.sh

# SFT warm-up (1x H100, ~2-3 hours)
sbatch slurm/train_sft.sh

# GRPO training (1x H100, up to 10hr wall, early stops when converged)
sbatch slurm/train_grpo.sh

# Export dual GGUF + upload (1x H100, ~1-2 hours)
sbatch slurm/export_gguf.sh
```

## HuggingFace Repos
- **Model**: `Ayansk11/FinSent-CoT-Qwen3-4B`
- **Dataset**: `Ayansk11/FinSent-CoT-50k`

## GitHub Repo
- `ayansk11/FinSent-CoT`

## Quantization Decision
- **Q5_K_M (recommended)**: ~2.7GB, ~0.5-1% perplexity loss vs FP16. Best quality for M4 Air 16GB.
- **Q4_K_M (alternative)**: ~2.3GB, ~1.5-2% perplexity loss. Only if 8GB RAM constrained.
- Both are exported and benchmarked side-by-side. Evidence logged to W&B.

## Dependencies
- vLLM (data generation with Qwen3-235B-A22B-FP8)
- unsloth (efficient SFT + GRPO training)
- trl (GRPO trainer)
- wandb (experiment tracking)
- datasets, transformers, huggingface_hub
- llama-cpp-python (GGUF conversion)

"""
GRPO Training for FinSent-CoT.

Trains Qwen3-4B with Group Relative Policy Optimization using 4 equal-weight
reward functions for sentiment correctness, format compliance, reasoning quality,
and reasoning-answer consistency.

Features:
- Early stopping on reward convergence (with full W&B evidence logging)
- Max steps set high (3000) — training stops when rewards plateau
- All 4 reward signals tracked individually per step

Usage:
    python train_grpo.py --sft-checkpoint ./checkpoints/sft --dataset-dir ./validated
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from rewards import (
    consistency_reward,
    format_compliance_reward,
    reasoning_quality_reward,
    sentiment_correctness_reward,
)


SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. Analyze the given financial text and provide:\n"
    "1. Your reasoning in <reasoning> tags\n"
    "2. Your sentiment classification (positive, negative, or neutral) in <answer> tags\n\n"
    "Always use this exact format:\n"
    "<reasoning>\n[Your step-by-step analysis]\n</reasoning>\n"
    "<answer>[positive/negative/neutral]</answer>"
)


# ─── Early Stopping Callback ────────────────────────────────────────────────

class RewardEarlyStoppingCallback(TrainerCallback):
    """
    Monitors mean reward and stops training when it plateaus.

    Convergence is detected when the mean reward does not improve by more than
    `min_delta` for `patience` consecutive eval windows. All evidence is
    logged to W&B for reproducibility and report generation.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.01, warmup_steps: int = 200):
        """
        Args:
            patience: Number of consecutive eval steps without improvement before stopping.
            min_delta: Minimum reward improvement to count as "not plateaued".
            warmup_steps: Don't check for early stopping before this many steps
                          (let the model warm up first).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_steps = warmup_steps

        # Tracking state
        self.best_reward = float("-inf")
        self.best_reward_step = 0
        self.no_improve_count = 0
        self.reward_history = []
        self.should_stop = False
        self.stop_reason = ""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # TRL GRPOTrainer logs reward metrics as "reward" or individual reward function names
        # Look for any reward-related key
        reward_keys = [k for k in logs if "reward" in k.lower()]
        if not reward_keys:
            return

        step = state.global_step

        # Compute mean reward across all reward functions
        reward_vals = [logs[k] for k in reward_keys if isinstance(logs.get(k), (int, float))]
        if not reward_vals:
            return

        mean_reward = sum(reward_vals) / len(reward_vals)
        self.reward_history.append({
            "step": step,
            "mean_reward": mean_reward,
            "individual_rewards": {k: logs[k] for k in reward_keys if isinstance(logs.get(k), (int, float))},
        })

        # Log individual reward tracking to W&B
        wandb.log({
            "early_stop/mean_reward": mean_reward,
            "early_stop/best_reward": self.best_reward,
            "early_stop/no_improve_count": self.no_improve_count,
            "early_stop/patience_remaining": self.patience - self.no_improve_count,
        }, step=step)

        # Don't check before warmup
        if step < self.warmup_steps:
            return

        # Check for improvement
        if mean_reward > self.best_reward + self.min_delta:
            self.best_reward = mean_reward
            self.best_reward_step = step
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        # Check if we should stop
        if self.no_improve_count >= self.patience:
            self.should_stop = True
            self.stop_reason = (
                f"Early stopping triggered at step {step}. "
                f"Mean reward plateaued at {self.best_reward:.4f} (best at step {self.best_reward_step}). "
                f"No improvement > {self.min_delta} for {self.patience} consecutive eval windows."
            )
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING: {self.stop_reason}")
            print(f"{'='*70}\n")
            control.should_training_stop = True

    def log_evidence_to_wandb(self):
        """Log comprehensive early stopping evidence to W&B for reports."""
        evidence = {
            "early_stop/triggered": self.should_stop,
            "early_stop/reason": self.stop_reason if self.should_stop else "Completed all max_steps",
            "early_stop/best_reward": self.best_reward,
            "early_stop/best_reward_step": self.best_reward_step,
            "early_stop/total_reward_checks": len(self.reward_history),
            "early_stop/patience": self.patience,
            "early_stop/min_delta": self.min_delta,
            "early_stop/warmup_steps": self.warmup_steps,
        }
        wandb.summary.update(evidence)

        # Log full reward trajectory as a table
        if self.reward_history:
            columns = ["step", "mean_reward"]
            # Get all individual reward keys from first entry
            if self.reward_history[0].get("individual_rewards"):
                individual_keys = sorted(self.reward_history[0]["individual_rewards"].keys())
                columns.extend(individual_keys)
            else:
                individual_keys = []

            rows = []
            for entry in self.reward_history:
                row = [entry["step"], entry["mean_reward"]]
                for k in individual_keys:
                    row.append(entry.get("individual_rewards", {}).get(k, 0.0))
                rows.append(row)

            wandb.log({
                "early_stop/reward_trajectory": wandb.Table(
                    columns=columns,
                    data=rows,
                ),
            })

        # Log the convergence analysis
        if len(self.reward_history) >= 10:
            # Compare first 10% vs last 10% of rewards
            n = len(self.reward_history)
            first_10pct = self.reward_history[:max(n // 10, 1)]
            last_10pct = self.reward_history[-max(n // 10, 1):]
            first_mean = sum(r["mean_reward"] for r in first_10pct) / len(first_10pct)
            last_mean = sum(r["mean_reward"] for r in last_10pct) / len(last_10pct)

            wandb.summary.update({
                "early_stop/first_10pct_mean_reward": first_mean,
                "early_stop/last_10pct_mean_reward": last_mean,
                "early_stop/total_reward_improvement": last_mean - first_mean,
            })


# ─── Dataset Loading ─────────────────────────────────────────────────────────

def load_grpo_dataset(dataset_dir: str) -> Dataset:
    """Load GRPO training data from validated directory."""
    data_path = Path(dataset_dir) / "grpo_train.jsonl"
    print(f"Loading GRPO dataset from {data_path}...")
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Convert to chat format for GRPO
    formatted = []
    for s in samples:
        formatted.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["prompt"]},
            ],
            "answer": s["label"],  # Ground truth for reward computation
        })

    print(f"Loaded {len(formatted)} training samples")
    return Dataset.from_list(formatted)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training for FinSent-CoT")
    parser.add_argument("--sft-checkpoint", required=True,
                        help="Path to SFT checkpoint")
    parser.add_argument("--dataset-dir", default="./validated",
                        help="Directory with validated GRPO data")
    parser.add_argument("--output-dir", default="./checkpoints/grpo",
                        help="Output directory for GRPO checkpoint")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Max training steps (will early stop before this)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-generations", type=int, default=6,
                        help="Number of generations per prompt for GRPO")
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--eval-steps", type=int, default=50)
    # Early stopping args
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="Eval windows without improvement before stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.01,
                        help="Minimum reward improvement to count as progress")
    parser.add_argument("--early-stop-warmup", type=int, default=200,
                        help="Steps before early stopping can trigger")
    args = parser.parse_args()

    print("=" * 70)
    print("FinSent-CoT GRPO Training (with Early Stopping)")
    print("=" * 70)

    # ─── Initialize W&B ─────────────────────────────────────────────────────
    wandb.init(
        project="FinSent-CoT",
        name=f"grpo-max{args.max_steps}-lr{args.lr}-es",
        tags=["grpo", "rl", "training", "early-stopping"],
        config={
            "phase": "grpo_training",
            "sft_checkpoint": args.sft_checkpoint,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * 2,
            "learning_rate": args.lr,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "lora_r": 32,
            "lora_alpha": 64,
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
            "early_stopping": {
                "enabled": True,
                "patience": args.early_stop_patience,
                "min_delta": args.early_stop_min_delta,
                "warmup_steps": args.early_stop_warmup,
            },
            "reward_functions": [
                "sentiment_correctness (weight=1.0)",
                "format_compliance (weight=1.0)",
                "reasoning_quality (weight=1.0)",
                "consistency (weight=1.0)",
            ],
        },
    )

    # Load model from SFT checkpoint
    print(f"\nLoading model from {args.sft_checkpoint}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.sft_checkpoint,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    dataset = load_grpo_dataset(args.dataset_dir)

    # GRPO config
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=512,
        logging_steps=10,
        save_steps=args.eval_steps,
        save_total_limit=5,
        bf16=True,
        gradient_accumulation_steps=2,
        report_to="wandb",
        run_name=wandb.run.name if wandb.run else "grpo-run",
        seed=42,
    )

    # Create early stopping callback
    early_stop_callback = RewardEarlyStoppingCallback(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        warmup_steps=args.early_stop_warmup,
    )

    # Build trainer with all 4 reward functions + early stopping
    trainer = GRPOTrainer(
        model=model,
        config=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=[
            sentiment_correctness_reward,
            format_compliance_reward,
            reasoning_quality_reward,
            consistency_reward,
        ],
        callbacks=[early_stop_callback],
    )

    print(f"\nStarting GRPO training...")
    print(f"  Max steps: {args.max_steps} (with early stopping)")
    print(f"  Early stop patience: {args.early_stop_patience} eval windows")
    print(f"  Early stop min delta: {args.early_stop_min_delta}")
    print(f"  Early stop warmup: {args.early_stop_warmup} steps")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Generations per prompt: {args.num_generations}")
    print(f"  Reward functions: 4 (correctness, format, reasoning, consistency)")
    print()

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Save final checkpoint
    print(f"\nSaving GRPO checkpoint to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ─── Log comprehensive evidence to W&B ───────────────────────────────────
    actual_steps = trainer.state.global_step

    # Log early stopping evidence
    early_stop_callback.log_evidence_to_wandb()

    # Log training summary
    summary = {
        "actual_steps": actual_steps,
        "max_steps": args.max_steps,
        "steps_saved": args.max_steps - actual_steps,
        "training_time_hours": training_time / 3600,
        "dataset_size": len(dataset),
        "early_stopped": early_stop_callback.should_stop,
    }

    if trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        summary["final_loss"] = last_log.get("loss", None)

    wandb.summary.update(summary)

    # Log human-readable training report
    report_lines = [
        "=" * 70,
        "GRPO TRAINING REPORT",
        "=" * 70,
        f"Actual steps completed: {actual_steps} / {args.max_steps}",
        f"Steps saved by early stopping: {args.max_steps - actual_steps}",
        f"Training time: {training_time/3600:.2f} hours",
        f"Early stopped: {early_stop_callback.should_stop}",
    ]
    if early_stop_callback.should_stop:
        report_lines.append(f"Stop reason: {early_stop_callback.stop_reason}")
    report_lines.append(f"Best mean reward: {early_stop_callback.best_reward:.4f} at step {early_stop_callback.best_reward_step}")
    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")

    wandb.log({"training_report": wandb.Table(
        columns=["field", "value"],
        data=[line.split(": ", 1) if ": " in line else [line, ""] for line in report_lines if line.strip("=")],
    )})

    wandb.finish()
    print("GRPO training complete!")


if __name__ == "__main__":
    main()

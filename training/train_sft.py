"""
SFT Warm-up Training for FinSent-CoT.

Trains Qwen3-4B base model on validated CoT data to learn the structured
<reasoning>/<answer> output format before GRPO optimization.

Usage:
    python train_sft.py --dataset-dir ./validated --output-dir ./checkpoints/sft
"""

import argparse
import json
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. Analyze the given financial text and provide:\n"
    "1. Your reasoning in <reasoning> tags\n"
    "2. Your sentiment classification (positive, negative, or neutral) in <answer> tags\n\n"
    "Always use this exact format:\n"
    "<reasoning>\n[Your step-by-step analysis]\n</reasoning>\n"
    "<answer>[positive/negative/neutral]</answer>"
)


def load_sft_dataset(dataset_dir: str, tokenizer) -> Dataset:
    """Load SFT training data and format as chat messages."""
    data_path = Path(dataset_dir) / "sft_train.jsonl"
    print(f"Loading SFT dataset from {data_path}...")
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    formatted = []
    for s in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["input"]},
            {"role": "assistant", "content": s["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted.append({"text": text})

    print(f"Loaded {len(formatted)} training samples")
    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser(description="SFT warm-up for FinSent-CoT")
    parser.add_argument("--base-model", default="unsloth/Qwen3-4B",
                        help="Base model to fine-tune")
    parser.add_argument("--dataset-dir", default="./validated",
                        help="Directory with validated SFT data")
    parser.add_argument("--output-dir", default="./checkpoints/sft",
                        help="Output directory for SFT checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 70)
    print("FinSent-CoT SFT Warm-up Training")
    print("=" * 70)

    # ─── Initialize W&B ─────────────────────────────────────────────────────
    wandb.init(
        project="FinSent-CoT",
        name=f"sft-{args.base_model.split('/')[-1]}-ep{args.epochs}",
        tags=["sft", "warm-up", "training"],
        config={
            "phase": "sft_training",
            "base_model": args.base_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * 4,
            "learning_rate": args.lr,
            "max_seq_length": args.max_seq_length,
            "lora_r": 32,
            "lora_alpha": 64,
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
        },
    )

    # Load base model
    print(f"\nLoading base model: {args.base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
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
    dataset = load_sft_dataset(args.dataset_dir, tokenizer)

    # SFT config
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="wandb",
        run_name=wandb.run.name if wandb.run else "sft-run",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=config,
        tokenizer=tokenizer,
    )

    print(f"\nStarting SFT training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * 4}")
    print(f"  LR: {args.lr}")
    print(f"  Dataset size: {len(dataset)}")
    print()

    trainer.train()

    # Save
    print(f"\nSaving SFT checkpoint to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Log final metrics to W&B
    wandb.summary.update({
        "final_train_loss": trainer.state.log_history[-1].get("loss", None),
        "total_steps": trainer.state.global_step,
        "dataset_size": len(dataset),
    })
    wandb.finish()
    print("SFT training complete!")


if __name__ == "__main__":
    main()

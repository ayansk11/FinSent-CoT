"""
SFT Warm-up Training for FinSent-CoT — Multi-Model.

Trains any of the 6 supported models on validated CoT data to learn the
structured <reasoning>/<answer> output format before GRPO optimization.

Models 1-5 use Unsloth QLoRA (2-3x faster).
Model 6 (MobileLLM) uses standard PEFT + bitsandbytes.

Usage:
    python train_sft.py --model-key qwen3-4b --dataset-dir ./validated
    python train_sft.py --model-key mobilellm-r1-950m --dataset-dir ./validated
"""

import argparse
import json
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from model_configs import get_config, resolve_model_key, ALL_MODEL_KEYS


SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. Analyze the given financial text and provide:\n"
    "1. Your reasoning in <reasoning> tags\n"
    "2. Your sentiment classification (positive, negative, or neutral) in <answer> tags\n\n"
    "Always use this exact format:\n"
    "<reasoning>\n[Your step-by-step analysis]\n</reasoning>\n"
    "<answer>[positive/negative/neutral]</answer>"
)


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model_unsloth(config: dict, lora_r: int, lora_alpha: int, max_seq: int):
    """Load model with Unsloth QLoRA (Qwen3 / DeepSeek)."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=config["target_modules"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer


def load_model_peft(config: dict, lora_r: int, lora_alpha: int):
    """Load model with standard PEFT + bitsandbytes QLoRA (MobileLLM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=config["target_modules"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ─── Dataset Loading ──────────────────────────────────────────────────────────

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
        # Some tokenizers (MobileLLM) may not have apply_chat_template
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # Fallback: manual formatting for models without chat template
            text = (
                f"### System:\n{SYSTEM_PROMPT}\n\n"
                f"### User:\n{s['input']}\n\n"
                f"### Assistant:\n{s['output']}"
            )
        formatted.append({"text": text})

    print(f"Loaded {len(formatted)} training samples")
    return Dataset.from_list(formatted)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT warm-up for FinSent-CoT (multi-model)")
    parser.add_argument("--model-key", required=True,
                        help=f"Model key: {', '.join(ALL_MODEL_KEYS)}")
    parser.add_argument("--dataset-dir", default="./validated",
                        help="Directory with validated SFT data")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ./checkpoints/sft/<model-key>)")
    # Override config defaults via CLI
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    args = parser.parse_args()

    # Resolve model
    model_key = resolve_model_key(args.model_key)
    config = get_config(model_key)
    sft_cfg = config["sft"]

    # Apply overrides (CLI args take priority over config defaults)
    epochs = args.epochs or sft_cfg["epochs"]
    batch_size = args.batch_size or sft_cfg["batch_size"]
    lr = args.lr or sft_cfg["lr"]
    max_seq_length = args.max_seq_length or config["max_seq_length"]
    lora_r = args.lora_r or sft_cfg["lora_r"]
    lora_alpha = args.lora_alpha or sft_cfg["lora_alpha"]
    grad_accum = sft_cfg["grad_accum"]
    output_dir = args.output_dir or f"./checkpoints/sft/{model_key}"

    print("=" * 70)
    print(f"FinSent-CoT SFT Training — {config['short_name']}")
    print("=" * 70)
    print(f"  Model key:    {model_key}")
    print(f"  Base model:   {config['base_model']}")
    print(f"  Backend:      {'Unsloth QLoRA' if config['use_unsloth'] else 'PEFT + bitsandbytes'}")
    print(f"  LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"  Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print(f"  LR: {lr}, Epochs: {epochs}")
    print(f"  Output: {output_dir}")
    print()

    # ─── Initialize W&B ─────────────────────────────────────────────────────
    wandb.init(
        project="FinSent-CoT",
        name=f"sft-{config['short_name']}-ep{epochs}",
        tags=["sft", "warm-up", "training", model_key, config["model_family"]],
        config={
            "phase": "sft_training",
            "model_key": model_key,
            "base_model": config["base_model"],
            "model_family": config["model_family"],
            "use_unsloth": config["use_unsloth"],
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "learning_rate": lr,
            "max_seq_length": max_seq_length,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
        },
    )

    # Load model (auto-selects Unsloth or PEFT)
    print(f"\nLoading {config['short_name']}...")
    if config["use_unsloth"]:
        model, tokenizer = load_model_unsloth(config, lora_r, lora_alpha, max_seq_length)
    else:
        model, tokenizer = load_model_peft(config, lora_r, lora_alpha)

    # Load dataset
    dataset = load_sft_dataset(args.dataset_dir, tokenizer)

    # SFT config
    training_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        report_to="wandb",
        run_name=wandb.run.name if wandb.run else "sft-run",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_config,
        tokenizer=tokenizer,
    )

    print(f"\nStarting SFT training...")
    print(f"  Dataset size: {len(dataset)}")
    print()

    trainer.train()

    # Save
    print(f"\nSaving SFT checkpoint to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log final metrics to W&B
    wandb.summary.update({
        "final_train_loss": trainer.state.log_history[-1].get("loss", None),
        "total_steps": trainer.state.global_step,
        "dataset_size": len(dataset),
        "model_key": model_key,
    })
    wandb.finish()
    print(f"SFT training complete for {config['short_name']}!")


if __name__ == "__main__":
    main()

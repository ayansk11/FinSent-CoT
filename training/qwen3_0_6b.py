"""
FinSent-CoT — Qwen3-0.6B: SFT -> GRPO -> Export -> Upload

Single self-contained script for the complete training pipeline.
Uses Unsloth for all phases (SFT, GRPO, Export).
Dataset: Ayansk11/FinSent-CoT-Dataset (local validated splits)

Usage:
    python qwen3_0_6b.py --phase all          # Full pipeline
    python qwen3_0_6b.py --phase sft          # SFT only
    python qwen3_0_6b.py --phase grpo         # GRPO only
    python qwen3_0_6b.py --phase export       # Export only
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


# ─── Model Configuration ─────────────────────────────────────────────────────

MODEL_KEY = "qwen3-0.6b"
BASE_MODEL = "unsloth/Qwen3-0.6B"
SHORT_NAME = "Qwen3-0.6B"
MODEL_FAMILY = "qwen3"
MAX_SEQ_LENGTH = 2048
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# SFT hyperparameters
SFT_BATCH_SIZE = 16
SFT_GRAD_ACCUM = 2
SFT_LR = 2e-4
SFT_LORA_R = 16
SFT_LORA_ALPHA = 32
SFT_EPOCHS = 3

# GRPO hyperparameters
GRPO_BATCH_SIZE = 8
GRPO_GRAD_ACCUM = 2
GRPO_LR = 5e-5
GRPO_LORA_R = 16
GRPO_LORA_ALPHA = 32
GRPO_NUM_GENERATIONS = 8
GRPO_MAX_STEPS = 3000
GRPO_MAX_COMPLETION_LENGTH = 512

# HuggingFace repos
HF_FULL = "Ayansk11/FinSent-CoT-Qwen3-0.6B"
HF_REPOS = {
    "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3-0.6B-Q4_K_M",
    "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3-0.6B-Q5_K_M",
    "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3-0.6B-Q8_0",
}
QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

# Paths
SFT_OUTPUT = f"./checkpoints/sft/{MODEL_KEY}"
GRPO_OUTPUT = f"./checkpoints/grpo/{MODEL_KEY}"
EXPORT_OUTPUT = f"./export/{MODEL_KEY}"
DATASET_DIR = "./validated"

SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. Analyze the given financial text and provide:\n"
    "1. Your reasoning in <reasoning> tags\n"
    "2. Your sentiment classification (positive, negative, or neutral) in <answer> tags\n\n"
    "Always use this exact format:\n"
    "<reasoning>\n[Your step-by-step analysis]\n</reasoning>\n"
    "<answer>[positive/negative/neutral]</answer>"
)

MODELFILE_TEMPLATE = """FROM ./{gguf_filename}

# Qwen3 chat template — prefill past <think> to skip empty thinking block
TEMPLATE \\"\\"\\"<|im_start|>system
You are a financial sentiment analyst. Analyze the given financial text and provide:
1. Your reasoning in <reasoning> tags
2. Your sentiment classification (positive, negative, or neutral) in <answer> tags

Always use this exact format:
<reasoning>
[Your step-by-step analysis]
</reasoning>
<answer>[positive/negative/neutral]</answer><|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
<think>
</think>

\\"\\"\\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "</answer>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_start|>"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 1024
PARAMETER num_predict 512
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: SFT
# ═══════════════════════════════════════════════════════════════════════════════

def run_sft():
    import unsloth  # noqa: F401
    import torch
    import wandb
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    print("=" * 70)
    print(f"FinSent-CoT SFT — {SHORT_NAME}")
    print("=" * 70)
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  LoRA:        r={SFT_LORA_R}, alpha={SFT_LORA_ALPHA}")
    print(f"  Batch:       {SFT_BATCH_SIZE} x {SFT_GRAD_ACCUM} = {SFT_BATCH_SIZE * SFT_GRAD_ACCUM}")
    print(f"  LR: {SFT_LR}, Epochs: {SFT_EPOCHS}")
    print()

    wandb.init(
        project="FinSent-CoT",
        name=f"sft-{SHORT_NAME}-ep{SFT_EPOCHS}",
        tags=["sft", "warm-up", MODEL_KEY, MODEL_FAMILY],
        config={
            "phase": "sft", "model_key": MODEL_KEY, "base_model": BASE_MODEL,
            "epochs": SFT_EPOCHS, "batch_size": SFT_BATCH_SIZE,
            "grad_accum": SFT_GRAD_ACCUM, "lr": SFT_LR,
            "lora_r": SFT_LORA_R, "lora_alpha": SFT_LORA_ALPHA,
        },
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=SFT_LORA_R, target_modules=TARGET_MODULES,
        lora_alpha=SFT_LORA_ALPHA, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
    )

    data_path = Path(DATASET_DIR) / "sft_train.jsonl"
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
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted.append({"text": text})
    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} SFT samples")

    trainer = SFTTrainer(
        model=model, train_dataset=dataset,
        args=SFTConfig(
            output_dir=SFT_OUTPUT, num_train_epochs=SFT_EPOCHS,
            per_device_train_batch_size=SFT_BATCH_SIZE,
            gradient_accumulation_steps=SFT_GRAD_ACCUM,
            learning_rate=SFT_LR, lr_scheduler_type="cosine", warmup_ratio=0.1,
            logging_steps=25, save_steps=500, save_total_limit=2,
            bf16=True, max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text", report_to="wandb",
            run_name=wandb.run.name if wandb.run else "sft", seed=42,
        ),
        tokenizer=tokenizer,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    trainer.save_model(SFT_OUTPUT)
    tokenizer.save_pretrained(SFT_OUTPUT)

    try:
        if wandb.run is not None:
            wandb.summary.update({"final_loss": trainer.state.log_history[-1].get("loss"), "training_hours": elapsed / 3600})
            wandb.finish()
    except Exception:
        pass

    print(f"\nSFT complete for {SHORT_NAME}! ({elapsed/3600:.2f}h)")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: GRPO
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_masked_batch_mean():
    import torch

    def _fixed(x, completion_mask):
        min_len = min(x.shape[-1], completion_mask.shape[-1])
        x = x[..., :min_len]
        completion_mask = completion_mask[..., :min_len]
        return (x * completion_mask).sum(dim=-1) / completion_mask.sum(dim=-1).clamp(min=1)

    patched = 0
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        if hasattr(mod, 'masked_batch_mean'):
            setattr(mod, 'masked_batch_mean', _fixed)
            patched += 1
        for attr_name in dir(mod):
            try:
                attr = getattr(mod, attr_name)
                for fn in [attr, getattr(attr, '__func__', None), getattr(attr, '__wrapped__', None)]:
                    if fn and hasattr(fn, '__globals__') and 'masked_batch_mean' in getattr(fn, '__globals__', {}):
                        fn.__globals__['masked_batch_mean'] = _fixed
                        patched += 1
            except Exception:
                pass
    print(f"  [Patch] masked_batch_mean: {patched} locations")
    return patched > 0


def run_grpo():
    import unsloth  # noqa: F401
    import torch
    import wandb
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from rewards import (
        sentiment_correctness_reward, format_compliance_reward,
        reasoning_quality_reward, consistency_reward,
    )
    from callbacks import RewardEarlyStoppingCallback

    _patch_masked_batch_mean()

    print("=" * 70)
    print(f"FinSent-CoT GRPO — {SHORT_NAME}")
    print("=" * 70)
    print(f"  LoRA:  r={GRPO_LORA_R}, alpha={GRPO_LORA_ALPHA}")
    print(f"  Batch: {GRPO_BATCH_SIZE} x {GRPO_GRAD_ACCUM} = {GRPO_BATCH_SIZE * GRPO_GRAD_ACCUM}")
    print(f"  LR: {GRPO_LR}, Gens: {GRPO_NUM_GENERATIONS}, Max steps: {GRPO_MAX_STEPS}")
    print()

    wandb.init(
        project="FinSent-CoT",
        name=f"grpo-{SHORT_NAME}-max{GRPO_MAX_STEPS}-es",
        tags=["grpo", "rl", "early-stopping", MODEL_KEY, MODEL_FAMILY],
        config={
            "phase": "grpo", "model_key": MODEL_KEY, "base_model": BASE_MODEL,
            "max_steps": GRPO_MAX_STEPS, "batch_size": GRPO_BATCH_SIZE,
            "grad_accum": GRPO_GRAD_ACCUM, "lr": GRPO_LR,
            "num_generations": GRPO_NUM_GENERATIONS,
            "lora_r": GRPO_LORA_R, "lora_alpha": GRPO_LORA_ALPHA,
        },
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_OUTPUT, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=True,
    )

    data_path = Path(DATASET_DIR) / "grpo_train.jsonl"
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    formatted = [
        {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": s["prompt"]}], "answer": s["label"]}
        for s in samples
    ]
    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} GRPO samples")

    early_stop = RewardEarlyStoppingCallback(patience=10, min_delta=0.01, warmup_steps=200)

    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(
            output_dir=GRPO_OUTPUT, max_steps=GRPO_MAX_STEPS,
            per_device_train_batch_size=GRPO_BATCH_SIZE,
            gradient_accumulation_steps=GRPO_GRAD_ACCUM,
            learning_rate=GRPO_LR, lr_scheduler_type="cosine", warmup_ratio=0.1,
            num_generations=GRPO_NUM_GENERATIONS,
            max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
            max_prompt_length=512,
            logging_steps=10, save_steps=50, save_total_limit=5,
            bf16=True, report_to="wandb",
            run_name=wandb.run.name if wandb.run else "grpo", seed=42,
        ),
        train_dataset=dataset, tokenizer=tokenizer,
        reward_funcs=[sentiment_correctness_reward, format_compliance_reward, reasoning_quality_reward, consistency_reward],
        callbacks=[early_stop],
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    trainer.save_model(GRPO_OUTPUT)
    tokenizer.save_pretrained(GRPO_OUTPUT)

    steps = trainer.state.global_step
    print(f"\n{'='*70}")
    print(f"GRPO REPORT — {SHORT_NAME}: {steps}/{GRPO_MAX_STEPS} steps, {elapsed/3600:.2f}h")
    print(f"  Early stopped: {early_stop.should_stop}, Best: {early_stop.best_reward:.4f}")
    print(f"{'='*70}")

    try:
        if wandb.run is not None:
            early_stop.log_evidence_to_wandb()
            wandb.summary.update({"actual_steps": steps, "training_hours": elapsed / 3600, "early_stopped": early_stop.should_stop})
            wandb.finish()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Export
# ═══════════════════════════════════════════════════════════════════════════════

def run_export(upload: bool = False):
    import unsloth  # noqa: F401
    import wandb
    from unsloth import FastLanguageModel

    output_dir = Path(EXPORT_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"FinSent-CoT-{SHORT_NAME}"

    print("=" * 70)
    print(f"FinSent-CoT Export — {SHORT_NAME}")
    print("=" * 70)

    wandb.init(project="FinSent-CoT", name=f"export-{SHORT_NAME}", tags=["export", "gguf", MODEL_KEY])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=GRPO_OUTPUT, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=True,
    )

    exports = []
    for quant in QUANTIZATIONS:
        quant_dir = output_dir / quant
        quant_dir.mkdir(parents=True, exist_ok=True)
        gguf_filename = f"{model_name}.{quant}.gguf"

        print(f"\n  Exporting {quant}...")
        start = time.time()
        model.save_pretrained_gguf(str(quant_dir), tokenizer, quantization_method=quant.lower())
        elapsed = time.time() - start

        final_path = None
        for f in quant_dir.glob("*.gguf"):
            if f.name != gguf_filename:
                shutil.move(str(f), str(quant_dir / gguf_filename))
                final_path = quant_dir / gguf_filename
            else:
                final_path = f

        size_gb = round(os.path.getsize(str(final_path)) / (1024**3), 2) if final_path else 0
        exports.append({"quant": quant, "filename": gguf_filename, "size_gb": size_gb, "dir": str(quant_dir)})
        print(f"    -> {gguf_filename}: {size_gb} GB ({elapsed:.0f}s)")

        with open(quant_dir / "Modelfile", "w") as mf:
            mf.write(MODELFILE_TEMPLATE.format(gguf_filename=gguf_filename))

    merged_dir = output_dir / "merged_hf"
    print(f"\nSaving merged HF weights to {merged_dir}...")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    if upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for e in exports:
            repo = HF_REPOS[e["quant"]]
            api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=os.path.join(e["dir"], e["filename"]), path_in_repo=e["filename"], repo_id=repo, repo_type="model")
            api.upload_file(path_or_fileobj=os.path.join(e["dir"], "Modelfile"), path_in_repo="Modelfile", repo_id=repo, repo_type="model")
            print(f"  Uploaded {e['quant']} -> {repo}")
        api.create_repo(repo_id=HF_FULL, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(merged_dir), repo_id=HF_FULL, repo_type="model")
        print(f"  Uploaded HF weights -> {HF_FULL}")

    wandb.finish()
    print(f"\nExport complete for {SHORT_NAME}!")
    for e in exports:
        rec = " (recommended)" if e["quant"] == "Q5_K_M" else ""
        print(f"  {e['quant']}: {e['size_gb']} GB{rec}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=f"FinSent-CoT {SHORT_NAME}: SFT -> GRPO -> Export")
    parser.add_argument("--phase", choices=["sft", "grpo", "export", "all"], default="all")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    phases = ["sft", "grpo", "export"] if args.phase == "all" else [args.phase]
    print(f"\n{'#'*70}")
    print(f"# FinSent-CoT Pipeline — {SHORT_NAME}")
    print(f"# Phases: {' -> '.join(phases)}")
    print(f"{'#'*70}\n")

    for phase in phases:
        if phase == "sft": run_sft()
        elif phase == "grpo": run_grpo()
        elif phase == "export": run_export(upload=args.upload)
        print()


if __name__ == "__main__":
    main()

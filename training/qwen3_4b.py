"""
FinSenti — Qwen3-4B: SFT -> GRPO -> Export -> Upload

Single self-contained script for the complete training pipeline.
Uses Unsloth for all phases (SFT, GRPO, Export).
Dataset: Ayansk11/FinSenti-Dataset (local validated splits)

Usage:
    python qwen3_4b.py --phase all          # Full pipeline
    python qwen3_4b.py --phase sft          # SFT only
    python qwen3_4b.py --phase grpo         # GRPO only (requires SFT checkpoint)
    python qwen3_4b.py --phase export       # Export only (requires GRPO checkpoint)
"""

import a100_compat  # noqa: F401 — must be before unsloth (patches addmm_ for A100)
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def _rebuild_grpo_config(d):
    """Reconstruct GRPOConfig from dict (avoids UnslothGRPOConfig identity issue)."""
    from trl import GRPOConfig
    obj = object.__new__(GRPOConfig)
    obj.__dict__.update(d)
    return obj


def _wandb_init_safe(**kwargs):
    """wandb.init with retry for concurrent SLURM job startup."""
    import wandb as _wb
    import time as _t
    for attempt in range(3):
        try:
            return _wb.init(**kwargs)
        except Exception as e:
            if attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"[wandb] init failed ({e}), retrying in {wait}s...")
                _t.sleep(wait)
            else:
                print(f"[wandb] init failed after 3 attempts, disabling")
                os.environ["WANDB_MODE"] = "disabled"
                return _wb.init(**kwargs)


# ─── Model Configuration ─────────────────────────────────────────────────────

MODEL_KEY = "qwen3-4b"
BASE_MODEL = "unsloth/Qwen3-4B"
SHORT_NAME = "Qwen3-4B"
MODEL_FAMILY = "qwen3"
MAX_SEQ_LENGTH = 2048
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# SFT hyperparameters
SFT_BATCH_SIZE = 4
SFT_GRAD_ACCUM = 4
SFT_LR = 2e-4
SFT_LORA_R = 32
SFT_LORA_ALPHA = 64
SFT_EPOCHS = 3

# GRPO hyperparameters
GRPO_BATCH_SIZE = 4
GRPO_GRAD_ACCUM = 2
GRPO_LR = 5e-5
GRPO_LORA_R = 32
GRPO_LORA_ALPHA = 64
GRPO_NUM_GENERATIONS = 4
GRPO_MAX_STEPS = 3000
GRPO_MAX_COMPLETION_LENGTH = 512

# HuggingFace repos
HF_FULL = "Ayansk11/FinSenti-Qwen3-4B"
HF_GGUF = "Ayansk11/FinSenti-Qwen3-4B-GGUF"
QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]
MLX_REPOS = {
    4: "Ayansk11/FinSenti-Qwen3-4B-MLX-4bit",
    8: "Ayansk11/FinSenti-Qwen3-4B-MLX-8bit",
}

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

# Ollama Modelfile template (Qwen3 chat format)
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

# Critical stop tokens
PARAMETER stop "<|im_end|>"
PARAMETER stop "</answer>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_start|>"

# Generation parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 1024
PARAMETER num_predict 512
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: SFT (Supervised Fine-Tuning)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sft():
    """SFT warm-up training using Unsloth."""
    import unsloth  # noqa: F401 — MUST be first, patches transformers/trl
    import torch
    import wandb
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    print("=" * 70)
    print(f"FinSenti SFT — {SHORT_NAME}")
    print("=" * 70)
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Backend:     Unsloth QLoRA")
    print(f"  LoRA:        r={SFT_LORA_R}, alpha={SFT_LORA_ALPHA}")
    print(f"  Batch:       {SFT_BATCH_SIZE} x {SFT_GRAD_ACCUM} = {SFT_BATCH_SIZE * SFT_GRAD_ACCUM} effective")
    print(f"  LR: {SFT_LR}, Epochs: {SFT_EPOCHS}")
    print(f"  Output:      {SFT_OUTPUT}")
    print()

    # ─── W&B ──────────────────────────────────────────────────────────────
    _wandb_init_safe(
        project="FinSenti",
        name=f"sft-{SHORT_NAME}-ep{SFT_EPOCHS}",
        tags=["sft", "warm-up", MODEL_KEY, MODEL_FAMILY],
        config={
            "phase": "sft",
            "model_key": MODEL_KEY,
            "base_model": BASE_MODEL,
            "epochs": SFT_EPOCHS,
            "batch_size": SFT_BATCH_SIZE,
            "grad_accum": SFT_GRAD_ACCUM,
            "effective_batch_size": SFT_BATCH_SIZE * SFT_GRAD_ACCUM,
            "learning_rate": SFT_LR,
            "lora_r": SFT_LORA_R,
            "lora_alpha": SFT_LORA_ALPHA,
            "max_seq_length": MAX_SEQ_LENGTH,
        },
    )

    # ─── Load model ───────────────────────────────────────────────────────
    print(f"Loading {SHORT_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=SFT_LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=SFT_LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ─── Load dataset (Ayansk11/FinSenti-Dataset, local validated split) ─
    data_path = Path(DATASET_DIR) / "sft_train.jsonl"
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

    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} SFT samples")

    # ─── Train ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=SFT_OUTPUT,
            num_train_epochs=SFT_EPOCHS,
            per_device_train_batch_size=SFT_BATCH_SIZE,
            gradient_accumulation_steps=SFT_GRAD_ACCUM,
            learning_rate=SFT_LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=25,
            save_steps=500,
            save_total_limit=2,
            bf16=True,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            report_to="wandb",
            run_name=wandb.run.name if wandb.run else "sft",
            seed=42,
        ),
        tokenizer=tokenizer,
    )

    print(f"\nStarting SFT training ({len(dataset)} samples, {SFT_EPOCHS} epochs)...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # ─── Save ─────────────────────────────────────────────────────────────
    print(f"\nSaving SFT checkpoint to {SFT_OUTPUT}...")
    trainer.save_model(SFT_OUTPUT)
    tokenizer.save_pretrained(SFT_OUTPUT)

    try:
        if wandb.run is not None:
            wandb.summary.update({
                "final_loss": trainer.state.log_history[-1].get("loss"),
                "total_steps": trainer.state.global_step,
                "training_hours": elapsed / 3600,
                "dataset_size": len(dataset),
            })
            wandb.finish()
    except Exception as e:
        print(f"W&B logging skipped: {e}")

    print(f"\nSFT complete for {SHORT_NAME}! ({elapsed/3600:.2f}h)")
    print(f"Checkpoint: {SFT_OUTPUT}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: GRPO (Group Relative Policy Optimization)
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_masked_batch_mean():
    """Fix Unsloth's masked_batch_mean tensor mismatch (closure — must patch source file)."""
    import importlib
    import importlib.util
    import glob as _glob
    OLD = "return (x * completion_mask).sum() / completion_token_count"
    NEW = (
        "_n = min(x.shape[-1], completion_mask.shape[-1]); "
        "return (x[..., :_n] * completion_mask[..., :_n]).sum() / "
        "completion_mask[..., :_n].sum().clamp(min=1)"
    )
    cache_mod = None
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        src = getattr(mod, '__file__', '') or ''
        if 'GRPOTrainer' in os.path.basename(src) and 'compiled_cache' in src:
            cache_mod = mod
            break
    if cache_mod is None:
        print("  [Patch] Unsloth GRPOTrainer cache not found — skipping")
        return None
    filepath = cache_mod.__file__
    with open(filepath, 'r') as f:
        content = f.read()
    count = content.count(OLD)
    if count == 0:
        print(f"  [Patch] {filepath} — already patched")
        return None
    with open(filepath, 'w') as f:
        f.write(content.replace(OLD, NEW))
    pycache = os.path.join(os.path.dirname(filepath), '__pycache__')
    if os.path.isdir(pycache):
        for pyc in _glob.glob(os.path.join(pycache, '*.pyc')):
            os.remove(pyc)
    # Robust manual reload — importlib.reload fails for dynamically-loaded modules
    mod_name = cache_mod.__name__
    sys.modules.pop(mod_name, None)
    loader = importlib.machinery.SourceFileLoader(mod_name, filepath)
    spec = importlib.util.spec_from_loader(mod_name, loader, origin=filepath)
    cache_mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = cache_mod
    spec.loader.exec_module(cache_mod)
    print(f"  [Patch] Fixed {count} masked_batch_mean occurrence(s) in {filepath}")
    return getattr(cache_mod, 'GRPOTrainer', None) or getattr(cache_mod, 'UnslothGRPOTrainer', None)


def run_grpo():
    """GRPO training using Unsloth's patched GRPOTrainer."""
    # CRITICAL ORDER: import unsloth BEFORE trl so GRPOTrainer gets patched
    import unsloth  # noqa: F401 — patches trl classes in-place
    import torch
    import wandb
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    from rewards import (
        sentiment_correctness_reward,
        format_compliance_reward,
        reasoning_quality_reward,
        consistency_reward,
    )
    from callbacks import RewardEarlyStoppingCallback

    # Apply tensor mismatch fix AFTER imports
    print("Applying Unsloth GRPOTrainer patches...")
    _patched = _patch_masked_batch_mean()
    if _patched is not None:
        GRPOTrainer = _patched

    print("=" * 70)
    print(f"FinSenti GRPO — {SHORT_NAME}")
    print("=" * 70)
    print(f"  Backend:     Unsloth GRPOTrainer")
    print(f"  SFT ckpt:    {SFT_OUTPUT}")
    print(f"  LoRA:        r={GRPO_LORA_R}, alpha={GRPO_LORA_ALPHA}")
    print(f"  Batch:       {GRPO_BATCH_SIZE} x {GRPO_GRAD_ACCUM} = {GRPO_BATCH_SIZE * GRPO_GRAD_ACCUM} effective")
    print(f"  LR: {GRPO_LR}, Generations: {GRPO_NUM_GENERATIONS}")
    print(f"  Max steps:   {GRPO_MAX_STEPS}")
    print(f"  Output:      {GRPO_OUTPUT}")
    print()

    # ─── W&B ──────────────────────────────────────────────────────────────
    _wandb_init_safe(
        project="FinSenti",
        name=f"grpo-{SHORT_NAME}-max{GRPO_MAX_STEPS}-es",
        tags=["grpo", "rl", "early-stopping", MODEL_KEY, MODEL_FAMILY],
        config={
            "phase": "grpo",
            "model_key": MODEL_KEY,
            "base_model": BASE_MODEL,
            "sft_checkpoint": SFT_OUTPUT,
            "max_steps": GRPO_MAX_STEPS,
            "batch_size": GRPO_BATCH_SIZE,
            "grad_accum": GRPO_GRAD_ACCUM,
            "effective_batch_size": GRPO_BATCH_SIZE * GRPO_GRAD_ACCUM,
            "learning_rate": GRPO_LR,
            "num_generations": GRPO_NUM_GENERATIONS,
            "max_completion_length": GRPO_MAX_COMPLETION_LENGTH,
            "lora_r": GRPO_LORA_R,
            "lora_alpha": GRPO_LORA_ALPHA,
            "reward_functions": [
                "sentiment_correctness (1.0)",
                "format_compliance (1.0)",
                "reasoning_quality (1.0)",
                "consistency (1.0)",
            ],
        },
    )

    # ─── Load SFT checkpoint ──────────────────────────────────────────────
    # Unsloth auto-detects existing LoRA adapters and continues training them
    print(f"Loading SFT checkpoint from {SFT_OUTPUT}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_OUTPUT,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # ─── Load GRPO dataset (Ayansk11/FinSenti-Dataset) ────────────────
    data_path = Path(DATASET_DIR) / "grpo_train.jsonl"
    print(f"Loading GRPO dataset from {data_path}...")
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    formatted = []
    for s in samples:
        formatted.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["prompt"]},
            ],
            "answer": s["label"],
        })

    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} GRPO samples")

    # ─── Early stopping callback ──────────────────────────────────────────
    early_stop = RewardEarlyStoppingCallback(
        patience=10,
        min_delta=0.01,
        warmup_steps=200,
    )

    # ─── Train ────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(
            output_dir=GRPO_OUTPUT,
            max_steps=GRPO_MAX_STEPS,
            per_device_train_batch_size=GRPO_BATCH_SIZE,
            gradient_accumulation_steps=GRPO_GRAD_ACCUM,
            learning_rate=GRPO_LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            num_generations=GRPO_NUM_GENERATIONS,
            max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
            max_prompt_length=512, mask_truncated_completions=True,
            logging_steps=10,
            save_steps=50,
            save_total_limit=5,
            bf16=True,
            report_to="wandb",
            run_name=wandb.run.name if wandb.run else "grpo",
            seed=42,
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=[
            sentiment_correctness_reward,
            format_compliance_reward,
            reasoning_quality_reward,
            consistency_reward,
        ],
        callbacks=[early_stop],
    )
    _cfg_cls = type(trainer.args)
    if 'Unsloth' in _cfg_cls.__name__:
        _cfg_cls.__reduce_ex__ = lambda self, protocol: (_rebuild_grpo_config, (self.__dict__.copy(),))

    print(f"\nStarting GRPO training (max {GRPO_MAX_STEPS} steps, early stopping enabled)...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # ─── Save ─────────────────────────────────────────────────────────────
    print(f"\nSaving GRPO checkpoint to {GRPO_OUTPUT}...")
    trainer.save_model(GRPO_OUTPUT)
    tokenizer.save_pretrained(GRPO_OUTPUT)

    # ─── Report ───────────────────────────────────────────────────────────
    steps = trainer.state.global_step
    print(f"\n{'='*70}")
    print(f"GRPO REPORT — {SHORT_NAME}")
    print(f"{'='*70}")
    print(f"  Steps completed: {steps}/{GRPO_MAX_STEPS}")
    print(f"  Steps saved:     {GRPO_MAX_STEPS - steps}")
    print(f"  Training time:   {elapsed/3600:.2f} hours")
    print(f"  Early stopped:   {early_stop.should_stop}")
    if early_stop.should_stop:
        print(f"  Reason: {early_stop.stop_reason}")
    print(f"  Best reward:     {early_stop.best_reward:.4f} (step {early_stop.best_reward_step})")
    print(f"{'='*70}")

    try:
        if wandb.run is not None:
            early_stop.log_evidence_to_wandb()
            wandb.summary.update({
                "actual_steps": steps,
                "max_steps": GRPO_MAX_STEPS,
                "steps_saved": GRPO_MAX_STEPS - steps,
                "training_hours": elapsed / 3600,
                "early_stopped": early_stop.should_stop,
                "best_reward": early_stop.best_reward,
            })
            if trainer.state.log_history:
                wandb.summary["final_loss"] = trainer.state.log_history[-1].get("loss")
            wandb.finish()
    except Exception as e:
        print(f"W&B logging skipped: {e}")

    print(f"\nGRPO complete for {SHORT_NAME}! ({elapsed/3600:.2f}h)")
    print(f"Checkpoint: {GRPO_OUTPUT}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Export (GGUF + HF Upload)
# ═══════════════════════════════════════════════════════════════════════════════

def run_export(upload: bool = False):
    """Export GGUF quantizations and optionally upload to HuggingFace."""
    import unsloth  # noqa: F401
    import torch
    torch.set_autocast_gpu_dtype(torch.bfloat16)  # A100 fix: autocast defaults to fp16
    import wandb
    from unsloth import FastLanguageModel

    output_dir = Path(EXPORT_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"FinSenti-{SHORT_NAME}"

    print("=" * 70)
    print(f"FinSenti Export — {SHORT_NAME}")
    print("=" * 70)
    print(f"  Source:        {GRPO_OUTPUT}")
    print(f"  Quantizations: {', '.join(QUANTIZATIONS)}")
    print(f"  Output:        {output_dir}")
    if upload:
        print(f"  Upload:        {HF_FULL} + 3 GGUF repos")
    print()

    # ─── W&B ──────────────────────────────────────────────────────────────
    _wandb_init_safe(
        project="FinSenti",
        name=f"export-{SHORT_NAME}",
        tags=["export", "gguf", "quantization", MODEL_KEY],
        config={
            "phase": "export",
            "model_key": MODEL_KEY,
            "grpo_checkpoint": GRPO_OUTPUT,
            "quantizations": QUANTIZATIONS,
            "hf_gguf": HF_GGUF,
        },
    )

    # ─── Load GRPO checkpoint ─────────────────────────────────────────────
    print(f"Loading GRPO checkpoint from {GRPO_OUTPUT}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=GRPO_OUTPUT,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # ─── Save merged HF weights ──────────────────────────────────────────
    merged_dir = output_dir / "merged_hf"
    print(f"\nSaving merged HF weights to {merged_dir}...")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    # ─── GGUF conversion via pre-built llama.cpp ──────────────────────────
    import subprocess as _sp
    _convert = shutil.which("convert_hf_to_gguf.py") or str(Path("llama.cpp/convert_hf_to_gguf.py"))
    _quantize = shutil.which("llama-quantize") or str(Path("llama.cpp/build/bin/llama-quantize"))

    bf16_gguf = output_dir / f"{model_name}.bf16.gguf"
    print(f"\n  Converting HF -> GGUF bf16...")
    _sp.run([sys.executable, _convert, str(merged_dir),
             "--outfile", str(bf16_gguf), "--outtype", "bf16"], check=True)

    exports = []
    for quant in QUANTIZATIONS:
        quant_dir = output_dir / quant
        quant_dir.mkdir(parents=True, exist_ok=True)
        gguf_filename = f"{model_name}.{quant}.gguf"
        final_path = quant_dir / gguf_filename

        print(f"\n  Quantizing {quant}...")
        start = time.time()
        _sp.run([_quantize, str(bf16_gguf), str(final_path), quant],
                check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        elapsed = time.time() - start

        size_mb = os.path.getsize(str(final_path)) / (1024 * 1024) if final_path.exists() else 0
        size_gb = round(size_mb / 1024, 2)

        exports.append({
            "quant": quant,
            "filename": gguf_filename,
            "size_gb": size_gb,
            "size_mb": round(size_mb, 1),
            "time_sec": round(elapsed, 1),
            "dir": str(quant_dir),
        })
        print(f"    -> {gguf_filename}: {size_gb} GB ({elapsed:.0f}s)")

        with open(quant_dir / "Modelfile", "w") as mf:
            mf.write(MODELFILE_TEMPLATE.format(gguf_filename=gguf_filename))

    bf16_gguf.unlink(missing_ok=True)

    # MLX export (for vllm-mlx / mlx-lm / mlx-vlm compatibility)
    try:
        from mlx_lm import convert as mlx_convert
        for q_bits, repo in MLX_REPOS.items():
            mlx_dir = output_dir / f"mlx-{q_bits}bit"
            print(f"\n  Converting to MLX {q_bits}-bit...")
            start = time.time()
            mlx_convert(hf_path=str(merged_dir), mlx_path=str(mlx_dir), quantize=True, q_bits=q_bits)
            elapsed = time.time() - start
            mlx_size = sum(os.path.getsize(str(f)) for f in mlx_dir.rglob("*") if f.is_file())
            print(f"    -> {mlx_size / (1024**3):.2f} GB ({elapsed:.0f}s)")
    except ImportError:
        print("\n  [SKIP] mlx-lm not installed — MLX export skipped (run on Apple Silicon)")
    except Exception as e:
        print(f"\n  [WARN] MLX conversion failed: {e}")

    # ─── Log to W&B ──────────────────────────────────────────────────────
    wandb.log({
        "quantization_comparison": wandb.Table(
            columns=["quantization", "size_gb", "size_mb", "export_time_sec"],
            data=[[e["quant"], e["size_gb"], e["size_mb"], e["time_sec"]] for e in exports],
        )
    })
    summary = {"recommended_quantization": "Q5_K_M"}
    for e in exports:
        summary[f"{e['quant'].lower()}_size_gb"] = e["size_gb"]
    wandb.summary.update(summary)

    # ─── Upload to HuggingFace ────────────────────────────────────────────
    if upload:
        from huggingface_hub import HfApi
        api = HfApi()

        print(f"\nUploading to HuggingFace...")

        # Upload GGUF repos (each quant to its own repo)
        api.create_repo(repo_id=HF_GGUF, repo_type="model", exist_ok=True)
        for e in exports:
            api.upload_file(path_or_fileobj=os.path.join(e["dir"], e["filename"]), path_in_repo=e["filename"], repo_id=HF_GGUF, repo_type="model")
            mf = os.path.join(e["dir"], "Modelfile")
            if os.path.exists(mf):
                api.upload_file(path_or_fileobj=mf, path_in_repo=f"Modelfile.{e['quant']}", repo_id=HF_GGUF, repo_type="model")
            print(f"  Uploaded {e['quant']} -> {HF_GGUF}")

        # Upload full-precision HF weights
        api.create_repo(repo_id=HF_FULL, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(merged_dir),
            repo_id=HF_FULL,
            repo_type="model",
        )
        print(f"  Uploaded HF weights -> {HF_FULL}")
        # Upload MLX models
        for q_bits, repo in MLX_REPOS.items():
            mlx_dir = output_dir / f"mlx-{q_bits}bit"
            if mlx_dir.exists():
                api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
                api.upload_folder(folder_path=str(mlx_dir), repo_id=repo, repo_type="model")
                print(f"  Uploaded MLX-{q_bits}bit -> {repo}")

    wandb.finish()

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE — {SHORT_NAME}")
    print(f"{'='*70}")
    for e in exports:
        rec = " (recommended)" if e["quant"] == "Q5_K_M" else ""
        print(f"  {e['quant']}: {e['size_gb']} GB{rec}")
    print(f"  HF weights: {merged_dir}")
    print(f"\nTo create Ollama model (using Q5_K_M):")
    print(f"  cd {output_dir}/Q5_K_M && ollama create finsent-{MODEL_KEY} -f Modelfile")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=f"FinSenti {SHORT_NAME}: SFT -> GRPO -> Export"
    )
    parser.add_argument(
        "--phase",
        choices=["sft", "grpo", "export", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace after export",
    )
    args = parser.parse_args()

    phases = ["sft", "grpo", "export"] if args.phase == "all" else [args.phase]

    print(f"\n{'#'*70}")
    print(f"# FinSenti Pipeline — {SHORT_NAME}")
    print(f"# Phases: {' -> '.join(phases)}")
    print(f"{'#'*70}\n")

    for phase in phases:
        if phase == "sft":
            run_sft()
        elif phase == "grpo":
            run_grpo()
        elif phase == "export":
            run_export(upload=args.upload)
        print()


if __name__ == "__main__":
    main()

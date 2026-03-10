"""
FinSent-CoT — Qwen3.5-4B: SFT -> GRPO -> Export -> Upload

Single self-contained script for the complete training pipeline.
Uses Unsloth for all phases (SFT, GRPO, Export).
Dataset: Ayansk11/FinSent-CoT-Dataset (local validated splits)

Usage:
    python qwen3_5_4b.py --phase all          # Full pipeline
    python qwen3_5_4b.py --phase sft          # SFT only
    python qwen3_5_4b.py --phase grpo         # GRPO only (requires SFT checkpoint)
    python qwen3_5_4b.py --phase export       # Export only (requires GRPO checkpoint)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


# ─── Model Configuration ─────────────────────────────────────────────────────

MODEL_KEY = "qwen3.5-4b"
BASE_MODEL = "unsloth/Qwen3.5-4B"
SHORT_NAME = "Qwen3.5-4B"
MODEL_FAMILY = "qwen3.5"
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
HF_FULL = "Ayansk11/FinSent-CoT-Qwen3.5-4B"
HF_REPOS = {
    "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-4B-Q4_K_M",
    "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-4B-Q5_K_M",
    "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3.5-4B-Q8_0",
}
QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]
MLX_REPOS = {
    4: "Ayansk11/FinSent-CoT-Qwen3.5-4B-MLX-4bit",
    8: "Ayansk11/FinSent-CoT-Qwen3.5-4B-MLX-8bit",
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

MODELFILE_TEMPLATE = """FROM ./{gguf_filename}

# Qwen3.5 chat template — prefill past <think> to skip empty thinking block
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

    wandb.init(project="FinSent-CoT", name=f"sft-{SHORT_NAME}-ep{SFT_EPOCHS}",
               tags=["sft", "warm-up", MODEL_KEY, MODEL_FAMILY],
               config={"phase": "sft", "model_key": MODEL_KEY, "base_model": BASE_MODEL,
                       "epochs": SFT_EPOCHS, "batch_size": SFT_BATCH_SIZE, "lr": SFT_LR,
                       "lora_r": SFT_LORA_R, "lora_alpha": SFT_LORA_ALPHA})

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH, dtype=torch.bfloat16, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=SFT_LORA_R, target_modules=TARGET_MODULES,
        lora_alpha=SFT_LORA_ALPHA, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth")

    data_path = Path(DATASET_DIR) / "sft_train.jsonl"
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line))
    formatted = []
    for s in samples:
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["input"]},
            {"role": "assistant", "content": s["output"]},
        ], tokenize=False, add_generation_prompt=False)
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
            run_name=wandb.run.name if wandb.run else "sft", seed=42),
        tokenizer=tokenizer)

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    trainer.save_model(SFT_OUTPUT)
    tokenizer.save_pretrained(SFT_OUTPUT)
    try:
        if wandb.run: wandb.summary.update({"final_loss": trainer.state.log_history[-1].get("loss"), "training_hours": elapsed/3600}); wandb.finish()
    except Exception: pass
    print(f"\nSFT complete for {SHORT_NAME}! ({elapsed/3600:.2f}h)")


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
    import unsloth  # noqa: F401
    import torch
    import wandb
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from rewards import sentiment_correctness_reward, format_compliance_reward, reasoning_quality_reward, consistency_reward
    from callbacks import RewardEarlyStoppingCallback

    # Monkey-patch Qwen3.5 compute_3d_position_ids to handle empty delta tensor
    # (transformers 5.2.0 bug: delta has size 0 when completions are empty)
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as _q35
        _orig_3d = _q35.compute_3d_position_ids
        def _safe_3d(*a, **kw):
            try:
                return _orig_3d(*a, **kw)
            except RuntimeError:
                import torch as _t
                input_ids = a[0] if a else kw.get("input_ids")
                bs, seq = input_ids.shape
                return _t.arange(seq, device=input_ids.device).unsqueeze(0).expand(bs, -1)
        _q35.compute_3d_position_ids = _safe_3d
        print("  [Patch] Monkey-patched Qwen3.5 compute_3d_position_ids")
    except (ImportError, AttributeError):
        pass

    _patched = _patch_masked_batch_mean()
    if _patched is not None:
        GRPOTrainer = _patched

    print("=" * 70)
    print(f"FinSent-CoT GRPO — {SHORT_NAME}")
    print("=" * 70)
    print(f"  LoRA:  r={GRPO_LORA_R}, alpha={GRPO_LORA_ALPHA}")
    print(f"  Batch: {GRPO_BATCH_SIZE} x {GRPO_GRAD_ACCUM} = {GRPO_BATCH_SIZE * GRPO_GRAD_ACCUM}")
    print(f"  LR: {GRPO_LR}, Gens: {GRPO_NUM_GENERATIONS}, Max steps: {GRPO_MAX_STEPS}")
    print()

    wandb.init(project="FinSent-CoT", name=f"grpo-{SHORT_NAME}-max{GRPO_MAX_STEPS}-es",
               tags=["grpo", "rl", "early-stopping", MODEL_KEY, MODEL_FAMILY],
               config={"phase": "grpo", "model_key": MODEL_KEY, "max_steps": GRPO_MAX_STEPS,
                       "batch_size": GRPO_BATCH_SIZE, "lr": GRPO_LR,
                       "num_generations": GRPO_NUM_GENERATIONS, "lora_r": GRPO_LORA_R})

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_OUTPUT, max_seq_length=MAX_SEQ_LENGTH, dtype=torch.bfloat16, load_in_4bit=True)

    data_path = Path(DATASET_DIR) / "grpo_train.jsonl"
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line))
    dataset = Dataset.from_list([
        {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": s["prompt"]}], "answer": s["label"]}
        for s in samples
    ])
    print(f"Loaded {len(dataset)} GRPO samples")

    early_stop = RewardEarlyStoppingCallback(patience=10, min_delta=0.01, warmup_steps=200)

    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(
            output_dir=GRPO_OUTPUT, max_steps=GRPO_MAX_STEPS,
            per_device_train_batch_size=GRPO_BATCH_SIZE,
            gradient_accumulation_steps=GRPO_GRAD_ACCUM,
            learning_rate=GRPO_LR, lr_scheduler_type="cosine", warmup_ratio=0.1,
            num_generations=GRPO_NUM_GENERATIONS, max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
            max_prompt_length=512, mask_truncated_completions=True,
            logging_steps=10, save_steps=50, save_total_limit=5,
            bf16=True, report_to="wandb", run_name=wandb.run.name if wandb.run else "grpo", seed=42),
        train_dataset=dataset, tokenizer=tokenizer,
        reward_funcs=[sentiment_correctness_reward, format_compliance_reward, reasoning_quality_reward, consistency_reward],
        callbacks=[early_stop])

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
        if wandb.run: early_stop.log_evidence_to_wandb(); wandb.summary.update({"actual_steps": steps, "training_hours": elapsed/3600}); wandb.finish()
    except Exception: pass


def run_export(upload=False):
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
        model_name=GRPO_OUTPUT, max_seq_length=MAX_SEQ_LENGTH, dtype=torch.bfloat16, load_in_4bit=True)

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
            if f.name != gguf_filename: shutil.move(str(f), str(quant_dir / gguf_filename)); final_path = quant_dir / gguf_filename
            else: final_path = f
        size_gb = round(os.path.getsize(str(final_path)) / (1024**3), 2) if final_path else 0
        exports.append({"quant": quant, "filename": gguf_filename, "size_gb": size_gb, "dir": str(quant_dir)})
        print(f"    -> {gguf_filename}: {size_gb} GB ({elapsed:.0f}s)")
        with open(quant_dir / "Modelfile", "w") as mf: mf.write(MODELFILE_TEMPLATE.format(gguf_filename=gguf_filename))

    merged_dir = output_dir / "merged_hf"
    print(f"\nSaving merged HF weights...")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

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
        # Upload MLX models
        for q_bits, repo in MLX_REPOS.items():
            mlx_dir = output_dir / f"mlx-{q_bits}bit"
            if mlx_dir.exists():
                api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
                api.upload_folder(folder_path=str(mlx_dir), repo_id=repo, repo_type="model")
                print(f"  Uploaded MLX-{q_bits}bit -> {repo}")

    wandb.finish()
    print(f"\nExport complete for {SHORT_NAME}!")
    for e in exports:
        rec = " (recommended)" if e["quant"] == "Q5_K_M" else ""
        print(f"  {e['quant']}: {e['size_gb']} GB{rec}")


def main():
    parser = argparse.ArgumentParser(description=f"FinSent-CoT {SHORT_NAME}: SFT -> GRPO -> Export")
    parser.add_argument("--phase", choices=["sft", "grpo", "export", "all"], default="all")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    phases = ["sft", "grpo", "export"] if args.phase == "all" else [args.phase]
    print(f"\n{'#'*70}\n# FinSent-CoT Pipeline — {SHORT_NAME}\n# Phases: {' -> '.join(phases)}\n{'#'*70}\n")
    for phase in phases:
        if phase == "sft": run_sft()
        elif phase == "grpo": run_grpo()
        elif phase == "export": run_export(upload=args.upload)
        print()

if __name__ == "__main__":
    main()

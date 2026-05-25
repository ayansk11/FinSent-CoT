"""
FinSenti - MobileLLM-R1-950M: SFT -> GRPO -> Export -> Upload

Single self-contained script for the complete training pipeline.
Uses standard PEFT + bitsandbytes (Unsloth does not support MobileLLM arch).
Export: merges PEFT adapters and saves HF weights. GGUF conversion requires
manual llama.cpp convert_hf_to_gguf.py.

Dataset: Ayansk11/FinSenti-Dataset (local validated splits)

Usage:
    python tiny_llm_10m.py --phase all          # Full pipeline
    python tiny_llm_10m.py --phase sft          # SFT only
    python tiny_llm_10m.py --phase grpo         # GRPO only
    python tiny_llm_10m.py --phase export       # Export only
"""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

# Ensure rewards.py and callbacks.py (in same dir) are importable from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))


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

MODEL_KEY = "tiny-llm-10m"
BASE_MODEL = "arnir0/Tiny-LLM"
SHORT_NAME = "Tiny-LLM-10M"
MODEL_FAMILY = "tinyllm"
# Tiny-LLM has context length 1024 only
MAX_SEQ_LENGTH = 1024
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
# Full-tensor saves (not LoRA) — see mobilellm_r1_950m.py for rationale.
MODULES_TO_SAVE = ["lm_head", "embed_tokens"]

# SFT hyperparameters
# Tiny-LLM uses Llama's 128k vocabulary even though model is only 13M params.
# Logits tensor (batch × seq=1024 × vocab=128k × 4 bytes) OOMs at batch=32.
# Reduce to batch=2, grad_accum=16 (effective batch=32).
SFT_BATCH_SIZE = 2
SFT_GRAD_ACCUM = 16
SFT_LR = 2e-4
SFT_LORA_R = 32
SFT_LORA_ALPHA = 64
SFT_EPOCHS = 5  # bumped from 3 -> 5: 10M model needs more passes to learn format

# GRPO hyperparameters - re-tuned for Tiny-LLM (10M params + 128K vocab).
# Earlier run hung with 4 gens * 512 tokens; the cap below stays conservative
# but bumps max steps from 800 -> 2000 to give the model more time to settle
# into the trained format. Tiny-LLM is a scaling-floor reference and may
# never reach high format compliance, but a longer GRPO budget gives it a
# fair shot.
GRPO_BATCH_SIZE = 2
GRPO_GRAD_ACCUM = 4
GRPO_LR = 5e-5
GRPO_LORA_R = 16
GRPO_LORA_ALPHA = 32
GRPO_NUM_GENERATIONS = 2          # was 4
GRPO_MAX_STEPS = 2000             # bumped from 800 -> 2000
GRPO_MAX_COMPLETION_LENGTH = 256  # was 512

# HuggingFace repos
HF_FULL = "Ayansk11/FinSenti-Tiny-LLM-10M"
HF_GGUF = "Ayansk11/FinSenti-Tiny-LLM-10M-GGUF"
QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]
MLX_REPOS = {
    4: "Ayansk11/FinSenti-Tiny-LLM-10M-MLX-4bit",
    8: "Ayansk11/FinSenti-Tiny-LLM-10M-MLX-8bit",
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

# Generic Modelfile (MobileLLM doesn't use Qwen chat template)
MODELFILE_TEMPLATE = """FROM ./{gguf_filename}

SYSTEM \\"\\"\\"You are a financial sentiment analyst. Analyze the given financial text and provide:
1. Your reasoning in <reasoning> tags
2. Your sentiment classification (positive, negative, or neutral) in <answer> tags

Always use this exact format:
<reasoning>
[Your step-by-step analysis]
</reasoning>
<answer>[positive/negative/neutral]</answer>\\"\\"\\"

PARAMETER stop "</answer>"
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 1024
PARAMETER num_predict 512
"""


# ─── Shared: PEFT model loading ──────────────────────────────────────────────

def _setup_pad_token(tokenizer, model):
    """Add a dedicated pad token if pad collides with eos.

    Also resizes lm_head when it's been untied from embed_tokens, otherwise
    LoRA on lm_head crashes with a shape mismatch (resize_token_embeddings
    only touches input embeddings when lm_head is a separate tensor).
    """
    import torch
    import torch.nn as nn
    needs_new_pad = (
        tokenizer.pad_token is None
        or (tokenizer.eos_token_id is not None and tokenizer.pad_token_id == tokenizer.eos_token_id)
    )
    if needs_new_pad:
        tokenizer.add_special_tokens({"pad_token": "<|finsenti_pad|>"})
        if model is not None:
            new_vocab = len(tokenizer)
            model.resize_token_embeddings(new_vocab)
            for inner in (model, getattr(model, 'model', None),
                          getattr(getattr(model, 'model', None), 'model', None)):
                if inner is None or not hasattr(inner, 'lm_head'):
                    continue
                lm_head = inner.lm_head
                if lm_head.weight.shape[0] != new_vocab:
                    old_w = lm_head.weight.data
                    new_lm = nn.Linear(
                        old_w.shape[1], new_vocab, bias=lm_head.bias is not None,
                    ).to(old_w.device, old_w.dtype)
                    with torch.no_grad():
                        new_lm.weight[:old_w.shape[0]].copy_(old_w)
                        if new_vocab > old_w.shape[0]:
                            new_lm.weight[old_w.shape[0]:] = old_w.mean(dim=0, keepdim=True)
                    inner.lm_head = new_lm
                    print(f"  [Fix] Manually resized lm_head {old_w.shape[0]} -> {new_vocab}")
                break
        print(f"  [Fix] Added dedicated pad_token (id={tokenizer.pad_token_id}, eos_id={tokenizer.eos_token_id})")
    else:
        print(f"  [Info] pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")


def _untie_lm_head(model):
    """Untie lm_head from embed_tokens. Required to avoid NaN gradients with PEFT."""
    import torch.nn as nn
    inner = model
    lm_head_owner = None
    while inner is not None:
        if hasattr(inner, 'lm_head'):
            lm_head_owner = inner
            break
        inner = getattr(inner, 'model', None)
    if lm_head_owner is None:
        print("  [Warn] Could not find lm_head - skipping untie")
        return
    embed = getattr(lm_head_owner, 'embed_tokens', None) \
        or getattr(getattr(lm_head_owner, 'model', None), 'embed_tokens', None)
    if embed is None:
        print("  [Warn] Could not find embed_tokens - skipping untie")
        return
    if lm_head_owner.lm_head.weight.data_ptr() == embed.weight.data_ptr():
        new_weight = nn.Parameter(embed.weight.detach().clone())
        lm_head_owner.lm_head.weight = new_weight
        print("  [Fix] Untied lm_head.weight from embed_tokens.weight (independent params)")
    else:
        print("  [Info] lm_head and embed_tokens already untied")


def _load_base_model_peft(base_model: str, lora_r: int, lora_alpha: int):
    """Load Tiny-LLM in full bf16 (no quantization) + PEFT LoRA.

    Same approach as MobileLLM: 4-bit BnB + weight tying caused NaN gradients.
    Tiny-LLM is only 10M params (~20 MB in bf16) so quantization is unnecessary.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Untie FIRST so _setup_pad_token can resize lm_head independently
    # of embed_tokens. If we resize first, transformers' resize only touches
    # input embeddings for the tied case, leaving lm_head at old vocab size,
    # which then crashes when LoRA-on-lm_head sees the mismatch in forward.
    _untie_lm_head(model)
    _setup_pad_token(tokenizer, model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        modules_to_save=MODULES_TO_SAVE,  # full-tensor save for lm_head+embeds
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def _load_peft_checkpoint(checkpoint_path: str, lora_r: int, lora_alpha: int):
    """Load PEFT checkpoint in bf16, merge SFT adapters, add fresh GRPO LoRA.

    Uses the explicit base+resize+adapter reload pattern (per PEFT issue #1457
    + transformers #36550) to avoid the AutoPeftModelForCausalLM auto-resize
    breaking on modules_to_save'd lm_head/embed_tokens with new vocab.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Tiny-LLM (arnir0/Tiny-LLM) ships without a chat template. SFT used a
    # Llama-4-style fallback format directly as a string. TRL's GRPOTrainer
    # passes prompts as a list of {role, content} dicts and unconditionally
    # calls tokenizer.apply_chat_template, which raises ValueError if no
    # template is set. So we install the same Llama-4-style format the SFT
    # phase trained on as a Jinja template, ensuring GRPO prompts match
    # SFT prompts exactly.
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "<|header_start|>{{ message['role'] }}<|header_end|>\n\n"
            "{{ message['content'] }}<|eot|>"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|header_start|>assistant<|header_end|>\n\n"
            "{% endif %}"
        )
        print("  [Fix] Installed Llama-4-style chat template (matches SFT format)")

    # Load fresh base, untie+resize, THEN load adapter on top (avoids
    # AutoPeftModelForCausalLM auto-resize size-mismatch).
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    _untie_lm_head(base)
    _setup_pad_token(tokenizer, base)
    model = PeftModel.from_pretrained(
        base, checkpoint_path,
        is_trainable=False,
        torch_dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()
    _untie_lm_head(model)
    _setup_pad_token(tokenizer, model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        modules_to_save=MODULES_TO_SAVE,  # full-tensor save for lm_head+embeds
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: SFT
# ═══════════════════════════════════════════════════════════════════════════════

def run_sft():
    import torch
    torch.set_autocast_gpu_dtype(torch.bfloat16)  # A100 fix: autocast defaults to fp16
    import wandb
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print("=" * 70)
    print(f"FinSenti SFT - {SHORT_NAME}")
    print("=" * 70)
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Backend:     PEFT + bitsandbytes (no Unsloth)")
    print(f"  LoRA:        r={SFT_LORA_R}, alpha={SFT_LORA_ALPHA}")
    print(f"  Batch:       {SFT_BATCH_SIZE} x {SFT_GRAD_ACCUM} = {SFT_BATCH_SIZE * SFT_GRAD_ACCUM}")
    print(f"  LR: {SFT_LR}, Epochs: {SFT_EPOCHS}")
    print()

    _wandb_init_safe(
        project="FinSenti",
        name=f"sft-{SHORT_NAME}-ep{SFT_EPOCHS}",
        tags=["sft", "warm-up", MODEL_KEY, MODEL_FAMILY, "peft"],
        config={
            "phase": "sft", "model_key": MODEL_KEY, "base_model": BASE_MODEL,
            "epochs": SFT_EPOCHS, "batch_size": SFT_BATCH_SIZE, "lr": SFT_LR,
            "lora_r": SFT_LORA_R, "lora_alpha": SFT_LORA_ALPHA,
            "backend": "peft+bnb",
        },
    )

    model, tokenizer = _load_base_model_peft(BASE_MODEL, SFT_LORA_R, SFT_LORA_ALPHA)

    # Load dataset
    data_path = Path(DATASET_DIR) / "sft_train.jsonl"
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Test if apply_chat_template works for this tokenizer
    _test_msgs = [{"role": "user", "content": "test"}]
    _has_chat_template = False
    try:
        _test = tokenizer.apply_chat_template(_test_msgs, tokenize=False)
        _has_chat_template = bool(_test and len(_test) > 10)
        print(f"  Chat template: {'available' if _has_chat_template else 'NOT available'}")
    except Exception as e:
        print(f"  Chat template: NOT available ({e})")

    formatted = []
    for s in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["input"]},
            {"role": "assistant", "content": s["output"]},
        ]
        if _has_chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Llama 4 format fallback (MobileLLM uses llama4_text architecture)
            text = (
                f"<|header_start|>system<|header_end|>\n\n{SYSTEM_PROMPT}<|eot|>"
                f"<|header_start|>user<|header_end|>\n\n{s['input']}<|eot|>"
                f"<|header_start|>assistant<|header_end|>\n\n{s['output']}<|eot|>"
            )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} SFT samples")
    # Debug: show first formatted sample to verify template
    print(f"  Sample 0 (first 300 chars): {formatted[0]['text'][:300]}")

    # TRL >=0.24: tokenizer renamed to processing_class
    import inspect
    _sft_trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    _tok_key = "tokenizer" if "tokenizer" in _sft_trainer_params else "processing_class"

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=SFT_OUTPUT,
            max_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
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
            report_to="wandb",
            run_name=wandb.run.name if wandb.run else "sft",
            seed=42,
        ),
        **{_tok_key: tokenizer},
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    trainer.save_model(SFT_OUTPUT)
    tokenizer.save_pretrained(SFT_OUTPUT)

    try:
        if wandb.run is not None:
            wandb.summary.update({
                "final_loss": trainer.state.log_history[-1].get("loss"),
                "training_hours": elapsed / 3600,
            })
            wandb.finish()
    except Exception:
        pass

    print(f"\nSFT complete for {SHORT_NAME}! ({elapsed/3600:.2f}h)")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: GRPO
# ═══════════════════════════════════════════════════════════════════════════════

def run_grpo():
    import torch
    torch.set_autocast_gpu_dtype(torch.bfloat16)  # A100 fix: autocast defaults to fp16
    import wandb
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    # Load rewards/callbacks via importlib (bypasses sys.path entirely).
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    def _load_local(_name):
        _spec = _ilu.spec_from_file_location(_name, _here / f"{_name}.py")
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod
    _rewards = _load_local("rewards")
    _callbacks = _load_local("callbacks")
    sentiment_correctness_reward = _rewards.sentiment_correctness_reward
    format_compliance_reward = _rewards.format_compliance_reward
    reasoning_quality_reward = _rewards.reasoning_quality_reward
    consistency_reward = _rewards.consistency_reward
    RewardEarlyStoppingCallback = _callbacks.RewardEarlyStoppingCallback

    print("=" * 70)
    print(f"FinSenti GRPO - {SHORT_NAME}")
    print("=" * 70)
    print(f"  Backend:     Standard TRL GRPOTrainer (no Unsloth)")
    print(f"  LoRA:        r={GRPO_LORA_R}, alpha={GRPO_LORA_ALPHA}")
    print(f"  Batch:       {GRPO_BATCH_SIZE} x {GRPO_GRAD_ACCUM} = {GRPO_BATCH_SIZE * GRPO_GRAD_ACCUM}")
    print(f"  LR: {GRPO_LR}, Gens: {GRPO_NUM_GENERATIONS}, Max steps: {GRPO_MAX_STEPS}")
    print()

    _wandb_init_safe(
        project="FinSenti",
        name=f"grpo-{SHORT_NAME}-max{GRPO_MAX_STEPS}-es",
        tags=["grpo", "rl", "early-stopping", MODEL_KEY, MODEL_FAMILY, "peft"],
        config={
            "phase": "grpo", "model_key": MODEL_KEY, "max_steps": GRPO_MAX_STEPS,
            "batch_size": GRPO_BATCH_SIZE, "lr": GRPO_LR,
            "num_generations": GRPO_NUM_GENERATIONS, "lora_r": GRPO_LORA_R,
            "backend": "peft+bnb",
        },
    )

    # Load SFT checkpoint (merge SFT adapters, add fresh GRPO LoRA)
    print(f"Loading SFT checkpoint from {SFT_OUTPUT}...")
    model, tokenizer = _load_peft_checkpoint(SFT_OUTPUT, GRPO_LORA_R, GRPO_LORA_ALPHA)

    # Load GRPO dataset
    data_path = Path(DATASET_DIR) / "grpo_train.jsonl"
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    dataset = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["prompt"]},
            ],
            "answer": s["label"],
        }
        for s in samples
    ])
    print(f"Loaded {len(dataset)} GRPO samples")

    early_stop = RewardEarlyStoppingCallback(patience=10, min_delta=0.01, warmup_steps=200)

    # Detect TRL API version (config= vs args=, tokenizer= vs processing_class=)
    _grpo_params = inspect.signature(GRPOTrainer.__init__).parameters
    _config_key = "config" if "config" in _grpo_params else "args"

    trainer_kwargs = {
        "model": model,
        _config_key: GRPOConfig(
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
        "train_dataset": dataset,
        "reward_funcs": [
            sentiment_correctness_reward,
            format_compliance_reward,
            reasoning_quality_reward,
            consistency_reward,
        ],
        "callbacks": [early_stop],
    }

    if "tokenizer" in _grpo_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in _grpo_params:
        trainer_kwargs["processing_class"] = tokenizer

    # TRL GRPOTrainer expects model.warnings_issued dict, but PEFT-wrapped
    # models delegate __getattr__ to the base model which lacks it.
    # Set on the underlying model so PEFT's attribute chain finds it.
    #
    # IMPORTANT: don't use `while hasattr(m, "base_model"): m = m.base_model`.
    # PEFT issue #1892 documents that `base_model` is a property that can
    # return self in certain init states, leading to an infinite loop. That
    # was the root cause of jobs 6914010/6914011 hanging at "Loaded N GRPO
    # samples" for 36-48h. Use PEFT's get_base_model() which is bounded.
    print("  [debug] resolving base model for warnings_issued...", flush=True)
    _base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Drop one more level if there's a `.model` attribute (e.g. LlamaForCausalLM.model -> LlamaModel)
    if hasattr(_base, "model") and not isinstance(getattr(_base, "model", None), str):
        _base = _base.model
    if not hasattr(_base, "warnings_issued"):
        _base.warnings_issued = {}
    model.warnings_issued = _base.warnings_issued
    print("  [debug] warnings_issued wired up", flush=True)

    # Force-disable kv cache: required for PEFT + GRPO; otherwise the
    # "Caching is incompatible with gradient checkpointing" warning loop
    # documented in TRL #3683 can stall the rollout loop.
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(_base, "config"):
        _base.config.use_cache = False

    # Disable gradient checkpointing before GRPO. For the 10M model with
    # small batch the memory save is irrelevant, and gradient checkpointing
    # + PEFT + bitsandbytes is a known source of silent hangs in TRL's
    # rollout loop (job 6900040 sat for 20h with zero CPU activity after
    # "Loaded 15249 GRPO samples").
    if hasattr(model, "gradient_checkpointing_disable"):
        print("  [debug] disabling gradient checkpointing for GRPO", flush=True)
        model.gradient_checkpointing_disable()

    # Warm generation probe: if model.generate() itself deadlocks with
    # PEFT+bnb we want to know in seconds, not hours.
    print("  [debug] warm-generate probe (max_new_tokens=16)...", flush=True)
    _probe_input = tokenizer(
        "Headline: Apple beats Q4 estimates.\n",
        return_tensors="pt",
    ).to(model.device)
    import torch as _t
    with _t.no_grad():
        _probe_out = model.generate(
            **_probe_input, max_new_tokens=16, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(f"  [debug] warm-generate OK ({_probe_out.shape[1]} tokens)", flush=True)

    print(f"  [debug] building GRPOTrainer ({GRPO_NUM_GENERATIONS} gens x "
          f"{GRPO_MAX_COMPLETION_LENGTH} completion tokens, max {GRPO_MAX_STEPS} steps)",
          flush=True)
    trainer = GRPOTrainer(**trainer_kwargs)
    print("  [debug] GRPOTrainer ready; calling trainer.train() ...", flush=True)

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"  [debug] GRPO training finished in {elapsed:.0f}s", flush=True)

    trainer.save_model(GRPO_OUTPUT)
    tokenizer.save_pretrained(GRPO_OUTPUT)

    steps = trainer.state.global_step
    print(f"\n{'='*70}")
    print(f"GRPO REPORT - {SHORT_NAME}: {steps}/{GRPO_MAX_STEPS} steps, {elapsed/3600:.2f}h")
    print(f"  Early stopped: {early_stop.should_stop}, Best: {early_stop.best_reward:.4f}")
    print(f"{'='*70}")

    try:
        if wandb.run is not None:
            early_stop.log_evidence_to_wandb()
            wandb.summary.update({
                "actual_steps": steps,
                "training_hours": elapsed / 3600,
                "early_stopped": early_stop.should_stop,
            })
            wandb.finish()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Export (PEFT merge - no Unsloth GGUF)
# ═══════════════════════════════════════════════════════════════════════════════

def run_export(upload=False):
    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    output_dir = Path(EXPORT_OUTPUT)
    merged_dir = output_dir / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FinSenti Export - {SHORT_NAME}")
    print("=" * 70)
    print(f"  Method:    PEFT merge (no Unsloth GGUF)")
    print(f"  Source:    {GRPO_OUTPUT}")
    print(f"  Output:    {merged_dir}")
    print(f"  NOTE: GGUF conversion requires manual llama.cpp convert_hf_to_gguf.py")
    print()

    _wandb_init_safe(
        project="FinSenti",
        name=f"export-{SHORT_NAME}",
        tags=["export", "peft-merge", MODEL_KEY],
        config={
            "phase": "export", "model_key": MODEL_KEY,
            "grpo_checkpoint": GRPO_OUTPUT,
            "method": "peft_merge",
        },
    )

    # Merge PEFT adapters and save HF weights.
    # IMPORTANT: load base, resize embeddings to match the adapter's vocab,
    # then attach adapter. See mobilellm_r1_950m.py:run_export for rationale.
    print(f"Loading and merging PEFT adapters from {GRPO_OUTPUT}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(GRPO_OUTPUT, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    _untie_lm_head(base)
    _setup_pad_token(tokenizer, base)
    model = PeftModel.from_pretrained(base, GRPO_OUTPUT)
    model = model.merge_and_unload()

    # Force tie_word_embeddings=False on the saved config. Tiny-LLM (based
    # on arnir0/Tiny-LLM) ties lm_head to embed_tokens. _untie_lm_head()
    # untied them at runtime so LoRA could train both independently, but
    # the saved config still says tied -> on reload, transformers re-ties
    # and discards the trained lm_head delta. Setting this here is what
    # makes the trained lm_head delta survive the round-trip.
    model.config.tie_word_embeddings = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.tie_word_embeddings = False

    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    elapsed = time.time() - start

    total_size = sum(
        os.path.getsize(str(f))
        for f in merged_dir.rglob("*")
        if f.is_file()
    )
    size_mb = round(total_size / (1024 * 1024), 1)

    print(f"  Merged weights saved ({size_mb} MB, {elapsed:.0f}s)")

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
        print("\n  [SKIP] mlx-lm not installed - MLX export skipped (run on Apple Silicon)")
    except Exception as e:
        print(f"\n  [WARN] MLX conversion failed: {e}")

    wandb.summary.update({
        "merged_size_mb": size_mb,
        "export_time_sec": round(elapsed, 1),
        "method": "peft_merge",
    })

    # Upload to HuggingFace
    if upload:
        from huggingface_hub import HfApi
        api = HfApi()

        # Upload full-precision weights
        api.create_repo(repo_id=HF_FULL, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(merged_dir),
            repo_id=HF_FULL,
            repo_type="model",
        )
        print(f"  Uploaded HF weights -> {HF_FULL}")

        # Create GGUF repo (awaiting manual conversion upload)
        api.create_repo(repo_id=HF_GGUF, repo_type="model", exist_ok=True)
        print(f"  Created repo {HF_GGUF} (upload GGUF after manual conversion)")
        # Upload MLX models
        for q_bits, repo in MLX_REPOS.items():
            mlx_dir = output_dir / f"mlx-{q_bits}bit"
            if mlx_dir.exists():
                api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
                api.upload_folder(folder_path=str(mlx_dir), repo_id=repo, repo_type="model")
                print(f"  Uploaded MLX-{q_bits}bit -> {repo}")

    wandb.finish()

    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE - {SHORT_NAME} (HF weights only)")
    print(f"{'='*70}")
    print(f"  Merged weights: {merged_dir} ({size_mb} MB)")
    print(f"  Full-precision repo: {HF_FULL}")
    print(f"\n  Convert to GGUF with llama.cpp:")
    for quant in QUANTIZATIONS:
        print(f"    python convert_hf_to_gguf.py {merged_dir} --outtype {quant.lower()}")
    print(f"\n  Then upload all quants to: {HF_GGUF}")


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
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    phases = ["sft", "grpo", "export"] if args.phase == "all" else [args.phase]

    print(f"\n{'#'*70}")
    print(f"# FinSenti Pipeline - {SHORT_NAME} (PEFT)")
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

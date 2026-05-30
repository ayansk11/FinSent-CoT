"""
FinSenti - MobileLLM-R1-950M: SFT -> GRPO -> Export -> Upload

Single self-contained script for the complete training pipeline.

Why this script does NOT use PEFT/LoRA (rewritten 2026-05-29):
  - First run trained facebook/MobileLLM-R1-950M (a Qwen-distilled reasoning
    base that natively uses <think> tags) with LoRA r=16 SFT + LoRA r=16
    GRPO. After GRPO the model emitted fluent reasoning content but
    NEVER closed </think> + emitted <answer> (0/848 closures across the
    finsenti benchmark, 0.000 accuracy).
  - Root cause: LoRA can't strongly override the model's deep <think>
    prior from distillation. The model learned "open <think>, ramble
    forever, hit token cap". Bumping max_new_tokens 512 -> 2048 didn't
    help -- the model fills 2048 tokens with reasoning and still never
    closes.
  - Verl#3226 + DAPO papers establish that GRPO + LoRA on small models
    drives entropy collapse and template misalignment. Full FT (lora_rank=0)
    is the documented fix.
  - 950M params x bf16 = 1.9 GB; with Adam fp32 states (~7.6 GB) +
    activations, full FT fits comfortably on a single A100 80 GB.

DAPO-style stabilisation on top of full FT (same as tiny-llm-10m):
  - epsilon_high > epsilon  (clip-higher, prevents entropy collapse)
  - beta = 0                 (no KL penalty; standard for reasoning RL)
  - num_generations = 4      (group of 4 rollouts for advantage variance)
  - lower learning rates     (1e-5 SFT, 5e-6 GRPO -- 10x lower than LoRA)

Prompt format:
  - Install a markdown-style chat template ("### System ... ### Input
    ... ### Response ...") and train SFT against it. This actively
    overrides the base model's native <think> chat template, which is
    the root cause of the format-collapse failure of the first run.

Dataset: Ayansk11/FinSenti-Dataset (local validated splits)

Usage:
    python mobilellm_r1_950m.py --phase all          # Full pipeline
    python mobilellm_r1_950m.py --phase sft          # SFT only
    python mobilellm_r1_950m.py --phase grpo         # GRPO only
    python mobilellm_r1_950m.py --phase export       # Export only
"""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

# Ensure rewards.py and callbacks.py (in same dir) are importable from any cwd
# Must use .resolve() for absolute path - trainer.train() changes cwd
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

MODEL_KEY = "mobilellm-r1-950m"
BASE_MODEL = "facebook/MobileLLM-R1-950M"
SHORT_NAME = "MobileLLM-R1-950M"
MODEL_FAMILY = "mobilellm"
MAX_SEQ_LENGTH = 2048

# SFT hyperparameters - FULL FINE-TUNING (no LoRA)
# 950M params x bf16 = 1.9 GB; Adam fp32 states (~7.6 GB) + grad bf16 (1.9 GB)
# + activations fit on A100 80 GB with batch=2 / grad_ckpt off.
# LR cut 20x from the failed LoRA run (2e-4 -> 1e-5) because we're updating
# ALL params now, not just low-rank deltas. Lower LR also helps overcome the
# base model's strong <think> chat-template prior without exploding.
SFT_BATCH_SIZE = 2
SFT_GRAD_ACCUM = 8                # effective batch 16
SFT_LR = 1e-5                     # was 2e-4 with LoRA
SFT_EPOCHS = 3                    # was 5; full FT converges faster

# GRPO hyperparameters - FULL FT + DAPO stabilisation
# DAPO ("Decoupled Clip and Dynamic sAmpling Policy Optimization", verl
# paper) fixes vanilla-GRPO entropy collapse via clip-higher and beta=0.
GRPO_BATCH_SIZE = 1
GRPO_GRAD_ACCUM = 8               # effective batch 8 (memory cap for 4 rollouts)
GRPO_LR = 5e-6                    # 10x lower than failed LoRA run (5e-5)
GRPO_NUM_GENERATIONS = 4
GRPO_MAX_STEPS = 500              # short run to catch collapse fast; bump if
                                  # reward curve looks stable past step 200
GRPO_MAX_COMPLETION_LENGTH = 512
GRPO_EPSILON = 0.20               # DAPO standard lower clip
GRPO_EPSILON_HIGH = 0.28          # DAPO clip-higher anti-collapse
GRPO_BETA = 0.0                   # no KL penalty

# HuggingFace repos
HF_FULL = "Ayansk11/FinSenti-MobileLLM-R1-950M"
HF_GGUF = "Ayansk11/FinSenti-MobileLLM-R1-950M-GGUF"
QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]
MLX_REPOS = {
    4: "Ayansk11/FinSenti-MobileLLM-R1-950M-MLX-4bit",
    8: "Ayansk11/FinSenti-MobileLLM-R1-950M-MLX-8bit",
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


# ─── Shared: full-FT model loading ───────────────────────────────────────────

def _setup_pad_token(tokenizer, model):
    """Add a dedicated pad token if pad collides with eos (causes label masking issues).

    Also ensures lm_head is resized to match new vocab size when it's been
    untied from embed_tokens. Otherwise LoRA on lm_head sees the old vocab
    size and crashes with a tensor-shape mismatch in the forward pass.
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
            # When lm_head is untied from embed_tokens, resize_token_embeddings
            # only resizes the input embeddings. Manually resize lm_head too
            # so the LoRA layer that gets added later sees the correct vocab.
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
                        # Initialize the new pad-token row with the mean of
                        # existing rows so it starts in-distribution.
                        if new_vocab > old_w.shape[0]:
                            new_lm.weight[old_w.shape[0]:] = old_w.mean(dim=0, keepdim=True)
                    inner.lm_head = new_lm
                    print(f"  [Fix] Manually resized lm_head {old_w.shape[0]} -> {new_vocab} "
                          f"(was untied from embed_tokens)")
                break
        print(f"  [Fix] Added dedicated pad_token (id={tokenizer.pad_token_id}, eos_id={tokenizer.eos_token_id})")
    else:
        print(f"  [Info] pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")


def _untie_lm_head(model):
    """Untie lm_head from embed_tokens so they can be trained independently.
    Required: weight tying with PEFT/quantization causes NaN gradients."""
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


# Markdown-style chat template that actively overrides MobileLLM-R1's
# native <think>-based chat template. By renaming the assistant role
# block to "### Response" we cut the base prior's affordance to emit
# <think>...</think> and replace it with the FinSenti <reasoning>/<answer>
# structure that the SFT data uses inside the response block.
PLAIN_TEXT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}### System\n{{ message['content'] }}\n\n"
    "{% elif message['role'] == 'user' %}### Input\n{{ message['content'] }}\n\n"
    "{% elif message['role'] == 'assistant' %}### Response\n{{ message['content'] }}"
    "{{ eos_token }}\n\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}### Response\n{% endif %}"
)


def _install_chat_template(tokenizer):
    """Install the markdown template, OVERWRITING any native template.

    Critical for MobileLLM-R1-950M specifically: the base ships with a
    Qwen-style <think>...</think> chat template that survived our first
    SFT pass (LoRA can't easily override behavioral templates). Stamping
    our own template on the tokenizer at SFT time means there is exactly
    one inference contract: ### Response -> <reasoning>...</reasoning>
    <answer>X</answer>. No <think> anywhere."""
    tokenizer.chat_template = PLAIN_TEXT_CHAT_TEMPLATE


def _load_base_model_fullft(base_model: str):
    """Load MobileLLM-R1-950M in bf16 with all parameters trainable.

    No PEFT / no quantisation / no LoRA. The first run used LoRA r=16 with
    modules_to_save on lm_head+embed_tokens and the model never closed
    </think> at eval (0/848 closures on finsenti). Full FT lets us
    actually reprogram the model's chat behaviour.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    _install_chat_template(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Untie BEFORE pad-token resize so embed_tokens and lm_head can move
    # independently during training.
    _untie_lm_head(model)
    _setup_pad_token(tokenizer, model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_train:,} / {n_total:,} (100% - full FT)")
    return model, tokenizer


def _load_sft_checkpoint_fullft(checkpoint_path: str):
    """Load the SFT-trained full-FT model for GRPO continuation.

    With full FT the SFT checkpoint IS a complete causal-LM (no adapter
    to merge). Just AutoModelForCausalLM.from_pretrained.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    _install_chat_template(tokenizer)  # defensive re-install

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_train:,} (100% - full FT continued from SFT)")
    return model, tokenizer

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
    print(f"  Backend:     FULL fine-tuning (no LoRA, no quantisation)")
    print(f"  Batch:       {SFT_BATCH_SIZE} x {SFT_GRAD_ACCUM} = {SFT_BATCH_SIZE * SFT_GRAD_ACCUM}")
    print(f"  LR: {SFT_LR}, Epochs: {SFT_EPOCHS}")
    print()

    _wandb_init_safe(
        project="FinSenti",
        name=f"sft-{SHORT_NAME}-fullft-ep{SFT_EPOCHS}",
        tags=["sft", "warm-up", MODEL_KEY, MODEL_FAMILY, "full-ft"],
        config={
            "phase": "sft", "model_key": MODEL_KEY, "base_model": BASE_MODEL,
            "epochs": SFT_EPOCHS, "batch_size": SFT_BATCH_SIZE, "lr": SFT_LR,
            "backend": "full-ft",
        },
    )

    model, tokenizer = _load_base_model_fullft(BASE_MODEL)
    print(f"  Chat template: installed (markdown ### System / ### Input / ### Response)")

    # Load dataset
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
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} SFT samples")
    # Debug: show first formatted sample to verify template
    print(f"  Sample 0 (first 400 chars):\n{formatted[0]['text'][:400]}")

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

    # Persist tie_word_embeddings=False so the trained lm_head survives reload
    model.config.tie_word_embeddings = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.tie_word_embeddings = False

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
    print(f"  Backend:     TRL GRPOTrainer, FULL FT (no LoRA), DAPO-stabilised")
    print(f"  Batch:       {GRPO_BATCH_SIZE} x {GRPO_GRAD_ACCUM} = {GRPO_BATCH_SIZE * GRPO_GRAD_ACCUM}")
    print(f"  LR: {GRPO_LR}, Gens: {GRPO_NUM_GENERATIONS}, Max steps: {GRPO_MAX_STEPS}")
    print(f"  DAPO:        epsilon={GRPO_EPSILON}, epsilon_high={GRPO_EPSILON_HIGH}, beta={GRPO_BETA}")
    print()

    _wandb_init_safe(
        project="FinSenti",
        name=f"grpo-{SHORT_NAME}-fullft-dapo-max{GRPO_MAX_STEPS}",
        tags=["grpo", "rl", "early-stopping", MODEL_KEY, MODEL_FAMILY, "full-ft", "dapo"],
        config={
            "phase": "grpo", "model_key": MODEL_KEY, "max_steps": GRPO_MAX_STEPS,
            "batch_size": GRPO_BATCH_SIZE, "lr": GRPO_LR,
            "num_generations": GRPO_NUM_GENERATIONS,
            "epsilon": GRPO_EPSILON, "epsilon_high": GRPO_EPSILON_HIGH,
            "beta": GRPO_BETA,
            "backend": "full-ft",
        },
    )

    # Load full-FT SFT checkpoint (no PEFT merge needed)
    print(f"Loading SFT checkpoint from {SFT_OUTPUT}...")
    model, tokenizer = _load_sft_checkpoint_fullft(SFT_OUTPUT)

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

    # Build GRPOConfig with DAPO stabilisation (probe TRL signature for
    # optional kwargs that may not exist on older installs).
    _grpoconfig_params = inspect.signature(GRPOConfig.__init__).parameters
    grpo_cfg_kwargs = dict(
        output_dir=GRPO_OUTPUT,
        max_steps=GRPO_MAX_STEPS,
        per_device_train_batch_size=GRPO_BATCH_SIZE,
        gradient_accumulation_steps=GRPO_GRAD_ACCUM,
        learning_rate=GRPO_LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_generations=GRPO_NUM_GENERATIONS,
        max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
        max_prompt_length=512,
        mask_truncated_completions=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=5,
        bf16=True,
        report_to="wandb",
        run_name=wandb.run.name if wandb.run else "grpo",
        seed=42,
    )
    if "epsilon" in _grpoconfig_params:
        grpo_cfg_kwargs["epsilon"] = GRPO_EPSILON
    if "epsilon_high" in _grpoconfig_params:
        grpo_cfg_kwargs["epsilon_high"] = GRPO_EPSILON_HIGH
        print(f"  [DAPO] clip-higher enabled: epsilon_high={GRPO_EPSILON_HIGH}")
    else:
        print(f"  [warn] epsilon_high not in this TRL build; "
              f"vanilla GRPO clipping (entropy-collapse risk)")
    if "beta" in _grpoconfig_params:
        grpo_cfg_kwargs["beta"] = GRPO_BETA
        print(f"  [DAPO] KL penalty disabled (beta=0)")

    trainer_kwargs = {
        "model": model,
        _config_key: GRPOConfig(**grpo_cfg_kwargs),
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

    # Wire up warnings_issued on the raw model (no PEFT chain anymore;
    # model IS the LlamaForCausalLM after full FT).
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # Force-disable kv cache: gradient checkpointing + cache = warning
    # loop that can stall the rollout loop per TRL #3683.
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Disable gradient checkpointing for GRPO. 950M with batch=1 on A100
    # 80 GB doesn't need the memory save, and grad_ckpt + GRPO is a
    # documented hang source in TRL's rollout loop.
    if hasattr(model, "gradient_checkpointing_disable"):
        print("  [debug] disabling gradient checkpointing for GRPO", flush=True)
        model.gradient_checkpointing_disable()

    # Warm generation probe: if model.generate() deadlocks with PEFT+bnb
    # we want to know in seconds, not hours.
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

    # Persist tie_word_embeddings=False so the trained lm_head survives reload
    model.config.tie_word_embeddings = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.tie_word_embeddings = False

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

    output_dir = Path(EXPORT_OUTPUT)
    merged_dir = output_dir / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FinSenti Export - {SHORT_NAME}")
    print("=" * 70)
    print(f"  Method:    Direct copy of full-FT checkpoint (no PEFT merge)")
    print(f"  Source:    {GRPO_OUTPUT}")
    print(f"  Output:    {merged_dir}")
    print(f"  NOTE: GGUF conversion requires manual llama.cpp convert_hf_to_gguf.py")
    print()

    _wandb_init_safe(
        project="FinSenti",
        name=f"export-{SHORT_NAME}",
        tags=["export", "full-ft", MODEL_KEY],
        config={
            "phase": "export", "model_key": MODEL_KEY,
            "grpo_checkpoint": GRPO_OUTPUT,
            "method": "full_ft_copy",
        },
    )

    # Full FT: GRPO_OUTPUT already contains the complete trained
    # causal-LM (no adapter to merge). Load it once to verify integrity,
    # then save to the merged_hf dir for the HF upload + MLX/GGUF
    # downstream steps.
    print(f"Loading full-FT checkpoint from {GRPO_OUTPUT}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(GRPO_OUTPUT, trust_remote_code=True)
    _install_chat_template(tokenizer)  # ensure benchmark.py sees the markdown template
    model = AutoModelForCausalLM.from_pretrained(
        GRPO_OUTPUT,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    # Belt + suspenders: assert the trained lm_head survived save/load.
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
        "method": "full_ft_copy",
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
    print(f"# FinSenti Pipeline - {SHORT_NAME} (full-FT + DAPO)")
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

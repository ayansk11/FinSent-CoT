"""
Model configurations for all 6 fine-tuning targets.

Central registry of per-model hyperparameters for SFT, GRPO, and export.
Each model has optimized settings for Hopper H100 GPUs.

Supported models:
  1. Qwen3-0.6B     — Smallest Qwen3, fastest inference
  2. Qwen3-1.7B     — Good balance of speed and quality
  3. Qwen3-4B       — Primary target, best quality/size ratio
  4. Qwen3-8B       — Highest quality, needs more VRAM
  5. DeepSeek-R1-Distill-Qwen-1.5B — R1 reasoning distilled into small model
  6. MobileLLM-R1-950M — Meta's mobile-optimized R1 model

Models 1-5 use Unsloth for QLoRA (2-3x faster training).
Model 6 uses standard PEFT + bitsandbytes (Unsloth doesn't support MobileLLM arch).
"""

MODEL_CONFIGS = {
    # ─── Qwen3 Family (Unsloth QLoRA) ───────────────────────────────────────

    "qwen3-0.6b": {
        "base_model": "unsloth/Qwen3-0.6B",
        "model_family": "qwen3",
        "use_unsloth": True,
        "short_name": "Qwen3-0.6B",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "max_seq_length": 2048,
        "sft": {
            "batch_size": 16,
            "grad_accum": 2,
            "lr": 2e-4,
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": 3,
        },
        "grpo": {
            "batch_size": 8,
            "grad_accum": 2,
            "lr": 5e-5,
            "lora_r": 16,
            "lora_alpha": 32,
            "num_generations": 8,
            "max_steps": 3000,
            "max_completion_length": 512,
        },
        "slurm": {
            "gpus": 1,
            "mem": "64G",
            "sft_time": "02:00:00",
            "grpo_time": "08:00:00",
        },
        "hf_repos": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3-0.6B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3-0.6B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3-0.6B-Q8_0",
        },
    },

    "qwen3-1.7b": {
        "base_model": "unsloth/Qwen3-1.7B",
        "model_family": "qwen3",
        "use_unsloth": True,
        "short_name": "Qwen3-1.7B",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "max_seq_length": 2048,
        "sft": {
            "batch_size": 8,
            "grad_accum": 4,
            "lr": 2e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "epochs": 3,
        },
        "grpo": {
            "batch_size": 4,
            "grad_accum": 2,
            "lr": 5e-5,
            "lora_r": 32,
            "lora_alpha": 64,
            "num_generations": 8,
            "max_steps": 3000,
            "max_completion_length": 512,
        },
        "slurm": {
            "gpus": 1,
            "mem": "64G",
            "sft_time": "03:00:00",
            "grpo_time": "10:00:00",
        },
        "hf_repos": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3-1.7B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3-1.7B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3-1.7B-Q8_0",
        },
    },

    "qwen3-4b": {
        "base_model": "unsloth/Qwen3-4B",
        "model_family": "qwen3",
        "use_unsloth": True,
        "short_name": "Qwen3-4B",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "max_seq_length": 2048,
        "sft": {
            "batch_size": 4,
            "grad_accum": 4,
            "lr": 2e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "epochs": 3,
        },
        "grpo": {
            "batch_size": 4,
            "grad_accum": 2,
            "lr": 5e-5,
            "lora_r": 32,
            "lora_alpha": 64,
            "num_generations": 6,
            "max_steps": 3000,
            "max_completion_length": 512,
        },
        "slurm": {
            "gpus": 1,
            "mem": "80G",
            "sft_time": "03:00:00",
            "grpo_time": "10:00:00",
        },
        "hf_repos": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3-4B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3-4B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3-4B-Q8_0",
        },
    },

    "qwen3-8b": {
        "base_model": "unsloth/Qwen3-8B",
        "model_family": "qwen3",
        "use_unsloth": True,
        "short_name": "Qwen3-8B",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "max_seq_length": 2048,
        "sft": {
            "batch_size": 4,
            "grad_accum": 4,
            "lr": 1e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "epochs": 3,
        },
        "grpo": {
            "batch_size": 2,
            "grad_accum": 4,
            "lr": 3e-5,
            "lora_r": 32,
            "lora_alpha": 64,
            "num_generations": 4,
            "max_steps": 3000,
            "max_completion_length": 512,
        },
        "slurm": {
            "gpus": 1,
            "mem": "80G",
            "sft_time": "04:00:00",
            "grpo_time": "14:00:00",
        },
        "hf_repos": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3-8B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3-8B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3-8B-Q8_0",
        },
    },

    # ─── DeepSeek R1 Distill (Unsloth QLoRA) ────────────────────────────────

    "deepseek-r1-1.5b": {
        "base_model": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
        "model_family": "deepseek",
        "use_unsloth": True,
        "short_name": "DeepSeek-R1-1.5B",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "max_seq_length": 2048,
        "sft": {
            "batch_size": 8,
            "grad_accum": 4,
            "lr": 2e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "epochs": 3,
        },
        "grpo": {
            "batch_size": 4,
            "grad_accum": 2,
            "lr": 5e-5,
            "lora_r": 32,
            "lora_alpha": 64,
            "num_generations": 8,
            "max_steps": 3000,
            "max_completion_length": 512,
        },
        "slurm": {
            "gpus": 1,
            "mem": "64G",
            "sft_time": "03:00:00",
            "grpo_time": "10:00:00",
        },
        "hf_repos": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-DeepSeek-R1-1.5B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-DeepSeek-R1-1.5B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-DeepSeek-R1-1.5B-Q8_0",
        },
    },

    # ─── MobileLLM R1 (Standard PEFT — no Unsloth) ─────────────────────────

    "mobilellm-r1-950m": {
        "base_model": "facebook/MobileLLM-R1-950M",
        "model_family": "mobilellm",
        "use_unsloth": False,
        "short_name": "MobileLLM-R1-950M",
        # MobileLLM uses LLaMA-style architecture with shared layers
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "max_seq_length": 2048,
        "sft": {
            "batch_size": 8,
            "grad_accum": 4,
            "lr": 2e-4,
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": 3,
        },
        "grpo": {
            "batch_size": 4,
            "grad_accum": 2,
            "lr": 5e-5,
            "lora_r": 16,
            "lora_alpha": 32,
            "num_generations": 6,
            "max_steps": 3000,
            "max_completion_length": 512,
        },
        "slurm": {
            "gpus": 1,
            "mem": "64G",
            "sft_time": "03:00:00",
            "grpo_time": "10:00:00",
        },
        "hf_repos": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-MobileLLM-R1-950M-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-MobileLLM-R1-950M-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-MobileLLM-R1-950M-Q8_0",
        },
    },
}

# All 3 quantization types exported per model
QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

# Ordered list for batch training (smallest to largest)
MODEL_ORDER = [
    "qwen3-0.6b",
    "mobilellm-r1-950m",
    "deepseek-r1-1.5b",
    "qwen3-1.7b",
    "qwen3-4b",
    "qwen3-8b",
]

ALL_MODEL_KEYS = list(MODEL_CONFIGS.keys())


def get_config(model_key: str) -> dict:
    """Get config by key. Raises KeyError with helpful message if not found."""
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(ALL_MODEL_KEYS)
        raise KeyError(
            f"Unknown model key '{model_key}'. Available: {available}"
        )
    return MODEL_CONFIGS[model_key]


def resolve_model_key(model_key_or_name: str) -> str:
    """
    Resolve a model key from either a key or a full HF model name.
    Examples:
        "qwen3-4b" -> "qwen3-4b"
        "unsloth/Qwen3-4B" -> "qwen3-4b"
        "facebook/MobileLLM-R1-950M" -> "mobilellm-r1-950m"
    """
    # Direct key match
    if model_key_or_name in MODEL_CONFIGS:
        return model_key_or_name

    # Try matching by base_model name
    for key, cfg in MODEL_CONFIGS.items():
        if cfg["base_model"].lower() == model_key_or_name.lower():
            return key

    # Try partial match on the model name part
    search = model_key_or_name.lower().replace("/", "").replace("-", "").replace("_", "")
    for key, cfg in MODEL_CONFIGS.items():
        base = cfg["base_model"].lower().replace("/", "").replace("-", "").replace("_", "")
        if search in base or base in search:
            return key

    available = ", ".join(ALL_MODEL_KEYS)
    raise KeyError(
        f"Cannot resolve '{model_key_or_name}' to a model key. Available: {available}"
    )


def print_model_table():
    """Print a summary table of all models."""
    print(f"{'Key':<22} {'Model':<45} {'Backend':<10} {'SFT BS':<8} {'GRPO BS':<8}")
    print("-" * 95)
    for key in MODEL_ORDER:
        cfg = MODEL_CONFIGS[key]
        backend = "Unsloth" if cfg["use_unsloth"] else "PEFT+bnb"
        sft_bs = cfg["sft"]["batch_size"] * cfg["sft"]["grad_accum"]
        grpo_bs = cfg["grpo"]["batch_size"] * cfg["grpo"]["grad_accum"]
        print(f"{key:<22} {cfg['base_model']:<45} {backend:<10} {sft_bs:<8} {grpo_bs:<8}")

"""
Generate HuggingFace model cards (README.md) for all FinSent-CoT model repos.

Creates 4 cards per model (1 full-precision + 3 GGUF) = 24 total.
Cards are auto-generated with model-specific details from model_configs.py
and uploaded alongside weights during export.

Usage:
    # Generate all cards locally (preview)
    python training/model_card_generator.py --output-dir ./model_cards

    # Generate for a specific model
    python training/model_card_generator.py --model-key qwen3-4b --output-dir ./model_cards

Called automatically by export_gguf.py during the upload phase.
"""

import argparse
from pathlib import Path

from model_configs import MODEL_CONFIGS, MODEL_ORDER, QUANTIZATIONS, get_config


DATASET_REPO = "Ayansk11/FinSent-CoT-Dataset"
COLLECTION_URL = "https://huggingface.co/collections/Ayansk11/finsent-cot"


def generate_full_precision_card(model_key: str, config: dict) -> str:
    """Generate README.md for a full-precision HF weights repo."""
    short_name = config["short_name"]
    base_model = config["base_model"]
    hf_full = config["hf_full"]
    hf_repos = config["hf_repos"]
    backend = "Unsloth QLoRA" if config["use_unsloth"] else "PEFT + bitsandbytes"
    sft = config["sft"]
    grpo = config["grpo"]

    gguf_links = "\n".join(
        f"| {q} | [{hf_repos[q]}](https://huggingface.co/{hf_repos[q]}) |{' **Recommended**' if q == 'Q5_K_M' else ''}"
        for q in QUANTIZATIONS
    )

    card = f"""---
language:
  - en
license: apache-2.0
library_name: transformers
base_model: {base_model}
tags:
  - finance
  - sentiment-analysis
  - chain-of-thought
  - cot
  - financial-nlp
  - finsent-cot
pipeline_tag: text-classification
---

# FinSent-CoT {short_name} — Full Precision

Financial sentiment analysis model with **chain-of-thought reasoning**, fine-tuned from [{base_model}](https://huggingface.co/{base_model}).

This repo contains the **full-precision HuggingFace weights** (merged 16-bit). For quantized GGUF versions optimized for Ollama/llama.cpp deployment, see the links below.

## Model Details

| Property | Value |
|----------|-------|
| **Base model** | [{base_model}](https://huggingface.co/{base_model}) |
| **Fine-tuning method** | {backend} |
| **Training stages** | SFT + GRPO (2-stage) |
| **Task** | Financial sentiment classification (3-class: positive / negative / neutral) |
| **Output format** | Chain-of-thought reasoning in `<reasoning>` tags + label in `<answer>` tags |
| **Training data** | [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO}) (16.9K samples) |
| **Compute** | Indiana University Big Red 200 (NVIDIA Hopper H100 GPUs) |

## Training Hyperparameters

### SFT (Supervised Fine-Tuning)
| Parameter | Value |
|-----------|-------|
| Epochs | {sft['epochs']} |
| Batch size | {sft['batch_size']} |
| Gradient accumulation | {sft['grad_accum']} |
| Effective batch size | {sft['batch_size'] * sft['grad_accum']} |
| Learning rate | {sft['lr']} |
| LoRA rank (r) | {sft['lora_r']} |
| LoRA alpha | {sft['lora_alpha']} |

### GRPO (Group Relative Policy Optimization)
| Parameter | Value |
|-----------|-------|
| Max steps | {grpo['max_steps']} |
| Batch size | {grpo['batch_size']} |
| Gradient accumulation | {grpo['grad_accum']} |
| Learning rate | {grpo['lr']} |
| Num generations | {grpo['num_generations']} |
| Max completion length | {grpo['max_completion_length']} |
| LoRA rank (r) | {grpo['lora_r']} |

**GRPO Reward Functions** (equal weight):
1. **Correctness** — predicted label matches ground truth
2. **Format** — proper `<reasoning>`/`<answer>` XML tag structure
3. **Reasoning Quality** — depth, financial terminology, analytical rigor
4. **Consistency** — reasoning logically supports the final answer

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{hf_full}")
tokenizer = AutoTokenizer.from_pretrained("{hf_full}")

text = "Apple reported record quarterly revenue of $123.9 billion, up 11% year over year."

messages = [
    {{"role": "system", "content": "You are a financial sentiment analyst. Analyze the given financial text and provide your reasoning in <reasoning> tags and your sentiment classification (positive, negative, or neutral) in <answer> tags."}},
    {{"role": "user", "content": text}}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=512, temperature=0.3)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## GGUF Quantized Versions

For deployment via Ollama or llama.cpp, use these quantized versions:

| Quantization | Repository | Note |
|-------------|------------|------|
{gguf_links}

## Training Pipeline

```
Qwen3-235B-A22B (teacher) → CoT Dataset (16.9K) → SFT warm-up → GRPO refinement → This model
```

## All Models in Collection

See the full [FinSent-CoT Collection]({COLLECTION_URL}) for all 6 models and quantizations.

## Dataset

Trained on [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO}) — 16,944 balanced samples (5,648 per class) with chain-of-thought reasoning generated by Qwen3-235B-A22B-FP8.

## Citation

```bibtex
@model{{finsent_cot_{model_key.replace('-', '_')}_2026,
  title     = {{FinSent-CoT {short_name}}},
  author    = {{Shaikh, Ayan}},
  year      = {{2026}},
  publisher = {{HuggingFace}},
  url       = {{https://huggingface.co/{hf_full}}}
}}
```

## License

Apache 2.0
"""
    return card.strip()


def generate_gguf_card(model_key: str, config: dict, quantization: str) -> str:
    """Generate README.md for a GGUF quantized repo."""
    short_name = config["short_name"]
    base_model = config["base_model"]
    hf_full = config["hf_full"]
    hf_repos = config["hf_repos"]
    repo_id = hf_repos[quantization]
    backend = "Unsloth QLoRA" if config["use_unsloth"] else "PEFT + bitsandbytes"
    model_family = config["model_family"]

    gguf_filename = f"FinSent-CoT-{short_name}.{quantization}.gguf"
    ollama_name = f"finsent-{model_key}"

    # Quantization descriptions
    quant_info = {
        "Q4_K_M": ("4-bit", "Smallest file size, fastest inference, slightly lower quality"),
        "Q5_K_M": ("5-bit", "Best balance of quality and size — recommended for most users"),
        "Q8_0": ("8-bit", "Highest quality, largest file size, closest to full precision"),
    }
    quant_bits, quant_desc = quant_info[quantization]

    is_recommended = quantization == "Q5_K_M"
    rec_badge = " ⭐ Recommended" if is_recommended else ""

    other_quants = "\n".join(
        f"| {q} | [{hf_repos[q]}](https://huggingface.co/{hf_repos[q]}) |{' **Recommended**' if q == 'Q5_K_M' else ''}"
        for q in QUANTIZATIONS
        if q != quantization
    )

    card = f"""---
language:
  - en
license: apache-2.0
base_model: {base_model}
tags:
  - finance
  - sentiment-analysis
  - chain-of-thought
  - cot
  - gguf
  - ollama
  - llama-cpp
  - finsent-cot
  - {quantization.lower().replace('_', '-')}
pipeline_tag: text-classification
---

# FinSent-CoT {short_name} — {quantization} GGUF{rec_badge}

Financial sentiment analysis model with **chain-of-thought reasoning**, quantized to **{quantization}** ({quant_bits}) GGUF format for deployment via [Ollama](https://ollama.ai/) or [llama.cpp](https://github.com/ggerganov/llama.cpp).

{quant_desc}.

## Quick Start with Ollama

```bash
# Download and create the model
ollama create {ollama_name} -f Modelfile

# Run inference
ollama run {ollama_name} "Tesla reported record deliveries of 1.8 million vehicles in 2023"
```

**Expected output:**
```
<reasoning>
The text reports Tesla achieving record deliveries of 1.8 million vehicles,
indicating strong demand and operational execution. Record figures signal
positive business momentum and growth trajectory...
</reasoning>
<answer>positive</answer>
```

## Files

| File | Description |
|------|-------------|
| `{gguf_filename}` | Quantized model weights ({quantization}, {quant_bits}) |
| `Modelfile` | Ollama configuration with system prompt and parameters |

## Model Details

| Property | Value |
|----------|-------|
| **Base model** | [{base_model}](https://huggingface.co/{base_model}) |
| **Quantization** | {quantization} ({quant_bits}) |
| **Fine-tuning** | {backend} |
| **Training stages** | SFT + GRPO (2-stage) |
| **Task** | Financial sentiment (3-class: positive / negative / neutral) |
| **Output** | `<reasoning>` CoT analysis + `<answer>` label |
| **Training data** | [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO}) (16.9K samples) |

## Other Versions

| Format | Repository | Note |
|--------|------------|------|
| Full precision | [{hf_full}](https://huggingface.co/{hf_full}) | HF weights (16-bit) |
{other_quants}

## Training Pipeline

```
Qwen3-235B-A22B (teacher) → CoT Dataset (16.9K) → SFT → GRPO (4 rewards) → GGUF {quantization}
```

**GRPO Reward Functions**: Correctness, Format, Reasoning Quality, Consistency (equal weight).

## All Models

See the full [FinSent-CoT Collection]({COLLECTION_URL}) for all 6 models and quantizations.

## Citation

```bibtex
@model{{finsent_cot_{model_key.replace('-', '_')}_{quantization.lower()}_2026,
  title     = {{FinSent-CoT {short_name} {quantization}}},
  author    = {{Shaikh, Ayan}},
  year      = {{2026}},
  publisher = {{HuggingFace}},
  url       = {{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache 2.0
"""
    return card.strip()


def generate_all_cards(model_key: str, config: dict, output_dir: Path) -> dict:
    """
    Generate all 4 model cards for a single model.
    Returns dict mapping repo_id -> card_path.
    """
    cards = {}

    # Full-precision card
    full_card = generate_full_precision_card(model_key, config)
    full_dir = output_dir / model_key / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / "README.md"
    full_path.write_text(full_card)
    cards[config["hf_full"]] = str(full_path)

    # GGUF cards
    for quant in QUANTIZATIONS:
        gguf_card = generate_gguf_card(model_key, config, quant)
        quant_dir = output_dir / model_key / quant
        quant_dir.mkdir(parents=True, exist_ok=True)
        quant_path = quant_dir / "README.md"
        quant_path.write_text(gguf_card)
        cards[config["hf_repos"][quant]] = str(quant_path)

    return cards


def main():
    parser = argparse.ArgumentParser(description="Generate model cards for FinSent-CoT repos")
    parser.add_argument("--model-key", default=None,
                        help="Generate for specific model (default: all)")
    parser.add_argument("--output-dir", default="./model_cards",
                        help="Output directory for generated cards")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.model_key:
        keys = [args.model_key]
    else:
        keys = MODEL_ORDER

    total_cards = 0
    for key in keys:
        config = get_config(key)
        cards = generate_all_cards(key, config, output_dir)
        total_cards += len(cards)
        print(f"\n{config['short_name']}:")
        for repo_id, path in cards.items():
            print(f"  {repo_id} -> {path}")

    print(f"\nGenerated {total_cards} model cards in {output_dir}/")


if __name__ == "__main__":
    main()

"""
Generate and upload detailed model cards for every FinSenti repo on the Hub.

What this does
--------------
Walks the MODELS list, builds a README.md for each base (SafeTensors) and
GGUF repo, and uploads it via the HuggingFace API. Skips repos that don't
exist yet so you can run the script repeatedly as new models land.

Usage
-----
    # Upload cards for everything that already exists on the Hub
    python training/generate_model_cards.py

    # Preview a single card without uploading
    python training/generate_model_cards.py --model qwen3-4b --dry-run

    # Save all cards to ./generated_cards/ instead of uploading
    python training/generate_model_cards.py --dry-run --out generated_cards

    # Upload only one model's cards (base + gguf)
    python training/generate_model_cards.py --model qwen3.5-2b
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


# -----------------------------------------------------------------------------
# Per-model facts. Numbers come from the W&B runs and SLURM logs we kept.
# Sizes are bf16 (full) or post-quant (GGUF). Reward is the best mean GRPO
# reward observed during the run, out of a possible 4.0 (four equal-weight
# reward functions in training/rewards.py).
# -----------------------------------------------------------------------------
MODELS = [
    {
        "key": "qwen3-0.6b",
        "name": "Qwen3-0.6B",
        "family": "Qwen3",
        "params_b": 0.6,
        "base_model": "Qwen/Qwen3-0.6B",
        "full_repo": "Ayansk11/FinSenti-Qwen3-0.6B",
        "gguf_repo": "Ayansk11/FinSenti-Qwen3-0.6B-GGUF",
        "size_bf16_gb": 1.2,
        "gguf_sizes": {"Q4_K_M": 0.37, "Q5_K_M": 0.41, "Q8_0": 0.60},
        "sft_hours": 0.32,
        "grpo_steps": 320,
        "best_reward": 3.59,
        "vram_bf16_gb": 2,
        "ram_gguf_gb": 1,
        "trainer": "Unsloth + TRL",
        "blurb": (
            "the smallest model in the Qwen3 batch and the easiest one "
            "to drop on almost anything, including a Raspberry Pi 5 or a "
            "five-year-old laptop CPU"
        ),
    },
    {
        "key": "qwen3-1.7b",
        "name": "Qwen3-1.7B",
        "family": "Qwen3",
        "params_b": 1.7,
        "base_model": "Qwen/Qwen3-1.7B",
        "full_repo": "Ayansk11/FinSenti-Qwen3-1.7B",
        "gguf_repo": "Ayansk11/FinSenti-Qwen3-1.7B-GGUF",
        "size_bf16_gb": 3.4,
        "gguf_sizes": {"Q4_K_M": 1.10, "Q5_K_M": 1.26, "Q8_0": 1.83},
        "sft_hours": 0.83,
        "grpo_steps": 300,
        "best_reward": 3.71,
        "vram_bf16_gb": 4,
        "ram_gguf_gb": 2,
        "trainer": "Unsloth + TRL",
        "blurb": (
            "a useful middle size: small enough to load on a 6 GB laptop "
            "GPU, big enough that the reasoning stays coherent on tricky "
            "headlines"
        ),
    },
    {
        "key": "qwen3-4b",
        "name": "Qwen3-4B",
        "family": "Qwen3",
        "params_b": 4.0,
        "base_model": "Qwen/Qwen3-4B",
        "full_repo": "Ayansk11/FinSenti-Qwen3-4B",
        "gguf_repo": "Ayansk11/FinSenti-Qwen3-4B-GGUF",
        "size_bf16_gb": 8.0,
        "gguf_sizes": {"Q4_K_M": 2.40, "Q5_K_M": 2.78, "Q8_0": 4.10},
        "sft_hours": 1.5,
        "grpo_steps": 300,
        "best_reward": 3.50,
        "vram_bf16_gb": 10,
        "ram_gguf_gb": 4,
        "trainer": "Unsloth + TRL",
        "blurb": (
            "the workhorse of the Qwen3 group: 4 billion params, fits in "
            "10 GB of VRAM at bf16, and the explanations stay sharper than "
            "the smaller siblings on edge cases"
        ),
    },
    {
        "key": "qwen3-8b",
        "name": "Qwen3-8B",
        "family": "Qwen3",
        "params_b": 8.0,
        "base_model": "Qwen/Qwen3-8B",
        "full_repo": "Ayansk11/FinSenti-Qwen3-8B",
        "gguf_repo": "Ayansk11/FinSenti-Qwen3-8B-GGUF",
        "size_bf16_gb": 16.0,
        "gguf_sizes": {"Q4_K_M": 4.70, "Q5_K_M": 5.40, "Q8_0": 8.20},
        "sft_hours": 2.0,
        "grpo_steps": 390,
        "best_reward": 3.50,
        "vram_bf16_gb": 18,
        "ram_gguf_gb": 6,
        "trainer": "Unsloth + TRL",
        "blurb": (
            "the biggest Qwen3 variant in this set. If you have a single "
            "A100 or a 24 GB consumer card, you'll get the cleanest "
            "explanations of the Qwen3 family from this one"
        ),
    },
    {
        "key": "qwen3.5-0.8b",
        "name": "Qwen3.5-0.8B",
        "family": "Qwen3.5",
        "params_b": 0.8,
        "base_model": "Qwen/Qwen3.5-0.8B",
        "full_repo": "Ayansk11/FinSenti-Qwen3.5-0.8B",
        "gguf_repo": "Ayansk11/FinSenti-Qwen3.5-0.8B-GGUF",
        "size_bf16_gb": 1.6,
        "gguf_sizes": {"Q4_K_M": 0.50, "Q5_K_M": 0.57, "Q8_0": 0.83},
        "sft_hours": 1.36,
        "grpo_steps": 320,
        "best_reward": 3.55,
        "vram_bf16_gb": 3,
        "ram_gguf_gb": 1,
        "trainer": "Unsloth + TRL",
        "blurb": (
            "an updated-pretraining sibling of Qwen3-0.6B with a slightly "
            "different output feel. Same lightweight footprint, slightly "
            "newer training mix"
        ),
    },
    {
        "key": "qwen3.5-2b",
        "name": "Qwen3.5-2B",
        "family": "Qwen3.5",
        "params_b": 2.0,
        "base_model": "Qwen/Qwen3.5-2B",
        "full_repo": "Ayansk11/FinSenti-Qwen3.5-2B",
        "gguf_repo": "Ayansk11/FinSenti-Qwen3.5-2B-GGUF",
        "size_bf16_gb": 4.0,
        "gguf_sizes": {"Q4_K_M": 1.20, "Q5_K_M": 1.40, "Q8_0": 2.10},
        "sft_hours": 3.0,
        "grpo_steps": 560,
        "best_reward": 3.55,
        "vram_bf16_gb": 5,
        "ram_gguf_gb": 2,
        "trainer": "Unsloth + TRL",
        "blurb": (
            "the mid-size Qwen3.5 variant. The longer GRPO run (560 steps) "
            "gave it the best format compliance of any model under 4B in "
            "this study"
        ),
    },
    {
        "key": "qwen3.5-4b",
        "name": "Qwen3.5-4B",
        "family": "Qwen3.5",
        "params_b": 4.0,
        "base_model": "Qwen/Qwen3.5-4B",
        "full_repo": "Ayansk11/FinSenti-Qwen3.5-4B",
        "gguf_repo": "Ayansk11/FinSenti-Qwen3.5-4B-GGUF",
        "size_bf16_gb": 8.0,
        "gguf_sizes": {"Q4_K_M": 2.40, "Q5_K_M": 2.78, "Q8_0": 4.10},
        "sft_hours": 5.0,
        "grpo_steps": 480,
        "best_reward": 3.50,
        "vram_bf16_gb": 10,
        "ram_gguf_gb": 4,
        "trainer": "Unsloth + TRL",
        "blurb": (
            "the 4B model on the newer Qwen3.5 backbone. Output style is "
            "close to Qwen3-4B with a slightly different feel from the "
            "updated pretraining data"
        ),
    },
    {
        "key": "qwen3.5-9b",
        "name": "Qwen3.5-9B",
        "family": "Qwen3.5",
        "params_b": 9.0,
        "base_model": "Qwen/Qwen3.5-9B",
        "full_repo": "Ayansk11/FinSenti-Qwen3.5-9B",
        "gguf_repo": "Ayansk11/FinSenti-Qwen3.5-9B-GGUF",
        "size_bf16_gb": 18.0,
        "gguf_sizes": {"Q4_K_M": 5.50, "Q5_K_M": 6.30, "Q8_0": 9.50},
        "sft_hours": 10.0,
        "grpo_steps": 420,
        "best_reward": 3.50,
        "vram_bf16_gb": 20,
        "ram_gguf_gb": 7,
        "trainer": "Unsloth + TRL",
        "blurb": (
            "the largest model in the FinSenti family. Reasoning chains "
            "are the most thorough of the bunch, but you'll need real GPU "
            "memory (~20 GB bf16) to run it without quantization"
        ),
    },
    {
        "key": "deepseek-r1-1.5b",
        "name": "DeepSeek-R1-1.5B",
        "family": "DeepSeek",
        "params_b": 1.5,
        "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "full_repo": "Ayansk11/FinSenti-DeepSeek-R1-1.5B",
        "gguf_repo": "Ayansk11/FinSenti-DeepSeek-R1-1.5B-GGUF",
        "size_bf16_gb": 3.0,
        "gguf_sizes": {"Q4_K_M": 1.00, "Q5_K_M": 1.16, "Q8_0": 1.70},
        "sft_hours": 0.6,
        "grpo_steps": 360,
        "best_reward": 3.13,
        "vram_bf16_gb": 4,
        "ram_gguf_gb": 2,
        "trainer": "Unsloth + TRL",
        "blurb": (
            "built on DeepSeek's R1 distillation, so it already had a "
            "decent reasoning prior coming in. The SFT + GRPO passes "
            "narrow that reasoning to financial sentiment specifically"
        ),
    },
]


# -----------------------------------------------------------------------------
# Card templates. Two flavors: full-precision SafeTensors and GGUF.
# Style notes: hyphens only (no em-dashes), contractions where they fit,
# specific numbers, no marketing fluff.
# -----------------------------------------------------------------------------

YAML_BASE = """---
license: apache-2.0
language:
  - en
base_model: {base_model}
datasets:
  - Ayansk11/FinSenti-Dataset
pipeline_tag: text-generation
library_name: transformers
tags:
  - finance
  - financial-sentiment
  - sentiment-analysis
  - chain-of-thought
  - reasoning
  - grpo
  - sft
  - lora
  - finsenti
---
"""

YAML_GGUF = """---
license: apache-2.0
language:
  - en
base_model: {full_repo}
datasets:
  - Ayansk11/FinSenti-Dataset
pipeline_tag: text-generation
library_name: gguf
tags:
  - finance
  - financial-sentiment
  - chain-of-thought
  - reasoning
  - gguf
  - llama-cpp
  - ollama
  - quantized
  - finsenti
---
"""


SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. For each headline you receive, "
    "write a short reasoning chain inside <reasoning>...</reasoning> tags, "
    "then give a single label inside <answer>...</answer> tags. The label "
    "must be exactly one of: positive, negative, neutral."
)


EXAMPLE_HEADLINE = "Apple beats Q4 estimates as iPhone sales jump 12% year over year."

EXAMPLE_RESPONSE = (
    "<reasoning>\n"
    "Beating estimates is a positive earnings surprise. A 12% YoY iPhone "
    "sales jump in the company's biggest product line points to demand "
    "strength. Both signals push the read positive.\n"
    "</reasoning>\n"
    "<answer>positive</answer>"
)


def _related_models_section(current_key: str) -> str:
    """Generate a short list of other FinSenti models, grouped by family."""
    by_family: dict[str, list[dict]] = {}
    for m in MODELS:
        if m["key"] == current_key:
            continue
        by_family.setdefault(m["family"], []).append(m)

    lines = []
    for fam, items in by_family.items():
        items.sort(key=lambda x: x["params_b"])
        bullets = ", ".join(
            f"[{m['name']}](https://huggingface.co/{m['full_repo']})" for m in items
        )
        lines.append(f"- **{fam}**: {bullets}")
    return "\n".join(lines)


def _hardware_recommendation(m: dict) -> str:
    """Tailor the hardware section to the model size."""
    p = m["params_b"]
    if p < 1.0:
        return (
            f"At bf16 the weights are about {m['size_bf16_gb']:.1f} GB on disk "
            f"and need ~{m['vram_bf16_gb']} GB of GPU memory for batch=1 "
            f"inference. CPU inference is fine too: on a modern laptop you'll "
            f"get a few tokens per second with the bf16 weights, and 15-30 "
            f"tok/s with the GGUF Q4_K_M build."
        )
    if p < 3.0:
        return (
            f"bf16 weights are about {m['size_bf16_gb']:.1f} GB. You want "
            f"~{m['vram_bf16_gb']} GB of VRAM for batch=1 inference. CPU "
            f"works but is slower; the Q4_K_M GGUF is the right pick if you "
            f"don't have a GPU."
        )
    if p < 6.0:
        return (
            f"bf16 weights are about {m['size_bf16_gb']:.0f} GB and need "
            f"~{m['vram_bf16_gb']} GB of VRAM for inference (a 12 GB card "
            f"will do it with headroom). For CPU-only or 8 GB GPUs, grab the "
            f"Q4_K_M GGUF."
        )
    return (
        f"bf16 weights are about {m['size_bf16_gb']:.0f} GB. You'll want a "
        f"24 GB consumer card or a single A100/H100 to run it without "
        f"quantization. The Q4_K_M GGUF (~{m['gguf_sizes']['Q4_K_M']:.1f} GB) "
        f"runs on a 12 GB GPU or pure CPU on most laptops with 16+ GB RAM."
    )


def _gguf_filename(m: dict, quant: str) -> str:
    """Match the filename pattern emitted by the per-model training scripts."""
    return f"FinSenti-{m['name']}.{quant}.gguf"


def _gguf_quant_table(m: dict) -> str:
    """Markdown table of GGUF files in the repo."""
    rows = [
        "| File | Quant | Size | Notes |",
        "|------|-------|------|-------|",
    ]
    notes = {
        "Q4_K_M": "Smallest, mild quality dip. Default pick for laptops.",
        "Q5_K_M": "Balanced quality and size.",
        "Q8_0": "Closest to bf16, biggest file.",
    }
    for q, gb in m["gguf_sizes"].items():
        fname = _gguf_filename(m, q)
        rows.append(f"| `{fname}` | {q} | {gb:.2f} GB | {notes[q]} |")
    return "\n".join(rows)


def make_base_card(m: dict) -> str:
    """Generate the README for the full-precision SafeTensors repo."""
    yaml = YAML_BASE.format(base_model=m["base_model"])
    related = _related_models_section(m["key"])
    hardware = _hardware_recommendation(m)
    other_in_family = ", ".join(
        x["name"] for x in MODELS
        if x["family"] == m["family"] and x["key"] != m["key"]
    ) or "(it's the only one in this family so far)"

    body = f"""# FinSenti-{m['name']}

FinSenti-{m['name']} is a {m['params_b']:.1f}B-parameter model fine-tuned to
read short financial text (headlines, earnings snippets, market commentary)
and explain its read of them before settling on positive, negative, or
neutral. It's {m['blurb']}.

The model is part of the [FinSenti
collection](https://huggingface.co/collections/Ayansk11/finsenti), a
scaling study of small models trained on the same data with the same recipe.

## What it's good at

- Classifying short financial text (1-3 sentences) into positive / negative
  / neutral
- Producing a short reasoning chain you can read or log
- Following a strict `<reasoning>...</reasoning><answer>...</answer>` output
  format that's easy to parse downstream

It was trained on news-style headlines and earnings snippets in English, so
that's where it shines. Outside that domain you'll see the format hold up
but the labels get noisier.

## How it was trained

Two-stage recipe, same across the whole FinSenti family:

1. **SFT** on the [FinSenti
   dataset](https://huggingface.co/datasets/Ayansk11/FinSenti-Dataset)
   (50.8K samples, balanced across the three labels, with chain-of-thought
   targets generated by a teacher model and filtered for label agreement).
   This stage took about {m['sft_hours']:.1f} hours on a single A100 80GB
   for this model.
2. **GRPO** with four reward functions (sentiment correctness, format
   compliance, reasoning quality, output consistency), each weighted equally
   for a maximum reward of 4.0. Best mean reward observed during the run
   was **{m['best_reward']:.2f} / 4.0** at around step {m['grpo_steps']}.

Trainer stack: {m['trainer']}, with LoRA adapters on the attention and MLP
projection layers. The adapters were merged into the base weights before
the final SafeTensors export, so this repo is a self-contained model and
doesn't need PEFT to load.

## Quick start

Standard `transformers` usage:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{m['full_repo']}"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

system = (
    "You are a financial sentiment analyst. For each headline you receive, "
    "write a short reasoning chain inside <reasoning>...</reasoning> tags, "
    "then give a single label inside <answer>...</answer> tags. The label "
    "must be exactly one of: positive, negative, neutral."
)
user = "{EXAMPLE_HEADLINE}"

messages = [
    {{"role": "system", "content": system}},
    {{"role": "user", "content": user}},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Expected output (your reasoning text will vary; the label should match):

```
{EXAMPLE_RESPONSE}
```

## Prompt format

The model expects the system prompt above, verbatim is best. The user turn
is the headline or short snippet you want classified. Output is two XML-ish
blocks in this order: `<reasoning>...</reasoning>` then
`<answer>...</answer>`. The `<answer>` content is one of `positive`,
`negative`, or `neutral` (lowercase, no punctuation).

If you want labels only and don't care about the reasoning, you can stop
generation as soon as you see `</answer>` to save tokens.

## Performance notes

The training reward (max 4.0) hit **{m['best_reward']:.2f}** on the
held-out validation slice. That breaks down across the four reward
functions roughly as:

- Sentiment correctness: dominant contributor; the model gets the label
  right on the validation split most of the time
- Format compliance: near-saturated by the end of GRPO; the model almost
  always produces well-formed `<reasoning>` and `<answer>` tags
- Reasoning quality: judged on length and presence of finance-relevant
  signal words; this one's the noisiest of the four
- Consistency: rewards stable labels across paraphrases of the same headline

Numbers on standard finance benchmarks (FPB, FiQA, Twitter Financial News)
are forthcoming and will be added once the eval pipeline lands.

## Hardware

{hardware}

## Limitations

A few things this model isn't built for:

- **Long documents.** Training context was capped at 1024 tokens. Anything
  much longer than a paragraph or two is out of distribution.
- **Multi-asset reasoning.** It classifies the sentiment of a single piece
  of text. It won't aggregate across multiple headlines or weigh sources.
- **Numerical reasoning.** It can read "beats by 12%" and call that
  positive, but it isn't doing math. Don't ask it to forecast.
- **Languages other than English.** Training data was English only.
- **Background knowledge.** If the headline needs you to know what a
  company does, the model only has whatever was in its base pretraining.
  It can't look anything up.
- **Three labels, hard cutoffs.** The output space is positive / negative /
  neutral. If you need a 5-class scale or a continuous score, you'll need
  to retrain or post-process.

## Training details

| | |
|---|---|
| Base model | [{m['base_model']}](https://huggingface.co/{m['base_model']}) |
| Dataset | [Ayansk11/FinSenti-Dataset](https://huggingface.co/datasets/Ayansk11/FinSenti-Dataset) (50.8K samples) |
| SFT length | ~{m['sft_hours']:.1f} hours on A100 80GB |
| GRPO steps | {m['grpo_steps']} |
| Best GRPO reward | {m['best_reward']:.2f} / 4.0 |
| Adapter | LoRA (r=16, alpha=32) on q/k/v/o/gate/up/down projections |
| Sequence length | 1024 |
| Optimizer | AdamW (8-bit), cosine LR schedule |
| Hardware | NVIDIA A100 80GB (Indiana University BigRed200 cluster) |
| Frameworks | {m['trainer']} |

## Related FinSenti models

Other sizes and bases trained with the same recipe:

{related}

There's a GGUF build of this same model at
[{m['gguf_repo']}](https://huggingface.co/{m['gguf_repo']}) for Ollama and
llama.cpp, and the dataset itself is at
[Ayansk11/FinSenti-Dataset](https://huggingface.co/datasets/Ayansk11/FinSenti-Dataset).

If you're picking a size, a rough guide:

- **Need it on a phone or browser?** Look at the smallest model in the
  group ({MODELS[0]['name']}) or its GGUF.
- **Laptop with no GPU?** Any model up to ~2B as Q4_K_M GGUF works.
- **Single 8-12 GB GPU?** The 1.5B-4B sizes are the sweet spot.
- **Server or workstation?** The 8B / 9B variants give the best reasoning
  but need the memory.

## Citation

If you use this model in research, please cite:

```bibtex
@misc{{shaikh2026finsenti,
  title  = {{FinSenti: Small Language Models for Financial Sentiment with Chain-of-Thought Reasoning}},
  author = {{Shaikh, Ayan}},
  year   = {{2026}},
  url    = {{https://huggingface.co/collections/Ayansk11/finsenti}},
  note   = {{Indiana University}}
}}
```

## License

Apache 2.0, same as the base model.

## Acknowledgements

Trained on the Indiana University BigRed200 cluster (account `r01510`).
Thanks to the Unsloth and TRL teams for the trainer stack, and to the
Qwen / DeepSeek teams for the base models.
"""
    return yaml + body


def make_gguf_card(m: dict) -> str:
    """Generate the README for the GGUF repo."""
    yaml = YAML_GGUF.format(full_repo=m["full_repo"])
    related = _related_models_section(m["key"])
    table = _gguf_quant_table(m)
    name_safe = m["name"].lower().replace(".", "-")

    smallest_q = "Q4_K_M"
    smallest_gb = m["gguf_sizes"][smallest_q]
    f_q4 = _gguf_filename(m, "Q4_K_M")

    body = f"""# FinSenti-{m['name']} - GGUF

GGUF builds of [FinSenti-{m['name']}](https://huggingface.co/{m['full_repo']})
for use with [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggerganov/llama.cpp),
LM Studio, KoboldCpp, and other GGUF-compatible runtimes.

This is the same model as the SafeTensors repo, just converted and
quantized so you can run it on a CPU or a small GPU without pulling in
PyTorch.

## Files in this repo

{table}

If you're not sure which to pick: **start with Q4_K_M**. It's the smallest
file, it runs everywhere, and the quality drop versus the original bf16
weights is small for a model this size.

## Quick start (llama.cpp)

```bash
# Download the Q4_K_M file (or pick a different quant from the table above)
huggingface-cli download {m['gguf_repo']} {f_q4} --local-dir .

# Run it
./llama-cli -m {f_q4} \\
  --system "You are a financial sentiment analyst. For each headline you receive, write a short reasoning chain inside <reasoning>...</reasoning> tags, then give a single label inside <answer>...</answer> tags. The label must be exactly one of: positive, negative, neutral." \\
  -p "Apple beats Q4 estimates as iPhone sales jump 12% year over year." \\
  -n 256
```

## Quick start (Ollama)

This repo ships a `Modelfile` for each quant. To register the Q4_K_M build
under the name `finsenti-{name_safe}`:

```bash
huggingface-cli download {m['gguf_repo']} \\
  {f_q4} Modelfile.Q4_K_M --local-dir ./finsenti-tmp
cd finsenti-tmp
ollama create finsenti-{name_safe} -f Modelfile.Q4_K_M

# Then chat with it
ollama run finsenti-{name_safe} "Apple beats Q4 estimates as iPhone sales jump 12% year over year."
```

You should see output like:

```
{EXAMPLE_RESPONSE}
```

## Quick start (Python via llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./{f_q4}",
    n_ctx=2048,
    n_threads=8,
)

system = (
    "You are a financial sentiment analyst. For each headline you receive, "
    "write a short reasoning chain inside <reasoning>...</reasoning> tags, "
    "then give a single label inside <answer>...</answer> tags. The label "
    "must be exactly one of: positive, negative, neutral."
)

resp = llm.create_chat_completion(
    messages=[
        {{"role": "system", "content": system}},
        {{"role": "user", "content": "Apple beats Q4 estimates as iPhone sales jump 12% year over year."}},
    ],
    max_tokens=256,
    temperature=0.0,
)
print(resp["choices"][0]["message"]["content"])
```

## Hardware

The {smallest_q} build is about {smallest_gb:.2f} GB on disk and needs
roughly {m['ram_gguf_gb']} GB of free RAM at runtime. On a modern laptop
CPU you should see 15-40 tokens per second depending on the size of the
model and your core count. Throwing it on a small GPU (Apple Silicon, a
6-8 GB NVIDIA card) gets you considerably faster generation.

If you need more headroom, the Q5_K_M and Q8_0 files are progressively
closer to the original bf16 quality at the cost of size.

## Picking a quant

- **Q4_K_M** ({m['gguf_sizes']['Q4_K_M']:.2f} GB): the default for laptops
  and small servers. Mild quality dip versus full precision but fits
  almost anywhere.
- **Q5_K_M** ({m['gguf_sizes']['Q5_K_M']:.2f} GB): a step up if you have
  the RAM. Most people won't notice the difference from Q8.
- **Q8_0** ({m['gguf_sizes']['Q8_0']:.2f} GB): closest to the bf16 weights.
  Use this if you want the cleanest output and have the disk space.

## Prompt format

Same as the base model. Use the system prompt verbatim, put the headline
or short snippet in the user turn, and parse the `<answer>...</answer>`
block for the label.

## Limitations

GGUF is a faithful conversion of the base model, so the same caveats apply:

- English only
- Short text only (training context was 1024 tokens)
- Three labels: positive, negative, neutral
- It explains its read but it isn't doing finance research; don't use the
  reasoning chain as investment advice

Quantization adds a small extra error on top of the base model. For
Q4_K_M on a model this size you'll see occasional disagreement with the
bf16 model on borderline headlines, usually neutral-vs-positive flips.

## Related FinSenti models

Other sizes and bases trained with the same recipe:

{related}

The full-precision SafeTensors version of this model is at
[{m['full_repo']}](https://huggingface.co/{m['full_repo']}), and the
training data is at
[Ayansk11/FinSenti-Dataset](https://huggingface.co/datasets/Ayansk11/FinSenti-Dataset).

## Citation

```bibtex
@misc{{shaikh2026finsenti,
  title  = {{FinSenti: Small Language Models for Financial Sentiment with Chain-of-Thought Reasoning}},
  author = {{Shaikh, Ayan}},
  year   = {{2026}},
  url    = {{https://huggingface.co/collections/Ayansk11/finsenti}},
  note   = {{Indiana University}}
}}
```

## License

Apache 2.0.
"""
    return yaml + body


# -----------------------------------------------------------------------------
# Upload helpers
# -----------------------------------------------------------------------------

def _upload_card(api: HfApi, repo_id: str, content: str) -> str:
    """Push a README.md to the given repo. Returns 'ok', 'missing', or 'error: ...'."""
    try:
        api.repo_info(repo_id, repo_type="model")
    except RepositoryNotFoundError:
        return "missing"
    except HfHubHTTPError as exc:
        return f"error: {exc}"

    tmp = Path("/tmp") / f"finsenti-card-{repo_id.replace('/', '_')}.md"
    tmp.write_text(content, encoding="utf-8")
    try:
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model card",
        )
    except HfHubHTTPError as exc:
        return f"error: {exc}"
    finally:
        if tmp.exists():
            tmp.unlink()
    return "ok"


def _save_card(out_dir: Path, repo_id: str, content: str) -> Path:
    """Write a README.md to disk for review (used in --dry-run with --out)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = repo_id.replace("/", "__") + ".README.md"
    path = out_dir / fname
    path.write_text(content, encoding="utf-8")
    return path


def _check_no_em_dash(text: str, label: str) -> None:
    """Loud sanity check: bail out if any em-dash slipped into a card."""
    if "\u2014" in text or "—" in text:
        raise ValueError(f"em-dash found in {label} - clean the template")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and upload FinSenti model cards"
    )
    parser.add_argument(
        "--model",
        help="Only process this model key (e.g. qwen3-4b). Default: all.",
    )
    parser.add_argument(
        "--base-only", action="store_true", help="Skip the GGUF cards."
    )
    parser.add_argument(
        "--gguf-only", action="store_true", help="Skip the SafeTensors cards."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't upload. Print or save cards instead.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="With --dry-run, write generated cards to this directory.",
    )
    args = parser.parse_args()

    if args.model:
        targets = [m for m in MODELS if m["key"] == args.model]
        if not targets:
            keys = ", ".join(m["key"] for m in MODELS)
            raise SystemExit(f"unknown --model {args.model!r}. Choose from: {keys}")
    else:
        targets = MODELS

    api = HfApi(token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))

    print(f"Generating cards for {len(targets)} model(s)...")
    print()

    summary: list[tuple[str, str]] = []

    for m in targets:
        if not args.gguf_only:
            card = make_base_card(m)
            _check_no_em_dash(card, f"{m['key']} base")
            label = m["full_repo"]
            if args.dry_run:
                if args.out:
                    p = _save_card(args.out, label, card)
                    print(f"  [base] saved -> {p}")
                    summary.append((label, "saved"))
                else:
                    print(f"\n=== {label} (base) ===\n")
                    print(card[:600] + "\n...\n")
                    summary.append((label, "preview"))
            else:
                status = _upload_card(api, label, card)
                print(f"  [base] {label}: {status}")
                summary.append((label, status))

        if not args.base_only:
            card = make_gguf_card(m)
            _check_no_em_dash(card, f"{m['key']} gguf")
            label = m["gguf_repo"]
            if args.dry_run:
                if args.out:
                    p = _save_card(args.out, label, card)
                    print(f"  [gguf] saved -> {p}")
                    summary.append((label, "saved"))
                else:
                    print(f"\n=== {label} (gguf) ===\n")
                    print(card[:600] + "\n...\n")
                    summary.append((label, "preview"))
            else:
                status = _upload_card(api, label, card)
                print(f"  [gguf] {label}: {status}")
                summary.append((label, status))

    print()
    print("Summary:")
    for repo, status in summary:
        print(f"  {status:8s}  {repo}")


if __name__ == "__main__":
    main()

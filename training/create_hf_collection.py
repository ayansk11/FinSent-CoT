"""
Create/update the FinSenti HuggingFace Collection.

Groups all 19 repos (1 dataset + 10 base + 9 GGUF) into a single
collection for easy discovery.

Run once after all models are exported and uploaded:
    python training/create_hf_collection.py
    python training/create_hf_collection.py --dry-run   # Preview without creating
"""

import argparse

from huggingface_hub import HfApi


# ─── All repos in the collection ─────────────────────────────────────────────

DATASET_REPO = "Ayansk11/FinSenti-Dataset"

# Models ordered by family, then ascending parameter count.
MODELS = [
    # ── Qwen3 family ────────────────────────────────────────────────────
    {"key": "qwen3-0.6b",     "name": "Qwen3-0.6B",       "family": "Qwen3",    "full": "Ayansk11/FinSenti-Qwen3-0.6B",         "gguf": "Ayansk11/FinSenti-Qwen3-0.6B-GGUF"},
    {"key": "qwen3-1.7b",     "name": "Qwen3-1.7B",       "family": "Qwen3",    "full": "Ayansk11/FinSenti-Qwen3-1.7B",         "gguf": "Ayansk11/FinSenti-Qwen3-1.7B-GGUF"},
    {"key": "qwen3-4b",       "name": "Qwen3-4B",         "family": "Qwen3",    "full": "Ayansk11/FinSenti-Qwen3-4B",           "gguf": "Ayansk11/FinSenti-Qwen3-4B-GGUF"},
    {"key": "qwen3-8b",       "name": "Qwen3-8B",         "family": "Qwen3",    "full": "Ayansk11/FinSenti-Qwen3-8B",           "gguf": "Ayansk11/FinSenti-Qwen3-8B-GGUF"},
    # ── Qwen3.5 family ──────────────────────────────────────────────────
    {"key": "qwen3.5-0.8b",   "name": "Qwen3.5-0.8B",     "family": "Qwen3.5",  "full": "Ayansk11/FinSenti-Qwen3.5-0.8B",       "gguf": "Ayansk11/FinSenti-Qwen3.5-0.8B-GGUF"},
    {"key": "qwen3.5-2b",     "name": "Qwen3.5-2B",       "family": "Qwen3.5",  "full": "Ayansk11/FinSenti-Qwen3.5-2B",         "gguf": "Ayansk11/FinSenti-Qwen3.5-2B-GGUF"},
    {"key": "qwen3.5-4b",     "name": "Qwen3.5-4B",       "family": "Qwen3.5",  "full": "Ayansk11/FinSenti-Qwen3.5-4B",         "gguf": "Ayansk11/FinSenti-Qwen3.5-4B-GGUF"},
    {"key": "qwen3.5-9b",     "name": "Qwen3.5-9B",       "family": "Qwen3.5",  "full": "Ayansk11/FinSenti-Qwen3.5-9B",         "gguf": "Ayansk11/FinSenti-Qwen3.5-9B-GGUF"},
    # ── DeepSeek family ─────────────────────────────────────────────────
    {"key": "deepseek-r1-1.5b","name": "DeepSeek-R1-1.5B", "family": "DeepSeek", "full": "Ayansk11/FinSenti-DeepSeek-R1-1.5B",   "gguf": "Ayansk11/FinSenti-DeepSeek-R1-1.5B-GGUF"},
    # ── MobileLLM family ────────────────────────────────────────────────
    {"key": "mobilellm-r1-950m","name": "MobileLLM-R1-950M","family": "MobileLLM","full": "Ayansk11/FinSenti-MobileLLM-R1-950M",  "gguf": None},
    # ── Gemma 4 family ──────────────────────────────────────────────────
    {"key": "gemma4-e2b",      "name": "Gemma4-E2B",       "family": "Gemma4",   "full": "Ayansk11/FinSenti-Gemma4-E2B",         "gguf": "Ayansk11/FinSenti-Gemma4-E2B-GGUF"},
    {"key": "gemma4-e4b",      "name": "Gemma4-E4B",       "family": "Gemma4",   "full": "Ayansk11/FinSenti-Gemma4-E4B",         "gguf": "Ayansk11/FinSenti-Gemma4-E4B-GGUF"},
    {"key": "gemma4-26b-a4b",  "name": "Gemma4-26B-A4B",   "family": "Gemma4",   "full": "Ayansk11/FinSenti-Gemma4-26B-A4B",     "gguf": "Ayansk11/FinSenti-Gemma4-26B-A4B-GGUF"},
    # ── Tiny-LLM family (scaling lower bound) ───────────────────────────
    {"key": "tiny-llm-10m",    "name": "Tiny-LLM-10M",     "family": "TinyLLM",  "full": "Ayansk11/FinSenti-Tiny-LLM-10M",       "gguf": "Ayansk11/FinSenti-Tiny-LLM-10M-GGUF"},
    # ── Llama 3.2 family ────────────────────────────────────────────────
    {"key": "llama-3.2-1b",    "name": "Llama-3.2-1B",     "family": "Llama3",   "full": "Ayansk11/FinSenti-Llama-3.2-1B",       "gguf": "Ayansk11/FinSenti-Llama-3.2-1B-GGUF"},
    # ── SmolLM family ───────────────────────────────────────────────────
    {"key": "smollm-1.7b",     "name": "SmolLM-1.7B",      "family": "SmolLM",   "full": "Ayansk11/FinSenti-SmolLM-1.7B",        "gguf": "Ayansk11/FinSenti-SmolLM-1.7B-GGUF"},
]

COLLECTION_TITLE = "FinSenti"
COLLECTION_DESCRIPTION = (
    "Financial sentiment analysis with chain-of-thought reasoning. "
    "16 small models (Qwen3, Qwen3.5, DeepSeek-R1, MobileLLM, Gemma 4, "
    "Tiny-LLM, Llama 3.2, SmolLM) fine-tuned with SFT + GRPO on 16.9K "
    "balanced samples. Scaling study spanning 8 architectures from 10M "
    "to 9B parameters. Each model available as full-precision SafeTensors "
    "and GGUF (Q4_K_M, Q5_K_M, Q8_0) for Ollama/llama.cpp on consumer hardware."
)


def main():
    parser = argparse.ArgumentParser(description="Create FinSenti HuggingFace collection")
    parser.add_argument("--dry-run", action="store_true", help="Preview repos without creating")
    parser.add_argument("--namespace", default="Ayansk11", help="HF namespace")
    args = parser.parse_args()

    items = []

    # 1. Dataset
    items.append({"repo_id": DATASET_REPO, "type": "dataset",
                  "note": "Training dataset - 50.8K balanced samples"})

    # 2. Base models + GGUF (interleaved per model for easy browsing)
    for m in MODELS:
        items.append({"repo_id": m["full"], "type": "model",
                      "note": f"{m['name']} - Full precision SafeTensors"})
        if m["gguf"]:
            items.append({"repo_id": m["gguf"], "type": "model",
                          "note": f"{m['name']} - GGUF (Q4_K_M, Q5_K_M, Q8_0)"})

    n_gguf = sum(1 for m in MODELS if m["gguf"])
    print(f"FinSenti Collection")
    print(f"  Total items: {len(items)} (1 dataset + {len(MODELS)} base + {n_gguf} GGUF)")
    print()

    for i, item in enumerate(items, 1):
        print(f"  {i:2d}. [{item['type']:7s}] {item['repo_id']}")
        if item.get("note"):
            print(f"      {item['note']}")

    if args.dry_run:
        print("\n  --dry-run: No collection created.")
        return

    api = HfApi()
    slug = COLLECTION_TITLE.lower().replace(" ", "-")

    # Check for existing collection
    try:
        existing = api.list_collections(owner=args.namespace, item=DATASET_REPO)
        existing = [c for c in existing if COLLECTION_TITLE.lower() in c.title.lower()]
    except Exception:
        existing = []

    if existing:
        collection = existing[0]
        print(f"\n  Found existing collection: {collection.slug}")
    else:
        collection = api.create_collection(
            title=COLLECTION_TITLE,
            description=COLLECTION_DESCRIPTION,
            namespace=args.namespace,
            private=False,
        )
        print(f"\n  Created collection: {collection.slug}")

    for item in items:
        try:
            api.add_collection_item(
                collection_slug=collection.slug,
                item_id=item["repo_id"],
                item_type=item["type"],
                note=item.get("note", ""),
                exists_ok=True,
            )
        except Exception as e:
            print(f"  [WARN] Could not add {item['repo_id']}: {e}")

    print(f"\n  Collection URL: https://huggingface.co/collections/{collection.slug}")


if __name__ == "__main__":
    main()

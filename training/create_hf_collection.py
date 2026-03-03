"""
Create/update the FinSent-CoT HuggingFace Collection.

Groups all 41 repos (1 dataset + 10 full-precision + 30 GGUF) into a single
collection for easy discovery.

Run once after all models are exported and uploaded:
    python training/create_hf_collection.py
    python training/create_hf_collection.py --dry-run   # Preview without creating
"""

import argparse

from huggingface_hub import HfApi


# ─── All repos in the collection ─────────────────────────────────────────────

DATASET_REPO = "Ayansk11/FinSent-CoT-Dataset"

MODELS = [
    {
        "key": "qwen3-0.6b",
        "name": "Qwen3-0.6B",
        "full": "Ayansk11/FinSent-CoT-Qwen3-0.6B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3-0.6B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3-0.6B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3-0.6B-Q8_0",
        },
    },
    {
        "key": "qwen3-1.7b",
        "name": "Qwen3-1.7B",
        "full": "Ayansk11/FinSent-CoT-Qwen3-1.7B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3-1.7B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3-1.7B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3-1.7B-Q8_0",
        },
    },
    {
        "key": "qwen3-4b",
        "name": "Qwen3-4B",
        "full": "Ayansk11/FinSent-CoT-Qwen3-4B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3-4B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3-4B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3-4B-Q8_0",
        },
    },
    {
        "key": "qwen3-8b",
        "name": "Qwen3-8B",
        "full": "Ayansk11/FinSent-CoT-Qwen3-8B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3-8B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3-8B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3-8B-Q8_0",
        },
    },
    {
        "key": "deepseek-r1-1.5b",
        "name": "DeepSeek-R1-1.5B",
        "full": "Ayansk11/FinSent-CoT-DeepSeek-R1-1.5B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-DeepSeek-R1-1.5B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-DeepSeek-R1-1.5B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-DeepSeek-R1-1.5B-Q8_0",
        },
    },
    {
        "key": "mobilellm-r1-950m",
        "name": "MobileLLM-R1-950M",
        "full": "Ayansk11/FinSent-CoT-MobileLLM-R1-950M",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-MobileLLM-R1-950M-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-MobileLLM-R1-950M-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-MobileLLM-R1-950M-Q8_0",
        },
    },
    {
        "key": "qwen3.5-0.8b",
        "name": "Qwen3.5-0.8B",
        "full": "Ayansk11/FinSent-CoT-Qwen3.5-0.8B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-0.8B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-0.8B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3.5-0.8B-Q8_0",
        },
    },
    {
        "key": "qwen3.5-2b",
        "name": "Qwen3.5-2B",
        "full": "Ayansk11/FinSent-CoT-Qwen3.5-2B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-2B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-2B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3.5-2B-Q8_0",
        },
    },
    {
        "key": "qwen3.5-4b",
        "name": "Qwen3.5-4B",
        "full": "Ayansk11/FinSent-CoT-Qwen3.5-4B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-4B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-4B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3.5-4B-Q8_0",
        },
    },
    {
        "key": "qwen3.5-9b",
        "name": "Qwen3.5-9B",
        "full": "Ayansk11/FinSent-CoT-Qwen3.5-9B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-9B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSent-CoT-Qwen3.5-9B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSent-CoT-Qwen3.5-9B-Q8_0",
        },
    },
]

COLLECTION_TITLE = "FinSent-CoT"
COLLECTION_DESCRIPTION = (
    "Financial sentiment analysis with chain-of-thought reasoning. "
    "10 small models (Qwen3, Qwen3.5, DeepSeek-R1, MobileLLM) fine-tuned "
    "with SFT + GRPO on 16.9K balanced samples (positive/negative/neutral). "
    "Each model available in full-precision HF weights and 3 GGUF "
    "quantizations (Q4_K_M, Q5_K_M, Q8_0) for Ollama/llama.cpp deployment "
    "on consumer hardware."
)


def main():
    parser = argparse.ArgumentParser(description="Create FinSent-CoT HuggingFace collection")
    parser.add_argument("--dry-run", action="store_true", help="Preview repos without creating")
    parser.add_argument("--namespace", default="Ayansk11", help="HF namespace")
    args = parser.parse_args()

    # Build ordered list of all repos to add
    items = []

    # Dataset first
    items.append({"repo_id": DATASET_REPO, "type": "dataset", "note": "Training dataset"})

    # Then each model: full-precision, then GGUF variants
    for m in MODELS:
        items.append({"repo_id": m["full"], "type": "model", "note": f"{m['name']} full-precision"})
        for quant, repo in m["gguf"].items():
            items.append({"repo_id": repo, "type": "model", "note": f"{m['name']} {quant} GGUF"})

    print(f"FinSent-CoT Collection")
    print(f"  Total items: {len(items)} (1 dataset + {len(MODELS)} full + {len(MODELS) * 3} GGUF)")
    print()

    for i, item in enumerate(items, 1):
        print(f"  {i:2d}. [{item['type']:7s}] {item['repo_id']}")

    if args.dry_run:
        print(f"\n  --dry-run: No collection created.")
        return

    # Create/update collection
    api = HfApi()

    print(f"\nCreating collection '{COLLECTION_TITLE}'...")
    collection = api.create_collection(
        title=COLLECTION_TITLE,
        description=COLLECTION_DESCRIPTION,
        namespace=args.namespace,
        exists_ok=True,
    )
    print(f"  Collection URL: {collection.url}")
    print(f"  Collection slug: {collection.slug}")

    # Get existing items to avoid duplicates
    existing_ids = set()
    if collection.items:
        existing_ids = {item.item_id for item in collection.items}

    added = 0
    skipped = 0
    for item in items:
        item_id = item["repo_id"]
        item_type = item["type"]

        if item_id in existing_ids:
            skipped += 1
            continue

        try:
            api.add_collection_item(
                collection_slug=collection.slug,
                item_id=item_id,
                item_type=item_type,
                note=item["note"],
            )
            added += 1
            print(f"  Added: {item_id}")
        except Exception as e:
            print(f"  Failed to add {item_id}: {e}")

    print(f"\nDone! Added {added}, skipped {skipped} (already in collection)")
    print(f"Collection: {collection.url}")


if __name__ == "__main__":
    main()

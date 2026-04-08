"""
Create/update the FinSenti HuggingFace Collection.

Groups all 41 repos (1 dataset + 10 full-precision + 30 GGUF) into a single
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
# This order is used directly in the HF collection for easy navigation.
MODELS = [
    # ── Qwen3 family (oldest → largest) ──────────────────────────────────
    {
        "key": "qwen3-0.6b",
        "name": "Qwen3-0.6B",
        "family": "Qwen3",
        "full": "Ayansk11/FinSenti-Qwen3-0.6B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-Qwen3-0.6B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-Qwen3-0.6B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-Qwen3-0.6B-Q8_0",
        },
    },
    {
        "key": "qwen3-1.7b",
        "name": "Qwen3-1.7B",
        "family": "Qwen3",
        "full": "Ayansk11/FinSenti-Qwen3-1.7B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-Qwen3-1.7B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-Qwen3-1.7B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-Qwen3-1.7B-Q8_0",
        },
    },
    {
        "key": "qwen3-4b",
        "name": "Qwen3-4B",
        "family": "Qwen3",
        "full": "Ayansk11/FinSenti-Qwen3-4B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-Qwen3-4B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-Qwen3-4B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-Qwen3-4B-Q8_0",
        },
    },
    {
        "key": "qwen3-8b",
        "name": "Qwen3-8B",
        "family": "Qwen3",
        "full": "Ayansk11/FinSenti-Qwen3-8B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-Qwen3-8B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-Qwen3-8B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-Qwen3-8B-Q8_0",
        },
    },
    # ── Qwen3.5 family (newest, ascending) ───────────────────────────────
    {
        "key": "qwen3.5-0.8b",
        "name": "Qwen3.5-0.8B",
        "family": "Qwen3.5",
        "full": "Ayansk11/FinSenti-Qwen3.5-0.8B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-Qwen3.5-0.8B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-Qwen3.5-0.8B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-Qwen3.5-0.8B-Q8_0",
        },
    },
    {
        "key": "qwen3.5-2b",
        "name": "Qwen3.5-2B",
        "family": "Qwen3.5",
        "full": "Ayansk11/FinSenti-Qwen3.5-2B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-Qwen3.5-2B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-Qwen3.5-2B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-Qwen3.5-2B-Q8_0",
        },
    },
    {
        "key": "qwen3.5-4b",
        "name": "Qwen3.5-4B",
        "family": "Qwen3.5",
        "full": "Ayansk11/FinSenti-Qwen3.5-4B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-Qwen3.5-4B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-Qwen3.5-4B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-Qwen3.5-4B-Q8_0",
        },
    },
    {
        "key": "qwen3.5-9b",
        "name": "Qwen3.5-9B",
        "family": "Qwen3.5",
        "full": "Ayansk11/FinSenti-Qwen3.5-9B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-Qwen3.5-9B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-Qwen3.5-9B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-Qwen3.5-9B-Q8_0",
        },
    },
    # ── DeepSeek family ──────────────────────────────────────────────────
    {
        "key": "deepseek-r1-1.5b",
        "name": "DeepSeek-R1-1.5B",
        "family": "DeepSeek",
        "full": "Ayansk11/FinSenti-DeepSeek-R1-1.5B",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-DeepSeek-R1-1.5B-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-DeepSeek-R1-1.5B-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-DeepSeek-R1-1.5B-Q8_0",
        },
    },
    # ── MobileLLM family ─────────────────────────────────────────────────
    {
        "key": "mobilellm-r1-950m",
        "name": "MobileLLM-R1-950M",
        "family": "MobileLLM",
        "full": "Ayansk11/FinSenti-MobileLLM-R1-950M",
        "gguf": {
            "Q4_K_M": "Ayansk11/FinSenti-MobileLLM-R1-950M-Q4_K_M",
            "Q5_K_M": "Ayansk11/FinSenti-MobileLLM-R1-950M-Q5_K_M",
            "Q8_0":   "Ayansk11/FinSenti-MobileLLM-R1-950M-Q8_0",
        },
    },
]

COLLECTION_TITLE = "FinSenti"
COLLECTION_DESCRIPTION = (
    "Financial sentiment analysis with chain-of-thought reasoning. "
    "10 small models (Qwen3, Qwen3.5, DeepSeek-R1, MobileLLM) fine-tuned "
    "with SFT + GRPO on 16.9K balanced samples (positive/negative/neutral). "
    "Each model available in full-precision HF weights and 3 GGUF "
    "quantizations (Q4_K_M, Q5_K_M, Q8_0) for Ollama/llama.cpp deployment "
    "on consumer hardware."
)


def main():
    parser = argparse.ArgumentParser(description="Create FinSenti HuggingFace collection")
    parser.add_argument("--dry-run", action="store_true", help="Preview repos without creating")
    parser.add_argument("--namespace", default="Ayansk11", help="HF namespace")
    args = parser.parse_args()

    # Build ordered list of all repos to add.
    # Order: Dataset → Full-precision (by family, ascending size) → GGUF (same order, Q8→Q5→Q4)
    # This makes the collection easy to browse: best-quality formats first,
    # then progressively smaller quantizations grouped by model.
    items = []

    # 1. Dataset
    items.append({"repo_id": DATASET_REPO, "type": "dataset",
                  "note": "📊 Training dataset — 50.8K balanced samples (positive/negative/neutral)"})

    # 2. Full-precision models (grouped by family, ascending size)
    for m in MODELS:
        items.append({"repo_id": m["full"], "type": "model",
                      "note": f"🔬 {m['name']} — Full precision (SafeTensors) for fine-tuning & vLLM inference"})

    # 3. GGUF quantizations (same model order, best quality first: Q8 → Q5 → Q4)
    quant_order = ["Q8_0", "Q5_K_M", "Q4_K_M"]
    quant_labels = {
        "Q8_0": "8-bit — best quality",
        "Q5_K_M": "5-bit — balanced",
        "Q4_K_M": "4-bit — smallest/fastest",
    }
    for quant in quant_order:
        for m in MODELS:
            if quant in m["gguf"]:
                items.append({"repo_id": m["gguf"][quant], "type": "model",
                              "note": f"📦 {m['name']} GGUF {quant} — {quant_labels[quant]}"})

    print(f"FinSenti Collection")
    print(f"  Total items: {len(items)} (1 dataset + {len(MODELS)} full-precision + {len(MODELS) * 3} GGUF)")
    print()

    # Pretty-print with section headers
    section = None
    for i, item in enumerate(items, 1):
        note = item.get("note", "")
        # Detect section transitions based on note prefix
        if "Training dataset" in note and section != "dataset":
            section = "dataset"
            print("  ── Dataset ──────────────────────────────────────────")
        elif "Full precision" in note and section != "full":
            section = "full"
            print("  ── Full-Precision Models (SafeTensors) ──────────────")
        elif "GGUF Q8_0" in note and section != "gguf-q8":
            section = "gguf-q8"
            print("  ── GGUF Q8_0 (8-bit, best quality) ──────────────────")
        elif "GGUF Q5_K_M" in note and section != "gguf-q5":
            section = "gguf-q5"
            print("  ── GGUF Q5_K_M (5-bit, balanced) ────────────────────")
        elif "GGUF Q4_K_M" in note and section != "gguf-q4":
            section = "gguf-q4"
            print("  ── GGUF Q4_K_M (4-bit, smallest) ────────────────────")
        print(f"  {i:2d}. {item['repo_id']}")

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

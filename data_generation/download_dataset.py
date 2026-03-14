"""Download FinSent-CoT-Dataset from HuggingFace and save as validated/*.jsonl files."""

import json
import os
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "validated")
DATASET_REPO = "Ayansk11/FinSent-CoT-Dataset"
SUBSETS = ["sft", "grpo", "raw"]
SPLITS = ["train", "validation", "test"]
# HF uses "validation" but our files use "val"
SPLIT_RENAME = {"validation": "val", "train": "train", "test": "test"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 0

    for subset in SUBSETS:
        print(f"\nDownloading subset: {subset}")
        ds = load_dataset(DATASET_REPO, subset)

        for split in SPLITS:
            if split not in ds:
                print(f"  {split}: not found, skipping")
                continue

            local_split = SPLIT_RENAME[split]
            filename = f"{subset}_{local_split}.jsonl"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "w") as f:
                for row in ds[split]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            count = len(ds[split])
            total += count
            print(f"  {filename}: {count} samples")

    print(f"\nTotal: {total} samples written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

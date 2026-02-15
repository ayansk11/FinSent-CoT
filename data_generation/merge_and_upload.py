"""
Merge validated dataset and upload to HuggingFace.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace")
    parser.add_argument("--validated-dir", default="./validated",
                        help="Directory with validated splits")
    parser.add_argument("--repo-id", default="Ayansk11/FinSent-CoT-50k",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--format", choices=["grpo", "sft", "raw"], default="grpo",
                        help="Which format to upload")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually upload, just show what would be done")
    args = parser.parse_args()

    validated_dir = Path(args.validated_dir)
    prefix = args.format

    # Load splits
    print(f"Loading {prefix} format from {validated_dir}/...")
    train = load_jsonl(validated_dir / f"{prefix}_train.jsonl")
    val = load_jsonl(validated_dir / f"{prefix}_val.jsonl")
    test = load_jsonl(validated_dir / f"{prefix}_test.jsonl")

    print(f"  train: {len(train)} samples")
    print(f"  val:   {len(val)} samples")
    print(f"  test:  {len(test)} samples")

    # Create HuggingFace datasets
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    })

    print(f"\nDataset: {dataset_dict}")

    if args.dry_run:
        print("\n[DRY RUN] Would upload to:", args.repo_id)
        return

    # Upload
    print(f"\nUploading to {args.repo_id}...")
    dataset_dict.push_to_hub(
        args.repo_id,
        private=False,
        commit_message=f"Upload {prefix} format ({len(train)+len(val)+len(test)} samples)",
    )
    print("Upload complete!")

    # Also upload all formats as additional files
    api = HfApi()
    for fmt in ["grpo", "sft", "raw"]:
        for split in ["train", "val", "test"]:
            fpath = validated_dir / f"{fmt}_{split}.jsonl"
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=f"data/{fmt}_{split}.jsonl",
                    repo_id=args.repo_id,
                    repo_type="dataset",
                )
                print(f"  Uploaded {fmt}_{split}.jsonl")

    # Upload validation summary
    summary_path = validated_dir / "validation_summary.json"
    if summary_path.exists():
        api.upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo="validation_summary.json",
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print("\nAll files uploaded!")


if __name__ == "__main__":
    main()

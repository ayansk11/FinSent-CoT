"""
Dataset validation and quality filtering.
Reads raw generated CoT data and produces a clean, validated dataset.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


REASONING_PATTERN = re.compile(
    r"<reasoning>\s*(.+?)\s*</reasoning>", re.DOTALL | re.IGNORECASE
)
ANSWER_PATTERN = re.compile(
    r"<answer>\s*(positive|negative|neutral)\s*</answer>", re.IGNORECASE
)


def validate_sample(sample: dict) -> tuple[bool, list[str]]:
    """
    Validate a single sample for quality.
    Returns (is_valid, list_of_issues).
    """
    issues = []

    text = sample.get("text", "").strip()
    reasoning = sample.get("reasoning", "").strip()
    answer = sample.get("answer", "").strip().lower()
    label = sample.get("label", "").strip().lower()

    # Basic checks
    if not text:
        issues.append("empty_text")
    elif len(text.split()) < 5:
        issues.append("text_too_short")

    if not reasoning:
        issues.append("empty_reasoning")
    elif len(reasoning.split()) < 20:
        issues.append("reasoning_too_short")

    if answer not in ("positive", "negative", "neutral"):
        issues.append("invalid_answer")

    if label not in ("positive", "negative", "neutral"):
        issues.append("invalid_label")

    # Check answer matches label
    if answer and label and answer != label:
        issues.append("answer_label_mismatch")

    # Check reasoning isn't just repeating the input
    if text and reasoning:
        text_words = set(text.lower().split())
        reasoning_words = set(reasoning.lower().split())
        if len(reasoning_words) > 0:
            overlap = len(text_words & reasoning_words) / len(reasoning_words)
            if overlap > 0.8:
                issues.append("reasoning_is_copy_of_input")

    # Check reasoning contains actual analysis (not just filler)
    if reasoning:
        filler_phrases = [
            "as an ai", "i cannot", "i don't have", "based on the text",
            "the text says", "the text mentions"
        ]
        reasoning_lower = reasoning.lower()
        filler_count = sum(1 for p in filler_phrases if p in reasoning_lower)
        analysis_phrases = [
            "indicates", "suggests", "implies", "reflects",
            "revenue", "earnings", "growth", "decline", "market",
            "positive", "negative", "neutral", "sentiment",
            "financial", "stock", "price", "analyst", "outlook"
        ]
        analysis_count = sum(1 for p in analysis_phrases if p in reasoning_lower)
        if filler_count > analysis_count:
            issues.append("reasoning_lacks_analysis")

    return len(issues) == 0, issues


def deduplicate(samples: list[dict]) -> list[dict]:
    """Remove duplicates based on text hash."""
    seen = set()
    unique = []
    for s in samples:
        h = hashlib.md5(s.get("text", "").lower().strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(s)
    return unique


def format_for_grpo(sample: dict) -> dict:
    """Convert a validated sample to GRPO training format."""
    prompt = (
        f'Analyze the sentiment of the following financial text:\n\n'
        f'"{sample["text"]}"\n\n'
        f'Provide your reasoning in <reasoning> tags and your '
        f'classification (positive, negative, or neutral) in <answer> tags.'
    )
    answer = (
        f'<reasoning>\n{sample["reasoning"]}\n</reasoning>\n'
        f'<answer>{sample["answer"]}</answer>'
    )
    return {"prompt": prompt, "answer": answer, "label": sample["answer"]}


def format_for_sft(sample: dict) -> dict:
    """Convert a validated sample to SFT training format."""
    return {
        "instruction": (
            "You are a financial sentiment analyst. Analyze the given financial text "
            "and provide your reasoning in <reasoning> tags and your sentiment "
            "classification (positive, negative, or neutral) in <answer> tags."
        ),
        "input": sample["text"],
        "output": (
            f'<reasoning>\n{sample["reasoning"]}\n</reasoning>\n'
            f'<answer>{sample["answer"]}</answer>'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate and clean CoT dataset")
    parser.add_argument("--input", required=True, help="Path to generated_cot.jsonl")
    parser.add_argument("--output-dir", default="./validated", help="Output directory")
    parser.add_argument("--strict", action="store_true",
                        help="Only keep samples that pass all quality checks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"Loading {args.input}...")
    samples = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Raw samples: {len(samples)}")

    # Deduplicate
    samples = deduplicate(samples)
    print(f"After dedup: {len(samples)}")

    # Validate
    valid_samples = []
    issue_counter = Counter()
    for s in samples:
        is_valid, issues = validate_sample(s)
        if is_valid or (not args.strict and len(issues) <= 1 and "reasoning_too_short" not in issues):
            valid_samples.append(s)
        for issue in issues:
            issue_counter[issue] += 1

    print(f"After validation: {len(valid_samples)}")
    print("\nIssue breakdown:")
    for issue, count in issue_counter.most_common():
        print(f"  {issue}: {count}")

    # Balance classes
    by_label = {"positive": [], "negative": [], "neutral": []}
    for s in valid_samples:
        by_label[s["answer"]].append(s)

    min_count = min(len(v) for v in by_label.values())
    print(f"\nPer-class counts: { {k: len(v) for k, v in by_label.items()} }")
    print(f"Balancing to {min_count} per class ({min_count * 3} total)")

    balanced = []
    for label in ["positive", "negative", "neutral"]:
        balanced.extend(by_label[label][:min_count])

    import random
    random.seed(42)
    random.shuffle(balanced)

    # Split: 90% train, 5% val, 5% test
    n = len(balanced)
    train_end = int(n * 0.90)
    val_end = int(n * 0.95)

    train_set = balanced[:train_end]
    val_set = balanced[train_end:val_end]
    test_set = balanced[val_end:]

    print(f"\nSplits: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # Write outputs
    for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        # GRPO format
        grpo_path = output_dir / f"grpo_{split_name}.jsonl"
        with open(grpo_path, "w") as f:
            for s in split_data:
                f.write(json.dumps(format_for_grpo(s)) + "\n")

        # SFT format
        sft_path = output_dir / f"sft_{split_name}.jsonl"
        with open(sft_path, "w") as f:
            for s in split_data:
                f.write(json.dumps(format_for_sft(s)) + "\n")

        # Raw format
        raw_path = output_dir / f"raw_{split_name}.jsonl"
        with open(raw_path, "w") as f:
            for s in split_data:
                f.write(json.dumps(s) + "\n")

    # Summary stats
    summary = {
        "raw_count": len(samples),
        "valid_count": len(valid_samples),
        "balanced_count": len(balanced),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
        "per_class": min_count,
        "issues": dict(issue_counter),
    }
    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs written to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()

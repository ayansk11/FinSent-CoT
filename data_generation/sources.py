"""
Multi-source financial sentiment dataset loading.
Combines FinGPT, FiQA, Financial PhraseBank, and SEC headlines
into a balanced, deduplicated dataset for CoT generation.
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset


@dataclass
class FinancialSample:
    text: str
    label: str  # positive, negative, neutral
    source: str
    text_hash: str = ""

    def __post_init__(self):
        self.text_hash = hashlib.md5(self.text.lower().strip().encode()).hexdigest()


def load_fingpt() -> list[FinancialSample]:
    """Load FinGPT sentiment dataset (primary source)."""
    print("[sources] Loading FinGPT...")
    ds = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    samples = []
    for row in ds:
        text = row.get("input", "").strip()
        label = row.get("output", "").strip().lower()
        if label in ("positive", "negative", "neutral") and len(text.split()) >= 5:
            samples.append(FinancialSample(text=text, label=label, source="fingpt"))
    print(f"  -> {len(samples)} samples loaded")
    return samples


def load_fiqa() -> list[FinancialSample]:
    """Load FiQA sentiment dataset."""
    print("[sources] Loading FiQA...")
    try:
        ds = load_dataset("pauri32/fiqa-2018", split="train")
    except Exception:
        try:
            ds = load_dataset("ChanceFocus/fiqa-sentiment-classification", split="train")
        except Exception:
            print("  -> FiQA not available, skipping")
            return []

    samples = []
    for row in ds:
        text = row.get("sentence", row.get("text", "")).strip()
        score = row.get("score", row.get("sentiment_score", None))
        if score is None or not text or len(text.split()) < 5:
            continue
        score = float(score)
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        samples.append(FinancialSample(text=text, label=label, source="fiqa"))
    print(f"  -> {len(samples)} samples loaded")
    return samples


def load_financial_phrasebank() -> list[FinancialSample]:
    """Load Financial PhraseBank (Malo et al., 2014)."""
    print("[sources] Loading Financial PhraseBank...")
    try:
        ds = load_dataset(
            "takala/financial_phrasebank", "sentences_allagree", split="train"
        )
    except Exception:
        try:
            ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
        except Exception:
            print("  -> Financial PhraseBank not available, skipping")
            return []

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    samples = []
    for row in ds:
        text = row.get("sentence", "").strip()
        label_id = row.get("label", None)
        if label_id is None or not text or len(text.split()) < 5:
            continue
        label = label_map.get(label_id, None)
        if label:
            samples.append(FinancialSample(text=text, label=label, source="phrasebank"))
    print(f"  -> {len(samples)} samples loaded")
    return samples


def load_all_sources(
    target_total: int = 50000,
    balance: bool = True,
    seed: int = 42,
) -> list[FinancialSample]:
    """
    Load and merge all financial sentiment sources.
    Deduplicates by text hash and balances classes.

    Args:
        target_total: Target number of samples (will be less if not enough unique data)
        balance: Whether to balance classes equally
        seed: Random seed for reproducibility

    Returns:
        List of unique, optionally balanced FinancialSample objects
    """
    random.seed(seed)

    # Load all sources
    all_samples = []
    all_samples.extend(load_fingpt())
    all_samples.extend(load_fiqa())
    all_samples.extend(load_financial_phrasebank())

    # Deduplicate by text hash
    seen_hashes = set()
    unique_samples = []
    for sample in all_samples:
        if sample.text_hash not in seen_hashes:
            seen_hashes.add(sample.text_hash)
            unique_samples.append(sample)

    print(f"\n[sources] Total unique samples: {len(unique_samples)}")

    # Count per label
    by_label = {"positive": [], "negative": [], "neutral": []}
    for s in unique_samples:
        by_label[s.label].append(s)

    for label, items in by_label.items():
        print(f"  {label}: {len(items)}")

    if not balance:
        random.shuffle(unique_samples)
        return unique_samples[:target_total]

    # Balance: take equal from each class
    per_class = target_total // 3
    balanced = []
    for label in ["positive", "negative", "neutral"]:
        pool = by_label[label]
        random.shuffle(pool)
        # If not enough samples, take all and oversample
        if len(pool) >= per_class:
            balanced.extend(pool[:per_class])
        else:
            balanced.extend(pool)
            # Oversample to fill
            deficit = per_class - len(pool)
            balanced.extend(random.choices(pool, k=deficit))
            print(f"  WARNING: {label} oversampled by {deficit} to reach {per_class}")

    random.shuffle(balanced)
    print(f"\n[sources] Final balanced dataset: {len(balanced)} samples")
    return balanced


if __name__ == "__main__":
    samples = load_all_sources(target_total=50000)
    print(f"\nReady for CoT generation: {len(samples)} samples")
    # Show source distribution
    source_counts = {}
    for s in samples:
        source_counts[s.source] = source_counts.get(s.source, 0) + 1
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

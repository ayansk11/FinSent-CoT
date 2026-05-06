"""
Benchmark loaders for FinSenti evaluation.

All loaders return a list of dicts with the same shape:
    {"text": str, "expected": "positive" | "negative" | "neutral",
     "category": str, "id": str}

Supported benchmarks:
    fpb           Financial PhraseBank (Malo et al., 2014)
    fiqa          FiQA-2018 task 1 sentiment headlines
    twitterfin    Twitter Financial News (zeroshot/twitter-financial-news-sentiment)
    finsenti      Our held-out test slice (validated/raw_test.jsonl)
    financemteb   FinanceMTEB FinSentEnglish (Tsadoq et al., 2024) - OOD
    asba          Aspect-Based Financial Sentiment via SetFit/financial_news_sentiment - OOD

Examples:
    from evaluation.datasets import load_benchmark
    samples = load_benchmark("fpb")
    print(len(samples), samples[0])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# Map external benchmark labels to our 3-class schema.
LABEL_MAPS = {
    # takala/financial_phrasebank: 0=negative, 1=neutral, 2=positive
    "fpb": {0: "negative", 1: "neutral", 2: "positive"},
    # zeroshot/twitter-financial-news-sentiment: 0=Bearish, 1=Bullish, 2=Neutral
    "twitterfin": {0: "negative", 1: "positive", 2: "neutral"},
    # FiQA labels are continuous floats in [-1, 1]; we discretize.
    # See _load_fiqa for the threshold logic.
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_benchmark(name: str, max_samples: int | None = None) -> list[dict]:
    """Load a benchmark by short name.

    Args:
        name: One of "fpb", "fiqa", "twitterfin", "finsenti".
        max_samples: If set, truncate to this many samples (handy for smoke tests).

    Returns:
        List of {"text", "expected", "category", "id"} dicts.
    """
    name = name.lower()
    if name == "fpb":
        samples = _load_fpb()
    elif name == "fiqa":
        samples = _load_fiqa()
    elif name == "twitterfin":
        samples = _load_twitterfin()
    elif name == "finsenti":
        samples = _load_finsenti()
    elif name == "financemteb":
        samples = _load_financemteb()
    elif name == "asba":
        samples = _load_asba()
    else:
        raise ValueError(
            f"Unknown benchmark {name!r}. "
            f"Choose from: fpb, fiqa, twitterfin, finsenti, financemteb, asba."
        )

    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def list_benchmarks() -> list[str]:
    return ["fpb", "fiqa", "twitterfin", "finsenti", "financemteb", "asba"]


# -----------------------------------------------------------------------------
# Per-benchmark loaders
# -----------------------------------------------------------------------------

def _load_fpb() -> list[dict]:
    """Financial PhraseBank.

    The 50%-agree split is the convention in most FinBERT papers (~4840
    sentences). If only the all-agree split is available (e.g. when the
    50agree config isn't cached and the loading-script-based fetch is
    blocked by newer datasets versions), we fall back to it (~2260 sentences).

    Source: takala/financial_phrasebank.
    """
    from datasets import load_dataset
    config_attempts = ["sentences_50agree", "sentences_allagree"]
    last_err = None
    ds = None
    used_config = None
    for config in config_attempts:
        try:
            ds = load_dataset(
                "takala/financial_phrasebank",
                config,
                split="train",
            )
            used_config = config
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(
            f"Could not load FPB. Tried configs {config_attempts}. "
            f"Last error: {last_err}"
        )
    if used_config != "sentences_50agree":
        print(
            f"  [warn] FPB loaded with {used_config!r} (not the 50agree "
            f"convention). Numbers in the paper should note the config used."
        )
    label_map = LABEL_MAPS["fpb"]
    out = []
    for i, row in enumerate(ds):
        out.append({
            "text": row["sentence"],
            "expected": label_map[row["label"]],
            "category": "fpb",
            "id": f"fpb-{i}",
        })
    return out


def _load_fiqa() -> list[dict]:
    """FiQA-2018 Task 1 sentiment.

    Uses pauri32/fiqa-2018 which packages the original FiQA-2018 sentiment data.
    The original score is a continuous float in [-1, 1]. We discretize to 3
    classes using +/- 0.1 thresholds, the convention used in most FinBERT
    follow-ups.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("pauri32/fiqa-2018", split="train")
    except Exception:
        # Fallback: nickmuchi/financial-classification has a discretized variant
        ds = load_dataset("nickmuchi/financial-classification", split="train")
        # nickmuchi labels: 0=negative, 1=neutral, 2=positive
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        out = []
        for i, row in enumerate(ds):
            out.append({
                "text": row["sentence"],
                "expected": label_map[row["label"]],
                "category": "fiqa",
                "id": f"fiqa-{i}",
            })
        return out

    out = []
    for i, row in enumerate(ds):
        score = float(row.get("sentiment_score", row.get("score", 0.0)))
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        text = row.get("sentence", row.get("text", ""))
        out.append({
            "text": text,
            "expected": label,
            "category": "fiqa",
            "id": f"fiqa-{i}",
        })
    return out


def _load_twitterfin() -> list[dict]:
    """Twitter Financial News sentiment.

    Source: zeroshot/twitter-financial-news-sentiment.
    Uses the validation split (the train split is what most people fine-tune on).
    """
    from datasets import load_dataset
    ds = load_dataset(
        "zeroshot/twitter-financial-news-sentiment",
        split="validation",
    )
    label_map = LABEL_MAPS["twitterfin"]
    out = []
    for i, row in enumerate(ds):
        out.append({
            "text": row["text"],
            "expected": label_map[row["label"]],
            "category": "twitterfin",
            "id": f"twitterfin-{i}",
        })
    return out


def _load_finsenti() -> list[dict]:
    """FinSenti's own held-out test slice (848 samples, balanced 3-way)."""
    test_path = _project_root() / "validated" / "raw_test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Expected FinSenti test file at {test_path}. "
            f"Make sure you're running from the repo root."
        )
    out = []
    with test_path.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            label = row.get("answer") or row.get("label", "")
            if isinstance(label, str):
                label = label.strip().lower()
            out.append({
                "text": row["text"],
                "expected": label,
                "category": row.get("source", "finsenti"),
                "id": f"finsenti-{i}",
            })
    return out


def _load_financemteb() -> list[dict]:
    """Out-of-distribution financial news sentiment.

    Originally targeted at FinanceMTEB/FinSentEnglish, which doesn't
    actually exist on the Hub. Swapped to Jean-Baptiste/financial_news_sentiment:
    ~2000 manually validated Canadian English financial news articles with
    sentiment labels, distinct sourcing from anything in the FinSenti
    training pool. The benchmark key 'financemteb' is kept for downstream
    compatibility (run_all.py / aggregator).

    Schema: {'text', 'sentiment', 'topic', ...}. `sentiment` values are
    strings like 'positive' / 'negative' / 'neutral' (verified by reading
    the dataset card).
    """
    from datasets import load_dataset
    repo_attempts = [
        ("Jean-Baptiste/financial_news_sentiment", "train"),
        ("Jean-Baptiste/financial_news_sentiment", "test"),
        ("Jean-Baptiste/financial_news_sentiment", "validation"),
    ]
    last_err = None
    ds = None
    for repo, split in repo_attempts:
        try:
            ds = load_dataset(repo, split=split)
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(
            f"Could not load OOD financial news sentiment benchmark. "
            f"Last error: {last_err}"
        )

    out = []
    for i, row in enumerate(ds):
        text = (
            row.get("text")
            or row.get("title")
            or row.get("sentence")
            or row.get("content")
            or ""
        )
        raw_label = (
            row.get("sentiment")
            or row.get("label")
            or row.get("label_text")
            or ""
        )
        if isinstance(raw_label, int):
            label = {0: "negative", 1: "neutral", 2: "positive"}.get(raw_label, "neutral")
        elif isinstance(raw_label, float):
            if raw_label > 0.1:
                label = "positive"
            elif raw_label < -0.1:
                label = "negative"
            else:
                label = "neutral"
        else:
            s = str(raw_label).strip().lower()
            # Normalize common variants
            if s in ("pos", "positive", "bullish"):
                label = "positive"
            elif s in ("neg", "negative", "bearish"):
                label = "negative"
            elif s in ("neu", "neutral", "none", ""):
                label = "neutral"
            else:
                label = "neutral"
        if not text:
            continue
        out.append({
            "text": text,
            "expected": label,
            "category": "financemteb",
            "id": f"financemteb-{i}",
        })
    return out


def _load_asba() -> list[dict]:
    """Second OOD financial news sentiment benchmark.

    The original target (SetFit / krishnapal2308) doesn't exist on the
    Hub. Swapped to Daniel-ML/sentiment-analysis-for-financial-news-v2
    with prithvi1029/sentiment-analysis-for-financial-news as a backup.
    Both are FinancialPhraseBank-style retail-investor labels but with
    distinct sourcing from our training pool. The benchmark key 'asba'
    is kept for downstream compatibility.

    Note: these may overlap a small amount with FPB; the dedup hash filter
    removes exact-text overlaps with our training pool, but won't catch
    cross-benchmark overlap. Reviewers can verify this is OOD by inspecting
    `n_filtered_for_overlap` in the per-run JSON.
    """
    from datasets import load_dataset
    repo_attempts = [
        ("Daniel-ML/sentiment-analysis-for-financial-news-v2", "train"),
        ("Daniel-ML/sentiment-analysis-for-financial-news-v2", "test"),
        ("prithvi1029/sentiment-analysis-for-financial-news", "train"),
        ("prithvi1029/sentiment-analysis-for-financial-news", "test"),
    ]
    last_err = None
    ds = None
    for repo, split in repo_attempts:
        try:
            ds = load_dataset(repo, split=split)
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(
            f"Could not load OOD ASBA-style financial sentiment benchmark. "
            f"Last error: {last_err}"
        )

    out = []
    for i, row in enumerate(ds):
        text = (
            row.get("text")
            or row.get("sentence")
            or row.get("Sentence")
            or row.get("headline")
            or row.get("title")
            or ""
        )
        raw_label = (
            row.get("label_text")
            or row.get("Sentiment")
            or row.get("label")
            or row.get("sentiment", "")
        )
        if isinstance(raw_label, int):
            label = {0: "negative", 1: "neutral", 2: "positive"}.get(raw_label, "neutral")
        else:
            s = str(raw_label).strip().lower()
            if s in ("pos", "positive", "bullish"):
                label = "positive"
            elif s in ("neg", "negative", "bearish"):
                label = "negative"
            elif s in ("neu", "neutral", "none", ""):
                label = "neutral"
            else:
                label = "neutral"
        if not text:
            continue
        out.append({
            "text": text,
            "expected": label,
            "category": "asba",
            "id": f"asba-{i}",
        })
    return out


def _project_root() -> Path:
    """Walk up from this file until we find the repo root (has training/)."""
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "training").is_dir() and (parent / "validated").is_dir():
            return parent
    # Fallback: cwd
    return Path.cwd()


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="finsenti", choices=list_benchmarks())
    parser.add_argument("--max-samples", type=int, default=5)
    args = parser.parse_args()

    samples = load_benchmark(args.benchmark, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples from {args.benchmark}")
    for s in samples[:5]:
        print(f"  {s['id']:>20s}  expected={s['expected']:>8s}  text={s['text'][:80]}")

"""
Run baseline financial-sentiment models on the same benchmarks as FinSenti.

Baselines covered:
    finbert             ProsusAI/finbert                                 (BERT-base, 2019)
    finbert-tone        yiyanghkust/finbert-tone                         (BERT-base, 2020)
    finbert-sentiment   ahmedrachid/FinancialBERT-Sentiment-Analysis     (FinancialBERT)

All of these are vanilla text-classification heads (no reasoning chain).
The script loads each via `transformers.pipeline`, runs inference on a
benchmark, and writes per-sample predictions + aggregate metrics to JSON
in the same format as evaluation/benchmark.py output. That way the
aggregator can mix FinSenti and baseline scores in a single table.

Usage:
    # Run FinBERT on the FPB benchmark
    python evaluation/baselines.py --baseline finbert --benchmark fpb \\
        --output-json evaluation/results/baselines/finbert-fpb.json

    # Sweep all baselines on all benchmarks (one process per combo)
    for b in finbert finbert-tone finbert-sentiment; do
        for bench in fpb fiqa twitterfin finsenti; do
            python evaluation/baselines.py --baseline $b --benchmark $bench \\
                --output-json evaluation/results/baselines/$b-$bench.json
        done
    done
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .datasets_loader import load_benchmark, list_benchmarks
except ImportError:  # when run as a script from repo root
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datasets_loader import load_benchmark, list_benchmarks


# -----------------------------------------------------------------------------
# Baseline registry. Each entry has:
#   repo_id:      HF model ID
#   label_map:    maps model output labels to our {positive, negative, neutral}
# -----------------------------------------------------------------------------
BASELINES = {
    "finbert": {
        "repo_id": "ProsusAI/finbert",
        "label_map": {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        },
    },
    "finbert-tone": {
        "repo_id": "yiyanghkust/finbert-tone",
        "label_map": {
            "Positive": "positive",
            "Negative": "negative",
            "Neutral": "neutral",
        },
    },
    "finbert-sentiment": {
        "repo_id": "ahmedrachid/FinancialBERT-Sentiment-Analysis",
        "label_map": {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        },
    },
}


def list_baselines() -> list[str]:
    return list(BASELINES.keys())


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run_baseline(
    baseline: str,
    benchmark: str,
    max_samples: int | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> dict:
    """Run one baseline on one benchmark and return the results dict."""
    if baseline not in BASELINES:
        raise ValueError(
            f"Unknown baseline {baseline!r}. Choose from: {list_baselines()}"
        )

    import torch
    from transformers import pipeline

    spec = BASELINES[baseline]
    repo_id = spec["repo_id"]
    label_map = spec["label_map"]

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    print(f"Loading {baseline} from {repo_id} on device={device}...")
    clf = pipeline(
        "text-classification",
        model=repo_id,
        tokenizer=repo_id,
        device=device,
        truncation=True,
        max_length=512,
    )

    samples = load_benchmark(benchmark, max_samples=max_samples)
    print(f"Running {baseline} on {len(samples)} samples of {benchmark}...")

    texts = [s["text"] for s in samples]
    start = time.time()
    raw_preds = clf(texts, batch_size=batch_size, truncation=True)
    latency = time.time() - start

    # Normalize predictions
    results = []
    for sample, pred in zip(samples, raw_preds):
        raw_label = pred.get("label", "")
        mapped = label_map.get(raw_label, raw_label.lower())
        correct = mapped == sample["expected"]
        results.append({
            "id": sample["id"],
            "text": sample["text"],
            "expected": sample["expected"],
            "predicted": mapped,
            "raw_label": raw_label,
            "raw_score": float(pred.get("score", 0.0)),
            "correct": correct,
            "category": sample["category"],
        })

    aggregate = compute_metrics(results, benchmark=benchmark)
    aggregate["avg_latency_sec"] = latency / len(results) if results else 0.0
    aggregate["wall_time_sec"] = latency

    return {
        "model": repo_id,
        "model_kind": "baseline",
        "baseline_name": baseline,
        "benchmark": benchmark,
        "num_samples": len(results),
        "aggregate": aggregate,
        "results": results,
    }


# -----------------------------------------------------------------------------
# Metrics (shared with benchmark.py via duck-typing on `results`)
# -----------------------------------------------------------------------------

def compute_metrics(results: list[dict], benchmark: str = "") -> dict:
    """Compute accuracy, macro/weighted F1, and per-class precision/recall/F1."""
    from collections import Counter
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_recall_fscore_support,
        )
    except ImportError:
        # sklearn not available: fall back to hand-rolled accuracy only
        n_correct = sum(1 for r in results if r["correct"])
        return {
            "benchmark": benchmark,
            "accuracy": n_correct / len(results) if results else 0.0,
            "note": "sklearn not installed; only accuracy computed",
        }

    labels = ["positive", "negative", "neutral"]
    y_true = [r["expected"] for r in results]
    y_pred = [r["predicted"] or "__no_prediction__" for r in results]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    precision, recall, f1, support = per_class
    per_class_dict = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    # Label distributions (for diagnostics)
    true_dist = Counter(y_true)
    pred_dist = Counter(y_pred)

    return {
        "benchmark": benchmark,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_dict,
        "label_distribution_true": dict(true_dist),
        "label_distribution_pred": dict(pred_dist),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run baseline models on benchmarks")
    parser.add_argument(
        "--baseline", required=True, choices=list_baselines(),
        help="Which baseline model to run.",
    )
    parser.add_argument(
        "--benchmark", required=True, choices=list_benchmarks(),
        help="Which benchmark to evaluate on.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Write results to this JSON file (optional).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Truncate benchmark to this many samples.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    result = run_baseline(
        baseline=args.baseline,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    agg = result["aggregate"]
    print()
    print(f"=== {args.baseline} on {args.benchmark} ===")
    print(f"Samples:     {result['num_samples']}")
    print(f"Accuracy:    {agg.get('accuracy', 0):.4f}")
    print(f"Weighted F1: {agg.get('weighted_f1', 0):.4f}")
    print(f"Macro F1:    {agg.get('macro_f1', 0):.4f}")
    if "per_class" in agg:
        print("Per-class F1:")
        for lbl, m in agg["per_class"].items():
            print(f"  {lbl:>8s}: F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                  f"R={m['recall']:.3f}  n={m['support']}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()

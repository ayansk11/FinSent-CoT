"""
Hash-based contamination filter for FinSenti benchmarks.

Why this exists
---------------
The FinSenti training pool was assembled from 6 financial sentiment HF
datasets. Three of the standard public benchmarks we report on (FPB, FiQA,
TwitterFin) overlap with that pool, so naive evaluation would leak training
samples into the test set. This module removes that overlap by re-applying
the EXACT same hash normalization the training pipeline used:

    md5(text.lower().strip().encode()).hexdigest()

(See `data_generation/sources.py::FinancialSample.text_hash` for the
canonical definition.)

Usage from Python
-----------------
    from evaluation.dedup import filter_benchmark
    clean_samples, n_removed = filter_benchmark("fpb")

Usage from CLI
--------------
    python evaluation/dedup.py --benchmark fpb --report
    # FPB: 4840 samples, 0 overlap with training (0.0% removed)
    #   -> 4840 clean samples remain

    python evaluation/dedup.py --all
    # Reports overlap for every benchmark in datasets_loader.list_benchmarks()
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

# Allow `python evaluation/dedup.py` directly from repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from datasets_loader import list_benchmarks, load_benchmark  # noqa: E402


# Files we treat as "training pool" for overlap purposes. Train + val are
# both seen by the model during training. Test is held out by definition,
# so we don't include it (otherwise we'd never report on FinSenti's own
# in-distribution test).
TRAINING_FILES = ("raw_train.jsonl", "raw_val.jsonl")


def normalize(text: str) -> str:
    """Match the normalization in data_generation/sources.py exactly."""
    return text.lower().strip()


def text_hash(text: str) -> str:
    """MD5 hex digest of normalized text. Same as training-side dedup."""
    return hashlib.md5(normalize(text).encode()).hexdigest()


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "training").is_dir() and (parent / "validated").is_dir():
            return parent
    return Path.cwd()


def training_hashes(verbose: bool = False) -> set[str]:
    """Load every text in train+val and return their hash set.

    Reads from validated/raw_train.jsonl + raw_val.jsonl. Each row is
    expected to have a 'text' field (matches the raw format from
    data_generation/sources.py).
    """
    seen: set[str] = set()
    val_dir = _project_root() / "validated"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected validated/ directory at {val_dir}. "
            "Run from repo root or make sure data_generation step ran."
        )

    for fname in TRAINING_FILES:
        path = val_dir / fname
        if not path.exists():
            if verbose:
                print(f"  [warn] {path} not found, skipping")
            continue
        n_loaded = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text", "")
                if text:
                    seen.add(text_hash(text))
                    n_loaded += 1
        if verbose:
            print(f"  loaded {n_loaded} training texts from {fname}")

    return seen


def filter_benchmark(
    name: str,
    max_samples: int | None = None,
    *,
    train_hashes: set[str] | None = None,
) -> tuple[list[dict], int]:
    """Load a benchmark and remove samples that overlap with training.

    Args:
        name: benchmark short name (fpb / fiqa / twitterfin / finsenti / ...).
        max_samples: cap applied AFTER filtering.
        train_hashes: pre-computed set of training hashes (optional, useful
            when filtering many benchmarks back-to-back to avoid re-reading
            train files each time).

    Returns:
        (clean_samples, n_removed) where clean_samples has the same shape as
        load_benchmark() output.
    """
    if train_hashes is None:
        train_hashes = training_hashes()

    raw = load_benchmark(name, max_samples=None)
    clean: list[dict] = []
    n_removed = 0
    for s in raw:
        if text_hash(s["text"]) in train_hashes:
            n_removed += 1
        else:
            clean.append(s)

    if max_samples is not None:
        clean = clean[:max_samples]
    return clean, n_removed


def report(name: str, train_hashes: set[str] | None = None) -> dict:
    """Print and return overlap stats for a single benchmark."""
    if train_hashes is None:
        train_hashes = training_hashes()

    raw = load_benchmark(name, max_samples=None)
    n_total = len(raw)
    n_removed = sum(1 for s in raw if text_hash(s["text"]) in train_hashes)
    n_clean = n_total - n_removed
    pct = (n_removed / n_total * 100) if n_total else 0.0

    line = f"  {name:<14s}: {n_total:>6d} total | {n_removed:>5d} overlap ({pct:5.2f}%) | {n_clean:>6d} clean"
    print(line)
    return {
        "benchmark": name,
        "total": n_total,
        "removed": n_removed,
        "clean": n_clean,
        "overlap_pct": pct,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hash-based contamination filter for FinSenti benchmarks"
    )
    parser.add_argument(
        "--benchmark", choices=list_benchmarks(),
        help="Report overlap for a single benchmark.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Report overlap for every known benchmark.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress informational output (just print rows).",
    )
    args = parser.parse_args()

    if not args.benchmark and not args.all:
        parser.error("Specify either --benchmark <name> or --all")

    if not args.quiet:
        print("Loading training-side hash set...")
    train = training_hashes(verbose=not args.quiet)
    if not args.quiet:
        print(f"  total unique training hashes: {len(train):,}")
        print()
        print(f"  {'benchmark':<14s}: {'total':>6s}        | {'overlap':>5s}          | {'clean':>6s}")
        print(f"  {'-' * 14}--{'-' * 6}--------|{'-' * 5}-----------|{'-' * 6}")

    if args.all:
        for b in list_benchmarks():
            try:
                report(b, train_hashes=train)
            except Exception as e:
                print(f"  {b:<14s}: ERROR loading: {e}")
    else:
        report(args.benchmark, train_hashes=train)

    return 0


if __name__ == "__main__":
    sys.exit(main())

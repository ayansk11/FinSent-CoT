"""
Aggregate per-run JSON files in evaluation/results/ into a single
comparison CSV and Markdown table ranked by weighted F1.

Expected layout under evaluation/results/:
    finsenti/<model_key>/<benchmark>.json   (produced by benchmark.py)
    baselines/<baseline>/<benchmark>.json   (produced by baselines.py)

Each JSON has at minimum:
    {
      "model": str,
      "benchmark": str,
      "aggregate": {
        "accuracy": float, "weighted_f1": float, "macro_f1": float, ...
      }
    }

Usage:
    python evaluation/aggregate.py
    python evaluation/aggregate.py --output-dir evaluation/results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BENCHMARKS = ["fpb", "fiqa", "twitterfin", "finsenti"]


def collect_runs(root: Path) -> list[dict]:
    """Walk the results directory and return a flat list of run records."""
    runs = []
    for bench_json in root.rglob("*.json"):
        if bench_json.name in {"comparison.csv", "comparison.md"}:
            continue
        try:
            with bench_json.open() as f:
                blob = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] could not read {bench_json}: {exc}")
            continue
        agg = blob.get("aggregate") or {}
        runs.append({
            "path": str(bench_json.relative_to(root)),
            "model": blob.get("model", "?"),
            "model_kind": blob.get("model_kind", "?"),
            "benchmark": blob.get("benchmark", "?"),
            "num_samples": blob.get("num_samples", 0),
            "accuracy": agg.get("accuracy", 0.0),
            "weighted_f1": agg.get("weighted_f1", 0.0),
            "macro_f1": agg.get("macro_f1", 0.0),
            "format_compliance": agg.get("format_compliance", None),
            "avg_latency_sec": agg.get("avg_latency_sec", None),
        })
    return runs


def pivot_by_benchmark(runs: list[dict]) -> dict[str, dict[str, float]]:
    """Return {model: {benchmark: weighted_f1}} for easy table rendering."""
    pivot: dict[str, dict[str, float]] = {}
    for r in runs:
        pivot.setdefault(r["model"], {})
        pivot[r["model"]][r["benchmark"]] = r["weighted_f1"]
    return pivot


def write_csv(runs: list[dict], out_path: Path) -> None:
    fields = [
        "model", "model_kind", "benchmark", "num_samples",
        "accuracy", "weighted_f1", "macro_f1",
        "format_compliance", "avg_latency_sec", "path",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            writer.writerow({k: r.get(k, "") for k in fields})


def write_markdown(runs: list[dict], out_path: Path) -> None:
    """Two tables: (1) long form sorted by F1; (2) wide form, model x benchmark."""
    lines = []
    lines.append("# FinSenti evaluation results\n")

    # Wide pivot table
    pivot = pivot_by_benchmark(runs)
    headers = ["Model"] + BENCHMARKS + ["avg F1"]
    lines.append("## Weighted F1 per (model, benchmark)\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    # Sort by the average F1 across the benchmarks the model actually ran on
    def avg_f1(model: str) -> float:
        scores = [v for v in pivot[model].values() if v]
        return sum(scores) / len(scores) if scores else 0.0

    for model in sorted(pivot.keys(), key=lambda m: -avg_f1(m)):
        row = [model]
        for b in BENCHMARKS:
            v = pivot[model].get(b)
            row.append(f"{v:.3f}" if v is not None else "-")
        row.append(f"{avg_f1(model):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Long form (sorted by weighted F1)\n")
    lines.append("| Model | Benchmark | Samples | Accuracy | Weighted F1 | Macro F1 |")
    lines.append("|-------|-----------|---------|----------|-------------|----------|")
    for r in sorted(runs, key=lambda r: -r["weighted_f1"]):
        lines.append(
            f"| {r['model']} | {r['benchmark']} | {r['num_samples']} | "
            f"{r['accuracy']:.3f} | {r['weighted_f1']:.3f} | {r['macro_f1']:.3f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("evaluation/results"),
        help="Directory containing per-run JSON files.",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"No results dir at {args.results_dir}")

    runs = collect_runs(args.results_dir)
    print(f"Found {len(runs)} run JSONs under {args.results_dir}")
    if not runs:
        return

    csv_path = args.results_dir / "comparison.csv"
    md_path = args.results_dir / "comparison.md"
    write_csv(runs, csv_path)
    write_markdown(runs, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

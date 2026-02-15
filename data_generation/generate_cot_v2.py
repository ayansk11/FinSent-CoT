"""
CoT Data Generation v2 — High-quality chain-of-thought financial sentiment data.

Uses Qwen3-235B-A22B-FP8 via vLLM (4x H100 tensor parallel) to generate
structured reasoning for financial text classification.

Features:
- Multi-source dataset (FinGPT + FiQA + PhraseBank)
- Strict XML validation (<reasoning>...</reasoning><answer>...</answer>)
- Checkpoint/resume support
- Parallel batch generation via vLLM
- Automatic quality filtering
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import wandb
from openai import OpenAI

from sources import FinancialSample, load_all_sources

# ─── Constants ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial sentiment analyst. Your task is to analyze financial text and classify its sentiment.

For each text, provide:
1. Your step-by-step reasoning in <reasoning> tags
2. Your final sentiment classification in <answer> tags

Your reasoning MUST:
- Identify specific financial indicators (revenue, earnings, stock price, etc.)
- Analyze the tone and language used (positive/negative/neutral words)
- Consider market implications
- Be at least 3 sentences long

Always use this EXACT format:
<reasoning>
[Your detailed step-by-step analysis here]
</reasoning>
<answer>[positive/negative/neutral]</answer>"""

USER_TEMPLATE = """Analyze the sentiment of the following financial text:

"{text}"

Provide your reasoning and classification."""


# ─── Validation ──────────────────────────────────────────────────────────────

REASONING_PATTERN = re.compile(
    r"<reasoning>\s*(.+?)\s*</reasoning>", re.DOTALL | re.IGNORECASE
)
ANSWER_PATTERN = re.compile(
    r"<answer>\s*(positive|negative|neutral)\s*</answer>", re.IGNORECASE
)

# Financial terms for quality checking
FINANCIAL_TERMS = {
    # Performance
    "revenue", "earnings", "profit", "loss", "growth", "decline", "margin",
    "income", "sales", "ebitda", "cash flow", "operating",
    # Market
    "stock", "share", "price", "market", "valuation", "cap", "volume",
    "traded", "listed", "ipo", "index",
    # Actions
    "beat", "miss", "exceed", "surpass", "fell", "rose", "gained", "lost",
    "surged", "plunged", "dropped", "rallied", "tumbled",
    # Entities
    "analyst", "investor", "shareholder", "ceo", "company", "firm",
    "bank", "fund", "portfolio",
    # Metrics
    "eps", "pe ratio", "dividend", "yield", "roi", "guidance",
    "forecast", "outlook", "target", "estimate", "consensus",
    # Sentiment
    "positive", "negative", "neutral", "bullish", "bearish",
    "optimistic", "pessimistic", "concern", "confidence",
    # Economic
    "rate", "inflation", "gdp", "employment", "fed", "monetary",
    "fiscal", "economic", "recession", "expansion",
}


def validate_response(response: str, expected_label: str) -> dict:
    """
    Validate a model response for format compliance and quality.

    Returns:
        dict with keys: valid, reasoning, answer, issues
    """
    issues = []

    # Check for reasoning tags
    reasoning_match = REASONING_PATTERN.search(response)
    if not reasoning_match:
        issues.append("missing_reasoning_tags")
        reasoning_text = ""
    else:
        reasoning_text = reasoning_match.group(1).strip()

    # Check for answer tags
    answer_match = ANSWER_PATTERN.search(response)
    if not answer_match:
        issues.append("missing_answer_tags")
        answer_text = ""
    else:
        answer_text = answer_match.group(1).strip().lower()

    # Check reasoning quality
    if reasoning_text:
        word_count = len(reasoning_text.split())
        if word_count < 20:
            issues.append(f"reasoning_too_short ({word_count} words)")

        # Check for financial terms
        reasoning_lower = reasoning_text.lower()
        found_terms = sum(1 for t in FINANCIAL_TERMS if t in reasoning_lower)
        if found_terms < 2:
            issues.append(f"low_financial_relevance ({found_terms} terms)")
    else:
        issues.append("empty_reasoning")

    # Check answer correctness
    if answer_text and answer_text != expected_label:
        issues.append(f"wrong_label (got={answer_text}, expected={expected_label})")

    valid = len(issues) == 0
    return {
        "valid": valid,
        "reasoning": reasoning_text,
        "answer": answer_text,
        "issues": issues,
    }


# ─── Generation ──────────────────────────────────────────────────────────────


def generate_cot_batch(
    client: OpenAI,
    model: str,
    samples: list[FinancialSample],
    max_retries: int = 2,
    temperature: float = 0.4,
) -> list[dict]:
    """
    Generate CoT for a batch of samples.
    Retries failed samples up to max_retries times.
    """
    results = []

    for sample in samples:
        response_text = None
        for attempt in range(max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": USER_TEMPLATE.format(text=sample.text),
                        },
                    ],
                    max_tokens=512,
                    temperature=temperature + (attempt * 0.1),  # Slightly raise temp on retry
                    top_p=0.9,
                )
                response_text = completion.choices[0].message.content.strip()

                # Strip <think>...</think> blocks from Qwen3
                response_text = re.sub(
                    r"<think>.*?</think>", "", response_text, flags=re.DOTALL
                ).strip()

                validation = validate_response(response_text, sample.label)

                if validation["valid"]:
                    results.append(
                        {
                            "text": sample.text,
                            "label": sample.label,
                            "source": sample.source,
                            "reasoning": validation["reasoning"],
                            "answer": validation["answer"],
                            "response_raw": response_text,
                            "attempt": attempt + 1,
                        }
                    )
                    break
                elif attempt == max_retries:
                    # Keep it but flag as imperfect
                    results.append(
                        {
                            "text": sample.text,
                            "label": sample.label,
                            "source": sample.source,
                            "reasoning": validation["reasoning"] or "",
                            "answer": validation["answer"] or sample.label,
                            "response_raw": response_text or "",
                            "attempt": attempt + 1,
                            "issues": validation["issues"],
                        }
                    )

            except Exception as e:
                if attempt == max_retries:
                    print(f"  ERROR: Failed after {max_retries + 1} attempts: {e}")
                    results.append(
                        {
                            "text": sample.text,
                            "label": sample.label,
                            "source": sample.source,
                            "reasoning": "",
                            "answer": sample.label,
                            "response_raw": "",
                            "attempt": attempt + 1,
                            "issues": [f"api_error: {str(e)}"],
                        }
                    )
                else:
                    time.sleep(2)

    return results


# ─── Checkpoint Management ───────────────────────────────────────────────────


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        self.output_file = self.checkpoint_dir / "generated_cot.jsonl"

    def load_progress(self) -> int:
        """Return the number of already-generated samples."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
            return data.get("completed", 0)
        return 0

    def save_progress(self, completed: int, stats: dict):
        with open(self.checkpoint_file, "w") as f:
            json.dump({"completed": completed, "stats": stats}, f, indent=2)

    def append_results(self, results: list[dict]):
        with open(self.output_file, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate CoT financial sentiment data")
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-FP8",
                        help="vLLM model name")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="vLLM API base URL")
    parser.add_argument("--target-samples", type=int, default=50000,
                        help="Target number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Samples per batch")
    parser.add_argument("--checkpoint-dir", default="./checkpoints",
                        help="Directory for checkpoints and output")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("FinSent-CoT Data Generation v2")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Target: {args.target_samples} samples")
    print(f"Batch size: {args.batch_size}")
    print()

    # ─── Initialize W&B ─────────────────────────────────────────────────────
    wandb.init(
        project="FinSent-CoT",
        name=f"datagen-{args.model.split('/')[-1]}",
        tags=["data-generation", "cot"],
        config={
            "generator_model": args.model,
            "target_samples": args.target_samples,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "seed": args.seed,
            "phase": "data_generation",
        },
    )

    # Initialize
    client = OpenAI(base_url=args.api_base, api_key="not-needed")
    ckpt = CheckpointManager(args.checkpoint_dir)

    # Load source data
    print("Loading source datasets...")
    samples = load_all_sources(
        target_total=args.target_samples,
        balance=True,
        seed=args.seed,
    )

    # Log source distribution to W&B
    source_counts = {}
    label_counts = {}
    for s in samples:
        source_counts[s.source] = source_counts.get(s.source, 0) + 1
        label_counts[s.label] = label_counts.get(s.label, 0) + 1
    wandb.log({
        "source_distribution": wandb.Table(
            columns=["source", "count"],
            data=[[k, v] for k, v in source_counts.items()],
        ),
        "label_distribution": wandb.Table(
            columns=["label", "count"],
            data=[[k, v] for k, v in label_counts.items()],
        ),
        "total_source_samples": len(samples),
    })

    # Resume from checkpoint
    start_idx = ckpt.load_progress()
    if start_idx > 0:
        print(f"\nResuming from checkpoint: {start_idx}/{len(samples)} completed")
    else:
        print(f"\nStarting fresh generation: {len(samples)} samples")

    # Stats tracking
    stats = {"total": 0, "valid": 0, "retried": 0, "failed": 0}
    # Per-label tracking for W&B
    label_stats = {
        "positive": {"total": 0, "valid": 0},
        "negative": {"total": 0, "valid": 0},
        "neutral": {"total": 0, "valid": 0},
    }
    reasoning_lengths = []

    # Generate in batches
    total = len(samples)
    start_time = time.time()

    for batch_start in range(start_idx, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch = samples[batch_start:batch_end]

        batch_num = (batch_start // args.batch_size) + 1
        total_batches = (total + args.batch_size - 1) // args.batch_size
        print(f"\n[Batch {batch_num}/{total_batches}] Generating {len(batch)} samples...")

        results = generate_cot_batch(
            client=client,
            model=args.model,
            samples=batch,
            temperature=args.temperature,
        )

        # Update stats
        batch_valid = 0
        for r in results:
            stats["total"] += 1
            label = r.get("label", "unknown")
            if label in label_stats:
                label_stats[label]["total"] += 1

            if not r.get("issues"):
                stats["valid"] += 1
                batch_valid += 1
                if label in label_stats:
                    label_stats[label]["valid"] += 1
            elif r.get("attempt", 1) > 1:
                stats["retried"] += 1
            if r.get("reasoning") == "":
                stats["failed"] += 1

            # Track reasoning length
            reasoning = r.get("reasoning", "")
            if reasoning:
                reasoning_lengths.append(len(reasoning.split()))

        # Save
        ckpt.append_results(results)
        ckpt.save_progress(batch_end, stats)

        # Progress report
        elapsed = time.time() - start_time
        samples_done = batch_end - start_idx
        rate = samples_done / elapsed if elapsed > 0 else 0
        eta = (total - batch_end) / rate if rate > 0 else 0
        valid_pct = (stats["valid"] / stats["total"] * 100) if stats["total"] > 0 else 0

        print(f"  Progress: {batch_end}/{total} ({batch_end/total*100:.1f}%)")
        print(f"  Valid: {stats['valid']}/{stats['total']} ({valid_pct:.1f}%)")
        print(f"  Rate: {rate:.1f} samples/sec | ETA: {eta/3600:.1f}h")

        # ─── Log batch metrics to W&B ───────────────────────────────────────
        wandb.log({
            "batch": batch_num,
            "progress": batch_end / total,
            "samples_completed": batch_end,
            "samples_total": total,
            # Cumulative quality metrics
            "valid_total": stats["valid"],
            "valid_rate": valid_pct,
            "failed_total": stats["failed"],
            "retried_total": stats["retried"],
            # Batch-level metrics
            "batch_valid_rate": batch_valid / len(results) * 100 if results else 0,
            # Speed
            "generation_rate_samples_per_sec": rate,
            "eta_hours": eta / 3600,
            # Per-label accuracy
            **{
                f"valid_rate_{label}": (
                    ls["valid"] / ls["total"] * 100 if ls["total"] > 0 else 0
                )
                for label, ls in label_stats.items()
            },
            # Reasoning quality
            "avg_reasoning_length": (
                sum(reasoning_lengths[-len(results):]) / len(results)
                if results else 0
            ),
        })

    # ─── Final report ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total samples: {stats['total']}")
    print(f"Valid (perfect format): {stats['valid']} ({stats['valid']/max(stats['total'],1)*100:.1f}%)")
    print(f"Retried: {stats['retried']}")
    print(f"Failed (no reasoning): {stats['failed']}")
    print(f"Time: {elapsed/3600:.1f} hours")
    print(f"Output: {ckpt.output_file}")

    # ─── Log final summary to W&B ───────────────────────────────────────────
    wandb.summary.update({
        "final_total": stats["total"],
        "final_valid": stats["valid"],
        "final_valid_rate": stats["valid"] / max(stats["total"], 1) * 100,
        "final_failed": stats["failed"],
        "final_retried": stats["retried"],
        "total_time_hours": elapsed / 3600,
        "avg_reasoning_word_count": (
            sum(reasoning_lengths) / len(reasoning_lengths)
            if reasoning_lengths else 0
        ),
    })

    # Log reasoning length distribution as histogram
    if reasoning_lengths:
        wandb.log({
            "reasoning_length_distribution": wandb.Histogram(reasoning_lengths),
        })

    # Log sample examples table
    sample_table = wandb.Table(
        columns=["text", "label", "reasoning_preview", "answer", "valid"]
    )
    output_file = ckpt.output_file
    if output_file.exists():
        with open(output_file) as f:
            for i, line in enumerate(f):
                if i >= 50:  # Log first 50 examples
                    break
                r = json.loads(line)
                sample_table.add_data(
                    r.get("text", "")[:200],
                    r.get("label", ""),
                    r.get("reasoning", "")[:300],
                    r.get("answer", ""),
                    "issues" not in r,
                )
        wandb.log({"sample_examples": sample_table})

    wandb.finish()


if __name__ == "__main__":
    main()

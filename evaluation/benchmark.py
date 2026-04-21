"""
Comprehensive model evaluation & W&B reporting for FinSenti.

Evaluates a fine-tuned model (or any HF chat/completion model) on the
held-out test set or one of the public finance benchmarks, and logs
detailed metrics to W&B plus an optional JSON result file for downstream
aggregation.

Usage:
    # Evaluate a FinSenti HF model on the FPB benchmark
    python evaluation/benchmark.py \\
        --backend transformers --model Ayansk11/FinSenti-Qwen3-0.6B \\
        --benchmark fpb \\
        --output-json evaluation/results/qwen3-0.6b/fpb.json

    # Evaluate via Ollama (local Mac M4 / cluster post-export)
    python evaluation/benchmark.py --backend ollama --model finsenti-qwen3-0.6b \\
        --benchmark finsenti

    # Evaluate via vLLM (OpenAI-compatible server)
    python evaluation/benchmark.py --backend vllm \\
        --model ./checkpoints/grpo --api-base http://localhost:8000/v1 \\
        --benchmark fpb

    # Use the original 10 curated smoke-test cases (no benchmark flag)
    python evaluation/benchmark.py --backend ollama --model finsenti-qwen3-0.6b
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import wandb

# Allow running either as a module (python -m evaluation.benchmark) or
# as a script (python evaluation/benchmark.py) with rewards / datasets
# imports working in both cases.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from datasets_loader import load_benchmark, list_benchmarks  # noqa: E402
from baselines import compute_metrics  # noqa: E402

# ─── Patterns ────────────────────────────────────────────────────────────────

REASONING_PATTERN = re.compile(
    r"<reasoning>\s*(.+?)\s*</reasoning>", re.DOTALL | re.IGNORECASE
)
ANSWER_PATTERN = re.compile(
    r"<answer>\s*(positive|negative|neutral)\s*</answer>", re.IGNORECASE
)

FINANCIAL_TERMS = {
    "revenue", "earnings", "profit", "loss", "growth", "decline", "margin",
    "income", "sales", "stock", "share", "price", "market", "analyst",
    "investor", "dividend", "guidance", "forecast", "outlook",
}

# ─── Test Cases ──────────────────────────────────────────────────────────────

DEFAULT_TEST_CASES = [
    {
        "text": "Apple reported record Q4 earnings of $89.5 billion in revenue, beating analyst expectations by 3%. iPhone sales surged 12% year-over-year.",
        "expected": "positive",
        "category": "strong_positive",
    },
    {
        "text": "Tesla shares plunged 15% after the company missed delivery targets by a wide margin. Multiple analysts downgraded the stock to sell.",
        "expected": "negative",
        "category": "strong_negative",
    },
    {
        "text": "The Federal Reserve held interest rates steady at 5.25-5.50%, as widely expected by markets. Officials said they would continue monitoring incoming economic data.",
        "expected": "neutral",
        "category": "neutral_policy",
    },
    {
        "text": "Goldman Sachs announced it will lay off 3,000 employees as part of a restructuring plan aimed at cutting costs by $1.2 billion annually.",
        "expected": "negative",
        "category": "layoffs",
    },
    {
        "text": "Amazon Web Services revenue growth decelerated to 12% in Q3, down from 16% the previous quarter. The company announced a $10 billion investment in AI infrastructure.",
        "expected": "neutral",
        "category": "mixed_signals",
    },
    {
        "text": "Microsoft reported cloud revenue of $31.8 billion, up 22% year-over-year. Azure growth accelerated to 33%, ahead of Wall Street estimates.",
        "expected": "positive",
        "category": "cloud_growth",
    },
    {
        "text": "Nvidia shares fell 5% in after-hours trading despite reporting revenue of $22.1 billion, as investors expressed concerns about future demand sustainability.",
        "expected": "negative",
        "category": "beats_but_sells_off",
    },
    {
        "text": "JPMorgan Chase maintained its quarterly dividend at $1.15 per share and reaffirmed its full-year guidance. The bank noted stable credit conditions.",
        "expected": "neutral",
        "category": "dividend_stable",
    },
    {
        "text": "Meta Platforms announced a 20% increase in advertising revenue and raised its full-year guidance. Mark Zuckerberg said AI investments are paying off.",
        "expected": "positive",
        "category": "ad_revenue_growth",
    },
    {
        "text": "Boeing reported a wider-than-expected quarterly loss of $1.64 per share. The company faces ongoing delivery delays and quality control issues.",
        "expected": "negative",
        "category": "wider_loss",
    },
]


# ─── Backends ────────────────────────────────────────────────────────────────

def query_ollama(model: str, text: str) -> str:
    """Query model via Ollama CLI."""
    import subprocess
    result = subprocess.run(
        ["ollama", "run", model, f"Analyze the following financial text: {text}"],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout.strip()


SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. Analyze the given financial text "
    "and provide:\n"
    "1. Your reasoning in <reasoning> tags\n"
    "2. Your sentiment classification (positive, negative, or neutral) in "
    "<answer> tags"
)


def query_vllm(client, model: str, text: str) -> str:
    """Query model via vLLM OpenAI-compatible API."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze the following financial text: {text}"},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    response = completion.choices[0].message.content.strip()
    # Strip <think> blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response


# -----------------------------------------------------------------------------
# transformers backend: load the HF model directly (no vLLM / Ollama server).
# Useful for evaluating SafeTensors repos on a GPU node without the cost of
# starting a separate serving process per model.
# -----------------------------------------------------------------------------

_TRANSFORMERS_STATE: dict = {}


def _load_transformers(model_id: str):
    """Load an HF causal-LM once and cache for subsequent calls in this run."""
    if _TRANSFORMERS_STATE.get("loaded_model") == model_id:
        return (
            _TRANSFORMERS_STATE["tokenizer"],
            _TRANSFORMERS_STATE["model"],
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading transformers model {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _TRANSFORMERS_STATE.update({
        "loaded_model": model_id,
        "tokenizer": tok,
        "model": model,
    })
    return tok, model


def query_transformers(model_id: str, text: str, max_new_tokens: int = 512) -> str:
    """Generate a response with a locally loaded HF model."""
    import torch

    tok, model = _load_transformers(model_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze the following financial text: {text}"},
    ]
    try:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Model has no chat template - fall back to a plain concat
        prompt = f"{SYSTEM_PROMPT}\n\nText: {text}\n\nResponse:"

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen = tok.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    # Strip <think> blocks if any model emits them
    gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL).strip()
    return gen


# ─── Evaluation Logic ────────────────────────────────────────────────────────

def evaluate_response(response: str, expected: str) -> dict:
    """Evaluate a single model response."""
    reasoning_match = REASONING_PATTERN.search(response)
    answer_match = ANSWER_PATTERN.search(response)

    has_reasoning_tags = reasoning_match is not None
    has_answer_tags = answer_match is not None

    reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
    predicted = answer_match.group(1).strip().lower() if answer_match else ""

    # If no answer tags, try to extract from plain text
    if not predicted:
        for label in ["positive", "negative", "neutral"]:
            if f"<answer>{label}" in response.lower():
                predicted = label
                break

    correct = predicted == expected.lower()

    # Reasoning quality metrics
    reasoning_words = len(reasoning_text.split()) if reasoning_text else 0
    reasoning_lower = reasoning_text.lower()
    financial_terms_found = sum(1 for t in FINANCIAL_TERMS if t in reasoning_lower)
    sentences = [s.strip() for s in reasoning_text.split(".") if len(s.strip()) > 10] if reasoning_text else []

    return {
        "predicted": predicted,
        "expected": expected.lower(),
        "correct": correct,
        "has_reasoning_tags": has_reasoning_tags,
        "has_answer_tags": has_answer_tags,
        "reasoning_text": reasoning_text,
        "reasoning_word_count": reasoning_words,
        "financial_terms_found": financial_terms_found,
        "num_sentences": len(sentences),
        "response_raw": response,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FinSenti model")
    parser.add_argument(
        "--backend",
        choices=["ollama", "vllm", "transformers"],
        default="transformers",
        help="ollama: local CLI. vllm: OpenAI-compatible server. "
             "transformers: load HF model directly with torch.",
    )
    parser.add_argument("--model", default="Ayansk11/FinSenti-Qwen3-0.6B",
                        help="Model name (Ollama), HF repo / path (vLLM/transformers).")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="vLLM API base URL")
    parser.add_argument(
        "--benchmark",
        choices=list_benchmarks(),
        default=None,
        help="Run on a public benchmark loader instead of a JSONL test file.",
    )
    parser.add_argument("--test-file", default=None,
                        help="Path to test JSONL file (raw format). "
                             "Ignored if --benchmark is given.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Truncate the benchmark / test file to this many "
                             "samples. Useful for smoke tests.")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Write per-sample predictions and aggregate "
                             "metrics to this JSON file.")
    parser.add_argument("--run-name", default=None,
                        help="W&B run name (auto-generated if not set)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip W&B logging entirely (useful for batch runs).")
    args = parser.parse_args()

    # Load test cases
    if args.benchmark:
        print(f"Loading benchmark {args.benchmark}...")
        bench_samples = load_benchmark(args.benchmark, max_samples=args.max_samples)
        test_cases = [
            {
                "text": s["text"],
                "expected": s["expected"],
                "category": s["category"],
                "id": s["id"],
            }
            for s in bench_samples
        ]
        print(f"Loaded {len(test_cases)} samples from {args.benchmark}")
    elif args.test_file and Path(args.test_file).exists():
        print(f"Loading test cases from {args.test_file}...")
        test_cases = []
        with open(args.test_file) as f:
            for line in f:
                r = json.loads(line)
                test_cases.append({
                    "text": r["text"],
                    "expected": r.get("answer", r.get("label", "")),
                    "category": r.get("source", "test_set"),
                    "id": r.get("id", f"line-{len(test_cases)}"),
                })
        if args.max_samples:
            test_cases = test_cases[:args.max_samples]
        print(f"Loaded {len(test_cases)} test cases")
    else:
        test_cases = [
            {**tc, "id": f"default-{i}"} for i, tc in enumerate(DEFAULT_TEST_CASES)
        ]
        if args.max_samples:
            test_cases = test_cases[:args.max_samples]
        print(f"Using {len(test_cases)} default test cases")

    # ─── Initialize W&B ─────────────────────────────────────────────────────
    run_name = (
        args.run_name
        or f"eval-{args.backend}-{args.model.split('/')[-1]}"
        + (f"-{args.benchmark}" if args.benchmark else "")
    )
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "FinSenti"),
                name=run_name,
                tags=["evaluation", "benchmark", args.backend,
                      args.benchmark or "smoke"],
                config={
                    "phase": "evaluation",
                    "backend": args.backend,
                    "model": args.model,
                    "benchmark": args.benchmark,
                    "num_test_cases": len(test_cases),
                },
            )
        except Exception as e:
            print(f"[warn] W&B init failed ({e}); continuing without W&B")
            use_wandb = False

    # Initialize backend
    client = None
    if args.backend == "vllm":
        from openai import OpenAI
        client = OpenAI(base_url=args.api_base, api_key="not-needed")

    # ─── Run evaluation ──────────────────────────────────────────────────────
    results = []
    print(f"\nRunning evaluation ({args.backend}: {args.model})...")
    print("-" * 70)

    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['text'][:80]}...")
        start = time.time()

        try:
            if args.backend == "ollama":
                response = query_ollama(args.model, tc["text"])
            elif args.backend == "vllm":
                response = query_vllm(client, args.model, tc["text"])
            else:
                response = query_transformers(args.model, tc["text"])

            latency = time.time() - start
            eval_result = evaluate_response(response, tc["expected"])
            eval_result["latency"] = latency
            eval_result["category"] = tc.get("category", "unknown")
            eval_result["id"] = tc.get("id", f"case-{i}")
            eval_result["text"] = tc["text"]
            results.append(eval_result)

            status = "CORRECT" if eval_result["correct"] else "WRONG"
            tags = "YES" if eval_result["has_reasoning_tags"] else "NO"
            print(f"  {status} | predicted={eval_result['predicted']} | "
                  f"tags={tags} | {latency:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": tc.get("id", f"case-{i}"),
                "text": tc["text"],
                "predicted": "",
                "expected": tc["expected"],
                "correct": False,
                "has_reasoning_tags": False,
                "has_answer_tags": False,
                "reasoning_text": "",
                "reasoning_word_count": 0,
                "financial_terms_found": 0,
                "num_sentences": 0,
                "response_raw": f"ERROR: {e}",
                "latency": time.time() - start,
                "category": tc.get("category", "unknown"),
            })

    # ─── Compute aggregate metrics ───────────────────────────────────────────
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100 if total > 0 else 0

    has_reasoning = sum(1 for r in results if r["has_reasoning_tags"])
    has_answer = sum(1 for r in results if r["has_answer_tags"])
    format_compliance = has_reasoning / total * 100 if total > 0 else 0

    avg_reasoning_words = (
        sum(r["reasoning_word_count"] for r in results) / total if total > 0 else 0
    )
    avg_financial_terms = (
        sum(r["financial_terms_found"] for r in results) / total if total > 0 else 0
    )
    avg_latency = sum(r["latency"] for r in results) / total if total > 0 else 0

    # Per-label accuracy
    label_correct = Counter()
    label_total = Counter()
    for r in results:
        label_total[r["expected"]] += 1
        if r["correct"]:
            label_correct[r["expected"]] += 1

    # sklearn-based F1 + per-class precision/recall (shared with baselines)
    sklearn_metrics = compute_metrics(results, benchmark=args.benchmark or "smoke")

    # ─── Print summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy:         {accuracy:.1f}% ({correct}/{total})")
    if "weighted_f1" in sklearn_metrics:
        print(f"Weighted F1:      {sklearn_metrics['weighted_f1']*100:.1f}%")
        print(f"Macro F1:         {sklearn_metrics['macro_f1']*100:.1f}%")
    print(f"Format Compliance: {format_compliance:.1f}% ({has_reasoning}/{total} with <reasoning> tags)")
    print(f"Answer Tags:      {has_answer}/{total}")
    print(f"Avg Reasoning:    {avg_reasoning_words:.0f} words")
    print(f"Avg Fin. Terms:   {avg_financial_terms:.1f} per response")
    print(f"Avg Latency:      {avg_latency:.2f}s")
    print()
    for label in ["positive", "negative", "neutral"]:
        t = label_total.get(label, 0)
        c = label_correct.get(label, 0)
        pct = c / t * 100 if t > 0 else 0
        print(f"  {label:>8s}: {pct:.1f}% ({c}/{t})")

    # ─── Write JSON output (if requested) ────────────────────────────────────
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        json_blob = {
            "model": args.model,
            "model_kind": "finsenti" if "FinSenti" in args.model else "other",
            "backend": args.backend,
            "benchmark": args.benchmark or ("test-file" if args.test_file else "smoke"),
            "num_samples": total,
            "aggregate": {
                **sklearn_metrics,
                "format_compliance": format_compliance / 100.0,
                "answer_tag_rate": (has_answer / total) if total else 0.0,
                "avg_reasoning_words": avg_reasoning_words,
                "avg_financial_terms": avg_financial_terms,
                "avg_latency_sec": avg_latency,
            },
            "results": results,
        }
        with args.output_json.open("w") as f:
            json.dump(json_blob, f, indent=2)
        print(f"\nWrote results to {args.output_json}")

    # ─── Log everything to W&B ───────────────────────────────────────────────
    if not use_wandb:
        print("\n(skipped W&B logging)")
        return

    # 1. Summary metrics
    wandb.summary.update({
        "accuracy": accuracy,
        "weighted_f1": sklearn_metrics.get("weighted_f1", 0) * 100,
        "macro_f1": sklearn_metrics.get("macro_f1", 0) * 100,
        "format_compliance": format_compliance,
        "answer_tag_rate": has_answer / total * 100 if total > 0 else 0,
        "avg_reasoning_words": avg_reasoning_words,
        "avg_financial_terms": avg_financial_terms,
        "avg_latency_sec": avg_latency,
        "total_test_cases": total,
        **{
            f"accuracy_{label}": (
                label_correct.get(label, 0) / label_total.get(label, 1) * 100
            )
            for label in ["positive", "negative", "neutral"]
        },
    })

    # 2. Detailed results table
    results_table = wandb.Table(columns=[
        "text", "expected", "predicted", "correct", "has_reasoning_tags",
        "reasoning_preview", "reasoning_words", "financial_terms",
        "latency_sec", "category",
    ])
    for i, r in enumerate(results):
        text_preview = test_cases[i]["text"][:200] if i < len(test_cases) else ""
        results_table.add_data(
            text_preview,
            r["expected"],
            r["predicted"],
            r["correct"],
            r["has_reasoning_tags"],
            r["reasoning_text"][:300],
            r["reasoning_word_count"],
            r["financial_terms_found"],
            r["latency"],
            r["category"],
        )
    wandb.log({"evaluation_results": results_table})

    # 3. Confusion matrix (as a table since wandb.plot.confusion_matrix needs exact class matching)
    labels = ["positive", "negative", "neutral"]
    cm_data = []
    for true_label in labels:
        for pred_label in labels:
            count = sum(
                1 for r in results
                if r["expected"] == true_label and r["predicted"] == pred_label
            )
            cm_data.append([true_label, pred_label, count])
    # Also count no_answer predictions
    no_answer_count = sum(1 for r in results if not r["predicted"])
    if no_answer_count > 0:
        for true_label in labels:
            count = sum(
                1 for r in results
                if r["expected"] == true_label and not r["predicted"]
            )
            cm_data.append([true_label, "no_answer", count])

    wandb.log({
        "confusion_matrix": wandb.Table(
            columns=["true_label", "predicted_label", "count"],
            data=cm_data,
        ),
    })

    # 4. Per-label accuracy bar chart
    wandb.log({
        "per_label_accuracy": wandb.Table(
            columns=["label", "accuracy", "correct", "total"],
            data=[
                [
                    label,
                    label_correct.get(label, 0) / label_total.get(label, 1) * 100,
                    label_correct.get(label, 0),
                    label_total.get(label, 0),
                ]
                for label in ["positive", "negative", "neutral"]
            ],
        ),
    })

    # 5. Quality metrics distributions
    wandb.log({
        "reasoning_length_distribution": wandb.Histogram(
            [r["reasoning_word_count"] for r in results]
        ),
        "latency_distribution": wandb.Histogram(
            [r["latency"] for r in results]
        ),
        "financial_terms_distribution": wandb.Histogram(
            [r["financial_terms_found"] for r in results]
        ),
    })

    # 6. Format compliance breakdown
    format_table = wandb.Table(
        columns=["metric", "count", "percentage"],
        data=[
            ["Has <reasoning> tags", has_reasoning, format_compliance],
            ["Has <answer> tags", has_answer, has_answer / total * 100],
            ["Both tags present", sum(1 for r in results if r["has_reasoning_tags"] and r["has_answer_tags"]),
             sum(1 for r in results if r["has_reasoning_tags"] and r["has_answer_tags"]) / total * 100],
            ["No tags at all", sum(1 for r in results if not r["has_reasoning_tags"] and not r["has_answer_tags"]),
             sum(1 for r in results if not r["has_reasoning_tags"] and not r["has_answer_tags"]) / total * 100],
        ],
    )
    wandb.log({"format_compliance_breakdown": format_table})

    run_url = wandb.run.url if wandb.run else "N/A"
    wandb.finish()
    print(f"\nW&B run complete! View at: {run_url}")


if __name__ == "__main__":
    main()

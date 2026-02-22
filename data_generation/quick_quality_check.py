"""Quick quality analysis of the generated dataset."""

import json
import sys
from collections import Counter
from pathlib import Path


def analyze(path: str):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    total = len(samples)
    print(f"{'='*70}")
    print(f"DATASET QUALITY REPORT: {path}")
    print(f"{'='*70}")
    print(f"Total samples: {total}")

    # ─── Label distribution ──────────────────────────────────────────────
    labels = Counter(s.get("label", "unknown") for s in samples)
    print(f"\nLabel distribution:")
    for label, count in labels.most_common():
        print(f"  {label}: {count} ({count/total*100:.1f}%)")

    # ─── Source distribution ─────────────────────────────────────────────
    sources = Counter(s.get("source", "unknown") for s in samples)
    print(f"\nSource distribution:")
    for source, count in sources.most_common():
        print(f"  {source}: {count} ({count/total*100:.1f}%)")

    # ─── Issues breakdown ────────────────────────────────────────────────
    has_issues = sum(1 for s in samples if s.get("issues"))
    no_issues = total - has_issues
    print(f"\nQuality:")
    print(f"  Perfect (no issues): {no_issues} ({no_issues/total*100:.1f}%)")
    print(f"  Has issues: {has_issues} ({has_issues/total*100:.1f}%)")

    if has_issues > 0:
        all_issues = []
        for s in samples:
            if s.get("issues"):
                all_issues.extend(s["issues"])
        issue_counts = Counter(all_issues)
        print(f"\n  Issue breakdown:")
        for issue, count in issue_counts.most_common():
            print(f"    {issue}: {count}")

    # ─── Reasoning quality ───────────────────────────────────────────────
    reasoning_lengths = []
    empty_reasoning = 0
    for s in samples:
        r = s.get("reasoning", "")
        if not r:
            empty_reasoning += 1
        else:
            reasoning_lengths.append(len(r.split()))

    if reasoning_lengths:
        avg = sum(reasoning_lengths) / len(reasoning_lengths)
        min_len = min(reasoning_lengths)
        max_len = max(reasoning_lengths)
        # Percentiles
        sorted_lens = sorted(reasoning_lengths)
        p10 = sorted_lens[int(len(sorted_lens) * 0.10)]
        p25 = sorted_lens[int(len(sorted_lens) * 0.25)]
        p50 = sorted_lens[int(len(sorted_lens) * 0.50)]
        p75 = sorted_lens[int(len(sorted_lens) * 0.75)]
        p90 = sorted_lens[int(len(sorted_lens) * 0.90)]

        print(f"\nReasoning length (words):")
        print(f"  Mean: {avg:.1f}")
        print(f"  Min: {min_len}, Max: {max_len}")
        print(f"  P10: {p10}, P25: {p25}, P50: {p50}, P75: {p75}, P90: {p90}")
        print(f"  Empty reasoning: {empty_reasoning}")

    # ─── Answer accuracy ─────────────────────────────────────────────────
    correct = sum(1 for s in samples if s.get("answer", "").lower() == s.get("label", "").lower())
    print(f"\nAnswer matches label: {correct}/{total} ({correct/total*100:.1f}%)")

    # ─── Retry stats ─────────────────────────────────────────────────────
    attempts = Counter(s.get("attempt", 1) for s in samples)
    print(f"\nAttempt distribution:")
    for attempt, count in sorted(attempts.items()):
        print(f"  Attempt {attempt}: {count} ({count/total*100:.1f}%)")

    # ─── Deduplication check ─────────────────────────────────────────────
    import hashlib
    texts = set()
    dupes = 0
    for s in samples:
        h = hashlib.md5(s.get("text", "").lower().strip().encode()).hexdigest()
        if h in texts:
            dupes += 1
        texts.add(h)
    print(f"\nDuplicate texts: {dupes} ({dupes/total*100:.1f}%)")

    # ─── Sample examples ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SAMPLE EXAMPLES (3 random)")
    print(f"{'='*70}")
    import random
    random.seed(42)
    examples = random.sample(samples, min(3, total))
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1} ---")
        print(f"  Text: {ex.get('text', '')[:200]}")
        print(f"  Label: {ex.get('label', '')}")
        print(f"  Answer: {ex.get('answer', '')}")
        print(f"  Reasoning (first 300 chars): {ex.get('reasoning', '')[:300]}")
        print(f"  Issues: {ex.get('issues', 'None')}")
        print(f"  Attempt: {ex.get('attempt', '?')}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/datagen/generated_cot.jsonl"
    analyze(path)

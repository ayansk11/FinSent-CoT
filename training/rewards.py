"""
Improved reward functions for GRPO training.

4 equal-weight rewards (1.0 each):
1. Sentiment correctness — Is the classification right?
2. Format compliance — Are <reasoning> and <answer> tags present and well-formed?
3. Reasoning quality — Is the reasoning detailed, relevant, and analytical?
4. Consistency — Does the reasoning logically support the answer?
"""

import re

# ─── Patterns ────────────────────────────────────────────────────────────────

REASONING_PATTERN = re.compile(
    r"<reasoning>\s*(.+?)\s*</reasoning>", re.DOTALL | re.IGNORECASE
)
ANSWER_PATTERN = re.compile(
    r"<answer>\s*(positive|negative|neutral)\s*</answer>", re.IGNORECASE
)

# Expanded financial vocabulary for quality checking
POSITIVE_INDICATORS = {
    "beat", "exceeded", "surpass", "surged", "gained", "rose", "rallied",
    "record", "strong", "growth", "profit", "upgrade", "bullish", "optimistic",
    "outperformed", "increased", "improved", "momentum", "breakthrough",
    "robust", "soared", "expanded", "recovery", "upbeat",
}

NEGATIVE_INDICATORS = {
    "fell", "dropped", "plunged", "declined", "lost", "missed", "downgrade",
    "bearish", "concern", "risk", "loss", "cut", "layoff", "weak",
    "tumbled", "slumped", "deteriorated", "warning", "pessimistic",
    "recession", "default", "bankruptcy", "lawsuit", "investigation",
}

NEUTRAL_INDICATORS = {
    "maintained", "steady", "unchanged", "flat", "mixed", "monitoring",
    "expected", "announced", "reported", "stated", "released", "filed",
    "scheduled", "planned", "proposed", "considered", "evaluated",
}

FINANCIAL_TERMS = {
    "revenue", "earnings", "profit", "loss", "growth", "decline", "margin",
    "income", "sales", "ebitda", "cash flow", "operating", "stock", "share",
    "price", "market", "valuation", "analyst", "investor", "dividend",
    "eps", "guidance", "forecast", "outlook", "target", "estimate",
    "rate", "inflation", "gdp", "fed", "monetary", "fiscal",
    "quarter", "annual", "year-over-year", "yoy", "qoq",
}


# ─── Reward 1: Sentiment Correctness ────────────────────────────────────────

def sentiment_correctness_reward(prompts, completions, **kwargs) -> list[float]:
    """
    +1.0 for correct sentiment, -0.5 for wrong, -1.0 for no answer.
    """
    answers = kwargs.get("answer", [])
    rewards = []

    for completion, expected in zip(completions, answers):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        match = ANSWER_PATTERN.search(text)
        if match:
            predicted = match.group(1).strip().lower()
            expected_label = expected.strip().lower()
            if predicted == expected_label:
                rewards.append(1.0)
            else:
                rewards.append(-0.5)
        else:
            rewards.append(-1.0)

    return rewards


# ─── Reward 2: Format Compliance ────────────────────────────────────────────

def format_compliance_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Checks for proper XML structure:
    +0.4 for <reasoning> tags present with content
    +0.3 for <answer> tags present with valid label
    +0.3 for correct ordering (reasoning before answer)
    -1.0 for no tags at all
    """
    rewards = []

    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0

        reasoning_match = REASONING_PATTERN.search(text)
        answer_match = ANSWER_PATTERN.search(text)

        if reasoning_match and len(reasoning_match.group(1).strip()) > 10:
            score += 0.4
        if answer_match:
            score += 0.3
        if reasoning_match and answer_match:
            if reasoning_match.start() < answer_match.start():
                score += 0.3

        if score == 0.0:
            score = -1.0

        rewards.append(score)

    return rewards


# ─── Reward 3: Reasoning Quality ────────────────────────────────────────────

def reasoning_quality_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Evaluates the depth and financial relevance of reasoning:
    - Word count (min 30, bonus for 50+)
    - Financial term usage (at least 3)
    - Sentiment indicator usage
    - Multi-sentence structure
    - Penalize copy-paste from input
    """
    rewards = []

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        reasoning_match = REASONING_PATTERN.search(text)

        if not reasoning_match:
            rewards.append(-0.5)
            continue

        reasoning = reasoning_match.group(1).strip()
        reasoning_lower = reasoning.lower()
        words = reasoning.split()
        word_count = len(words)

        score = 0.0

        # Length scoring
        if word_count >= 30:
            score += 0.25
        if word_count >= 50:
            score += 0.15
        if word_count < 15:
            score -= 0.3

        # Financial term relevance
        found_financial = sum(1 for t in FINANCIAL_TERMS if t in reasoning_lower)
        if found_financial >= 3:
            score += 0.25
        elif found_financial >= 1:
            score += 0.1
        else:
            score -= 0.2

        # Sentiment indicator alignment
        pos_count = sum(1 for t in POSITIVE_INDICATORS if t in reasoning_lower)
        neg_count = sum(1 for t in NEGATIVE_INDICATORS if t in reasoning_lower)
        neu_count = sum(1 for t in NEUTRAL_INDICATORS if t in reasoning_lower)
        if pos_count + neg_count + neu_count >= 2:
            score += 0.15

        # Multi-sentence bonus
        sentences = [s.strip() for s in reasoning.split(".") if len(s.strip()) > 10]
        if len(sentences) >= 3:
            score += 0.2
        elif len(sentences) >= 2:
            score += 0.1

        # Penalize if reasoning is just copying the input
        prompt_text = prompts[i] if i < len(prompts) else ""
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[-1]["content"] if prompt_text else ""
        prompt_words = set(prompt_text.lower().split())
        reasoning_words = set(reasoning_lower.split())
        if len(reasoning_words) > 0 and len(prompt_words) > 0:
            overlap = len(prompt_words & reasoning_words) / len(reasoning_words)
            if overlap > 0.7:
                score -= 0.3

        rewards.append(max(min(score, 1.0), -1.0))

    return rewards


# ─── Reward 4: Consistency ───────────────────────────────────────────────────

def consistency_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Checks if the reasoning logically supports the answer.
    - Positive answer should have more positive indicators in reasoning
    - Negative answer should have more negative indicators
    - Penalize contradictions
    """
    rewards = []

    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion

        reasoning_match = REASONING_PATTERN.search(text)
        answer_match = ANSWER_PATTERN.search(text)

        if not reasoning_match or not answer_match:
            rewards.append(0.0)
            continue

        reasoning_lower = reasoning_match.group(1).strip().lower()
        answer = answer_match.group(1).strip().lower()

        pos_count = sum(1 for t in POSITIVE_INDICATORS if t in reasoning_lower)
        neg_count = sum(1 for t in NEGATIVE_INDICATORS if t in reasoning_lower)

        score = 0.0

        if answer == "positive":
            if pos_count > neg_count:
                score = 1.0
            elif pos_count == neg_count:
                score = 0.3
            else:
                score = -0.5  # Reasoning contradicts answer

        elif answer == "negative":
            if neg_count > pos_count:
                score = 1.0
            elif neg_count == pos_count:
                score = 0.3
            else:
                score = -0.5

        elif answer == "neutral":
            # For neutral, we want balanced or few sentiment indicators
            total_sentiment = pos_count + neg_count
            if total_sentiment <= 2:
                score = 1.0
            elif abs(pos_count - neg_count) <= 1:
                score = 0.7  # Balanced = good for neutral
            else:
                score = 0.0

        rewards.append(score)

    return rewards

# ─────────────────────────────────────────────────────────────────────────────
# Module: hallucination_classifier.py
# Purpose: Zero-shot Hallucination Classification
#
# Technique: Use a zero-shot classifier (HuggingFace) or a keyword-based
# trained scorer to classify text as hallucinated vs. factual.
# Inspired by TruthfulQA and HaluEval benchmarks.
# ─────────────────────────────────────────────────────────────────────────────

import re
import numpy as np
from typing import Optional

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ── Known Hallucination Patterns (from HaluEval / TruthfulQA research) ────────

HALLUCINATION_TRIGGERS = {
    "fabricated_citations": [
        r"according to (a )?study",
        r"researchers (found|showed|discovered|reported)",
        r"a \d+\d+ study",
        r"published in [A-Z][a-z]+ (Journal|Review|Magazine)",
        r"experts (say|claim|agree|believe)",
    ],
    "false_certainty": [
        r"it is (a fact|well-known|established|proven) that",
        r"definitely",
        r"certainly",
        r"without (a )?doubt",
        r"absolutely",
        r"100%",
    ],
    "vague_authority": [
        r"many (people|experts|scientists|researchers) (think|believe|say)",
        r"some studies",
        r"it has been shown",
        r"it is widely (believed|accepted|known)",
    ],
    "self_contradiction_markers": [
        r"however.*previously",
        r"on the other hand.*earlier",
        r"but.*stated above",
    ],
    "overconfident_numbers": [
        r"\b\d{1,3}(\.\d+)?%\b",                    # Specific percentages
        r"\$[\d,]+(\.\d+)?\s*(million|billion)?",    # Dollar amounts
        r"\b\d+\s*(million|billion|trillion)\b",     # Large numbers
    ],
}

FACTUAL_SIGNALS = [
    r"\bi (don'?t|do not) know\b",
    r"\bi'?m (not sure|uncertain|unsure)\b",
    r"\bthis (may|might|could) (vary|differ|depend)\b",
    r"\bplease (verify|check|confirm)\b",
    r"\baccording to (the|this) (text|document|context|passage)\b",
    r"\bbased on (the|this) (information|context|data) (provided|given|above)\b",
]


def classify_with_heuristics(text: str) -> dict:
    """
    Rule-based hallucination classifier using linguistic patterns.
    Works without any ML model — instant results.

    Args:
        text: LLM response text to classify

    Returns:
        dict with category scores, overall score, and verdict
    """
    text_lower = text.lower()
    category_scores = {}
    all_triggers_found = []

    # Check hallucination trigger categories
    for category, patterns in HALLUCINATION_TRIGGERS.items():
        found = []
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            found.extend(matches)
        category_scores[category] = len(found)
        if found:
            all_triggers_found.append(f"{category}: {len(found)} instances")

    # Check factual signals (these REDUCE risk)
    factual_signal_count = 0
    for pattern in FACTUAL_SIGNALS:
        matches = re.findall(pattern, text_lower)
        factual_signal_count += len(matches)

    # ── Compute composite hallucination score ──────────────────────────────────
    # Weighted category scores
    weights = {
        "fabricated_citations": 3.0,
        "false_certainty": 1.5,
        "vague_authority": 2.0,
        "self_contradiction_markers": 2.5,
        "overconfident_numbers": 1.0,
    }
    raw_score = sum(category_scores[cat] * weights[cat] for cat in category_scores)
    normalized = min(raw_score / 15.0, 1.0)  # cap at 1.0

    # Factual signals reduce risk
    normalized = max(0.0, normalized - (factual_signal_count * 0.08))

    if normalized >= 0.6:
        risk = "High"
        color = "red"
        verdict = "🚨 High hallucination indicators detected"
        label = "LIKELY_HALLUCINATED"
    elif normalized >= 0.3:
        risk = "Medium"
        color = "orange"
        verdict = "⚠️ Some hallucination indicators found"
        label = "UNCERTAIN"
    else:
        risk = "Low"
        color = "green"
        verdict = "✅ Few hallucination indicators"
        label = "LIKELY_FACTUAL"

    return {
        "method": "Rule-Based Heuristic Classifier",
        "hallucination_score": round(normalized, 4),
        "hallucination_label": label,
        "hallucination_risk": risk,
        "risk_color": color,
        "verdict": verdict,
        "category_scores": category_scores,
        "triggers_found": all_triggers_found,
        "factual_signals_found": factual_signal_count,
        "raw_score": round(raw_score, 2),
    }


def classify_with_zero_shot(
    text: str,
    model_name: str = "facebook/bart-large-mnli",
) -> dict:
    """
    Use a zero-shot NLI classifier to detect hallucination.

    Labels: ["factual and accurate", "contains hallucinations", "uncertain or unverifiable"]

    Args:
        text: LLM response to classify
        model_name: Zero-shot classification model

    Returns:
        dict with classification results and confidence
    """
    if not TRANSFORMERS_AVAILABLE:
        return {
            "method": "Zero-Shot Classifier",
            "error": "transformers not installed",
            "verdict": "Install with: pip install transformers torch",
        }

    try:
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,  # CPU
        )

        candidate_labels = [
            "factual and accurate information",
            "contains hallucinations or false information",
            "uncertain or unverifiable claims",
        ]

        result = classifier(
            text[:512],  # truncate to 512 chars
            candidate_labels=candidate_labels,
        )

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        if "hallucination" in top_label:
            risk = "High"
            color = "red"
            verdict = f"🚨 Classified as: {top_label} ({top_score:.0%} confidence)"
        elif "uncertain" in top_label:
            risk = "Medium"
            color = "orange"
            verdict = f"⚠️ Classified as: {top_label} ({top_score:.0%} confidence)"
        else:
            risk = "Low"
            color = "green"
            verdict = f"✅ Classified as: {top_label} ({top_score:.0%} confidence)"

        hallucination_idx = result["labels"].index("contains hallucinations or false information")
        hallucination_score = result["scores"][hallucination_idx]

        return {
            "method": f"Zero-Shot NLI ({model_name})",
            "top_label": top_label,
            "top_confidence": round(top_score, 4),
            "hallucination_score": round(hallucination_score, 4),
            "all_labels": result["labels"],
            "all_scores": [round(s, 4) for s in result["scores"]],
            "hallucination_risk": risk,
            "risk_color": color,
            "verdict": verdict,
        }

    except Exception as e:
        return {
            "method": "Zero-Shot Classifier",
            "error": str(e),
            "verdict": f"Error: {e}",
        }


def run_full_classifier(text: str, use_zero_shot: bool = False) -> dict:
    """
    Run the complete hallucination classifier pipeline.

    Args:
        text: LLM response to classify
        use_zero_shot: Whether to also run zero-shot NLI (requires transformers)

    Returns:
        dict with heuristic and optional NLI results
    """
    heuristic_result = classify_with_heuristics(text)
    results = {"heuristic": heuristic_result}

    if use_zero_shot and TRANSFORMERS_AVAILABLE:
        zero_shot_result = classify_with_zero_shot(text)
        results["zero_shot"] = zero_shot_result

        # Combine scores
        h_score = heuristic_result["hallucination_score"]
        z_score = zero_shot_result.get("hallucination_score", h_score)
        combined = h_score * 0.4 + z_score * 0.6
        results["combined_hallucination_score"] = round(combined, 4)
    else:
        results["combined_hallucination_score"] = heuristic_result["hallucination_score"]

    combined = results["combined_hallucination_score"]
    if combined >= 0.6:
        results["final_risk"] = "High"
        results["final_color"] = "red"
        results["final_verdict"] = "🚨 Classifier: HIGH hallucination risk"
    elif combined >= 0.3:
        results["final_risk"] = "Medium"
        results["final_color"] = "orange"
        results["final_verdict"] = "⚠️ Classifier: MEDIUM hallucination risk"
    else:
        results["final_risk"] = "Low"
        results["final_color"] = "green"
        results["final_verdict"] = "✅ Classifier: LOW hallucination risk"

    return results

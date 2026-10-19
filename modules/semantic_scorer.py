# ─────────────────────────────────────────────────────────────────────────────
# Module: semantic_scorer.py
# Purpose: Semantic Similarity / NLI Scoring for Hallucination Detection
#
# Technique: Compare LLM response against a reference answer using:
#   1. Cosine similarity (semantic embedding-based)
#   2. ROUGE scores (n-gram overlap)
#   3. Hallucination-specific heuristics (hedging phrases, specific claims)
# ─────────────────────────────────────────────────────────────────────────────

import re
import numpy as np
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# ── Heuristic Hallucination Signals ──────────────────────────────────────────

HEDGING_PHRASES = [
    "i think", "i believe", "i'm not sure", "i am not sure", "i'm not certain",
    "probably", "possibly", "perhaps", "it might be", "it could be",
    "as far as i know", "to the best of my knowledge", "i'm not 100%",
    "approximately", "around", "roughly", "i may be wrong", "don't quote me",
    "not entirely sure", "i recall", "if i recall correctly",
]

HALLUCINATION_INDICATORS = [
    r"\b\d{4}\b",                  # Years (specific dates can be fabricated)
    r"\b\d+\s*(million|billion|trillion)\b",  # Large specific numbers
    r"\baccording to\b",           # Citing sources that may not exist
    r"\bstudies show\b",
    r"\bresearch (shows|suggests|indicates|found)\b",
    r"\bexperts (say|claim|believe)\b",
    r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Proper names (people often fabricated)
]


def detect_heuristic_signals(text: str) -> dict:
    """
    Detect linguistic signals that correlate with hallucination.

    Returns:
        dict with detected signals and a heuristic risk score
    """
    text_lower = text.lower()
    detected_hedges = [p for p in HEDGING_PHRASES if p in text_lower]
    detected_indicators = []
    for pattern in HALLUCINATION_INDICATORS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            detected_indicators.extend(matches[:3])  # cap at 3 examples each

    # Heuristic risk: more hedges + specific claims = higher uncertainty
    hedge_score = min(len(detected_hedges) / 5.0, 1.0)  # normalize to [0,1]
    indicator_score = min(len(detected_indicators) / 10.0, 1.0)
    heuristic_risk_score = (hedge_score * 0.4 + indicator_score * 0.6)

    if heuristic_risk_score >= 0.6:
        risk_level = "High"
        color = "red"
    elif heuristic_risk_score >= 0.3:
        risk_level = "Medium"
        color = "orange"
    else:
        risk_level = "Low"
        color = "green"

    return {
        "hedging_phrases_found": detected_hedges,
        "specific_claim_patterns": list(set(detected_indicators)),
        "hedge_count": len(detected_hedges),
        "indicator_count": len(detected_indicators),
        "heuristic_risk_score": round(heuristic_risk_score, 4),
        "risk_level": risk_level,
        "risk_color": color,
    }


# ── Semantic Similarity Scoring ───────────────────────────────────────────────

def compute_semantic_similarity(
    response: str,
    reference: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Compute semantic similarity between response and a reference answer.

    Args:
        response: LLM-generated response
        reference: Ground-truth reference answer
        model_name: SBERT model name

    Returns:
        dict with similarity scores and verdict
    """
    result = {
        "method": "Semantic Similarity",
        "semantic_similarity": 0.0,
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "hallucination_risk": "Unknown",
        "risk_color": "gray",
        "verdict": "",
        "explanation": "",
    }

    # ── SBERT Cosine Similarity ────────────────────────────────────────────────
    if SBERT_AVAILABLE:
        try:
            model = SentenceTransformer(model_name)
            emb1 = model.encode(response, convert_to_tensor=True)
            emb2 = model.encode(reference, convert_to_tensor=True)
            sim = float(util.cos_sim(emb1, emb2))
            result["semantic_similarity"] = round(sim, 4)
            result["sbert_model"] = model_name
        except Exception as e:
            result["sbert_error"] = str(e)
    else:
        # Fallback: normalized word overlap
        words_r = set(response.lower().split())
        words_ref = set(reference.lower().split())
        union = len(words_r | words_ref)
        sim = len(words_r & words_ref) / union if union > 0 else 0.0
        result["semantic_similarity"] = round(sim, 4)
        result["sbert_model"] = "Jaccard (fallback)"

    # ── ROUGE Scores ──────────────────────────────────────────────────────────
    if ROUGE_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(reference, response)
            result["rouge1"] = round(scores["rouge1"].fmeasure, 4)
            result["rouge2"] = round(scores["rouge2"].fmeasure, 4)
            result["rougeL"] = round(scores["rougeL"].fmeasure, 4)
        except Exception:
            pass

    # ── Combined Score & Risk Classification ──────────────────────────────────
    combined = result["semantic_similarity"] * 0.6 + result["rougeL"] * 0.4
    result["combined_score"] = round(combined, 4)

    if combined >= 0.65:
        result["hallucination_risk"] = "Low"
        result["risk_color"] = "green"
        result["verdict"] = "✅ Response closely matches the reference answer"
        result["explanation"] = (
            f"Semantic similarity = {result['semantic_similarity']:.2f}, "
            f"ROUGE-L = {result['rougeL']:.2f}. The response is well-aligned with the reference."
        )
    elif combined >= 0.35:
        result["hallucination_risk"] = "Medium"
        result["risk_color"] = "orange"
        result["verdict"] = "⚠️ Partial match — some divergence from reference"
        result["explanation"] = (
            f"Semantic similarity = {result['semantic_similarity']:.2f}, "
            f"ROUGE-L = {result['rougeL']:.2f}. The response partially matches but has notable differences."
        )
    else:
        result["hallucination_risk"] = "High"
        result["risk_color"] = "red"
        result["verdict"] = "🚨 Low similarity — response may be hallucinated"
        result["explanation"] = (
            f"Semantic similarity = {result['semantic_similarity']:.2f}, "
            f"ROUGE-L = {result['rougeL']:.2f}. The response significantly diverges from the reference."
        )

    return result


# ── Composite Hallucination Score ─────────────────────────────────────────────

def compute_composite_score(
    semantic_score: float,
    consistency_score: float,
    grounding_score: float,
    weights: tuple = (0.35, 0.30, 0.35),
) -> dict:
    """
    Compute a weighted composite hallucination risk score from all three methods.

    Args:
        semantic_score: 0-1 score from semantic similarity (1 = match)
        consistency_score: 0-1 score from self-consistency (1 = consistent)
        grounding_score: 0-1 score from RAG grounding (1 = well-grounded)
        weights: (w_semantic, w_consistency, w_grounding) that sum to 1

    Returns:
        dict with composite_score and overall risk level
    """
    w_s, w_c, w_g = weights
    composite = w_s * semantic_score + w_c * consistency_score + w_g * grounding_score

    if composite >= 0.70:
        risk = "Low"
        color = "green"
        verdict = "✅ Overall: LOW hallucination risk"
        badge = "🟢"
    elif composite >= 0.45:
        risk = "Medium"
        color = "orange"
        verdict = "⚠️ Overall: MEDIUM hallucination risk — verify key claims"
        badge = "🟡"
    else:
        risk = "High"
        color = "red"
        verdict = "🚨 Overall: HIGH hallucination risk — treat with caution"
        badge = "🔴"

    return {
        "composite_score": round(composite, 4),
        "hallucination_risk": risk,
        "risk_color": color,
        "verdict": verdict,
        "badge": badge,
        "breakdown": {
            "semantic_similarity": round(semantic_score, 4),
            "self_consistency": round(consistency_score, 4),
            "rag_grounding": round(grounding_score, 4),
        },
        "weights": {
            "semantic": w_s,
            "consistency": w_c,
            "grounding": w_g,
        },
    }

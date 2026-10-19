# ─────────────────────────────────────────────────────────────────────────────
# Module: self_consistency.py
# Purpose: Detect hallucinations via Self-Consistency Checking
#
# Technique: Ask the same question multiple times with temperature > 0.
# If the model's answers are consistent → likely factual.
# If they diverge a lot → possible hallucination.
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
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Lightweight text overlap fallback (no model needed) ───────────────────────

def _jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts (word-level)."""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _pairwise_similarity_jaccard(responses: list[str]) -> np.ndarray:
    """Return symmetric pairwise Jaccard similarity matrix."""
    n = len(responses)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = _jaccard_similarity(responses[i], responses[j])
    return matrix


def _pairwise_similarity_sbert(responses: list[str], model) -> np.ndarray:
    """Return symmetric pairwise cosine similarity using sentence-transformers."""
    embeddings = model.encode(responses, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).numpy()
    return sim_matrix


# ── Main Self-Consistency Analysis ────────────────────────────────────────────

def analyze_self_consistency(
    responses: list[str],
    use_sbert: bool = True,
    sbert_model_name: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Analyze a list of LLM responses for self-consistency.

    Steps:
    1. Compute pairwise semantic similarity between all responses.
    2. Calculate mean similarity (consistency score).
    3. Classify hallucination risk based on threshold.

    Args:
        responses: List of responses to the same prompt
        use_sbert: Use sentence-transformers for better embeddings
        sbert_model_name: SBERT model to use

    Returns:
        dict with keys:
            - consistency_score (float, 0-1)
            - hallucination_risk (str: Low / Medium / High)
            - risk_color (str: green / orange / red)
            - similarity_matrix (np.ndarray)
            - method_used (str)
            - explanation (str)
            - verdict (str)
    """
    if len(responses) < 2:
        return {
            "consistency_score": 1.0,
            "hallucination_risk": "Unknown",
            "risk_color": "gray",
            "similarity_matrix": np.array([[1.0]]),
            "method_used": "N/A",
            "explanation": "Need at least 2 responses to analyze consistency.",
            "verdict": "Insufficient data",
        }

    method_used = "Jaccard Word Overlap (fallback)"
    sim_matrix = _pairwise_similarity_jaccard(responses)

    if use_sbert and SBERT_AVAILABLE:
        try:
            model = SentenceTransformer(sbert_model_name)
            sim_matrix = _pairwise_similarity_sbert(responses, model)
            method_used = f"Sentence-BERT ({sbert_model_name})"
        except Exception:
            pass  # Fall back to Jaccard

    # Extract upper triangle (exclude diagonal)
    n = len(responses)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(sim_matrix[i][j])

    consistency_score = float(np.mean(similarities)) if similarities else 0.0

    # ── Classify risk ──────────────────────────────────────────────────────────
    if consistency_score >= 0.75:
        risk = "Low"
        color = "green"
        explanation = (
            f"The {n} responses are highly consistent (avg. similarity = {consistency_score:.2f}). "
            "This suggests the model is confident and the answer is likely factual."
        )
        verdict = "✅ Likely factual — responses converge"
    elif consistency_score >= 0.45:
        risk = "Medium"
        color = "orange"
        explanation = (
            f"The {n} responses show moderate variation (avg. similarity = {consistency_score:.2f}). "
            "The model may be uncertain. Verify key claims with external sources."
        )
        verdict = "⚠️ Uncertain — responses partially diverge"
    else:
        risk = "High"
        color = "red"
        explanation = (
            f"The {n} responses are highly inconsistent (avg. similarity = {consistency_score:.2f}). "
            "The model is likely hallucinating — different runs produce very different answers."
        )
        verdict = "🚨 Likely hallucination — responses diverge significantly"

    return {
        "consistency_score": round(consistency_score, 4),
        "hallucination_risk": risk,
        "risk_color": color,
        "similarity_matrix": sim_matrix,
        "method_used": method_used,
        "explanation": explanation,
        "verdict": verdict,
        "num_responses": n,
        "pairwise_scores": similarities,
    }

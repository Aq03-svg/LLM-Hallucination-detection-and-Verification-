# ─────────────────────────────────────────────────────────────────────────────
# Module: rag_verifier.py
# Purpose: RAG-Based Hallucination Verification
#
# Technique: Ground LLM responses against a provided knowledge base (context).
# Use NLI (Natural Language Inference) or semantic similarity to check if
# each sentence in the response is supported by the context.
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
    from transformers import pipeline
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False


# ── Sentence Splitter ─────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ── Semantic Similarity Verification (SBERT) ──────────────────────────────────

def verify_with_sbert(
    response: str,
    context: str,
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.45,
) -> dict:
    """
    Verify response against context using semantic similarity.

    For each sentence in the response, find the most similar sentence
    in the context and check if similarity exceeds threshold.

    Args:
        response: LLM-generated response to verify
        context: Ground-truth knowledge base text
        model_name: SBERT model name
        threshold: Min cosine similarity to consider a claim "supported"

    Returns:
        dict with per-sentence verdicts and overall grounding score
    """
    response_sentences = split_into_sentences(response)
    context_sentences = split_into_sentences(context)

    if not response_sentences:
        return {"error": "No sentences found in response"}
    if not context_sentences:
        return {"error": "No sentences found in context"}

    results = {
        "method": f"Semantic Similarity (SBERT: {model_name})",
        "threshold": threshold,
        "sentence_results": [],
        "grounding_score": 0.0,
        "hallucination_risk": "Unknown",
        "risk_color": "gray",
        "verdict": "",
        "explanation": "",
    }

    if not SBERT_AVAILABLE:
        # Fallback: Jaccard similarity
        results["method"] = "Jaccard Word Overlap (fallback — install sentence-transformers)"
        for sent in response_sentences:
            scores = []
            for ctx_sent in context_sentences:
                words_r = set(sent.lower().split())
                words_c = set(ctx_sent.lower().split())
                union = len(words_r | words_c)
                score = len(words_r & words_c) / union if union > 0 else 0.0
                scores.append(score)
            best_score = max(scores) if scores else 0.0
            best_ctx = context_sentences[scores.index(best_score)] if scores else ""
            supported = best_score >= threshold
            results["sentence_results"].append({
                "sentence": sent,
                "best_similarity": round(best_score, 3),
                "best_matching_context": best_ctx,
                "supported": supported,
                "label": "✅ Supported" if supported else "🚨 Unsupported",
            })
    else:
        model = SentenceTransformer(model_name)
        response_embeddings = model.encode(response_sentences, convert_to_tensor=True)
        context_embeddings = model.encode(context_sentences, convert_to_tensor=True)
        sim_matrix = util.cos_sim(response_embeddings, context_embeddings).numpy()

        for i, sent in enumerate(response_sentences):
            best_j = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i][best_j])
            supported = best_score >= threshold
            results["sentence_results"].append({
                "sentence": sent,
                "best_similarity": round(best_score, 3),
                "best_matching_context": context_sentences[best_j],
                "supported": supported,
                "label": "✅ Supported" if supported else "🚨 Unsupported",
            })

    # Calculate overall grounding score
    n = len(results["sentence_results"])
    supported_count = sum(1 for r in results["sentence_results"] if r["supported"])
    grounding_score = supported_count / n if n > 0 else 0.0
    results["grounding_score"] = round(grounding_score, 4)
    results["supported_count"] = supported_count
    results["total_sentences"] = n

    # Risk classification
    if grounding_score >= 0.75:
        results["hallucination_risk"] = "Low"
        results["risk_color"] = "green"
        results["verdict"] = "✅ Well-grounded — most claims are supported by context"
        results["explanation"] = (
            f"{supported_count}/{n} sentences ({grounding_score:.0%}) are semantically "
            f"supported by the provided knowledge base."
        )
    elif grounding_score >= 0.45:
        results["hallucination_risk"] = "Medium"
        results["risk_color"] = "orange"
        results["verdict"] = "⚠️ Partially grounded — some claims may be hallucinated"
        results["explanation"] = (
            f"Only {supported_count}/{n} sentences ({grounding_score:.0%}) are supported. "
            "Review the unsupported sentences carefully."
        )
    else:
        results["hallucination_risk"] = "High"
        results["risk_color"] = "red"
        results["verdict"] = "🚨 Poorly grounded — most claims are NOT supported by context"
        results["explanation"] = (
            f"Only {supported_count}/{n} sentences ({grounding_score:.0%}) are supported. "
            "The response likely contains significant hallucinations."
        )

    return results


# ── NLI-Based Verification ────────────────────────────────────────────────────

def verify_with_nli(
    response: str,
    context: str,
    nli_model: str = "cross-encoder/nli-deberta-v3-small",
) -> dict:
    """
    Verify response against context using Natural Language Inference (NLI).

    NLI classifies each (premise=context, hypothesis=response_sentence) pair as:
    - ENTAILMENT   → context supports the claim  (grounded)
    - NEUTRAL      → context neither supports nor contradicts
    - CONTRADICTION → context contradicts the claim (hallucination detected)

    Args:
        response: LLM-generated response
        context: Ground truth context
        nli_model: HuggingFace NLI model name

    Returns:
        dict with NLI labels per sentence and overall verdict
    """
    response_sentences = split_into_sentences(response)

    results = {
        "method": f"Natural Language Inference ({nli_model})",
        "sentence_results": [],
        "grounding_score": 0.0,
        "hallucination_risk": "Unknown",
        "risk_color": "gray",
        "verdict": "",
        "explanation": "",
        "nli_available": NLI_AVAILABLE,
    }

    if not NLI_AVAILABLE:
        results["verdict"] = "⚠️ NLI unavailable — install transformers"
        results["explanation"] = "Install with: pip install transformers"
        return results

    try:
        nli_pipe = pipeline("zero-shot-classification", model=nli_model, device=-1)
    except Exception as e:
        results["verdict"] = f"[ERROR] Could not load NLI model: {e}"
        return results

    label_map = {
        "entailment": ("✅ Entailed", "green", 1.0),
        "neutral": ("⚪ Neutral", "gray", 0.5),
        "contradiction": ("🚨 Contradiction", "red", 0.0),
    }

    scores = []
    for sent in response_sentences:
        try:
            out = nli_pipe(sent, candidate_labels=["entailment", "neutral", "contradiction"],
                           hypothesis_template="{}",
                           multi_label=False)
            top_label = out["labels"][0].lower()
            top_score = out["scores"][0]
            label_text, color, numeric = label_map.get(top_label, ("Unknown", "gray", 0.5))
            scores.append(numeric)
            results["sentence_results"].append({
                "sentence": sent,
                "nli_label": label_text,
                "confidence": round(top_score, 3),
                "color": color,
                "raw_labels": out["labels"],
                "raw_scores": [round(s, 3) for s in out["scores"]],
            })
        except Exception as e:
            results["sentence_results"].append({
                "sentence": sent,
                "nli_label": f"ERROR: {e}",
                "confidence": 0.0,
                "color": "gray",
            })

    grounding_score = float(np.mean(scores)) if scores else 0.0
    results["grounding_score"] = round(grounding_score, 4)

    if grounding_score >= 0.75:
        results["hallucination_risk"] = "Low"
        results["risk_color"] = "green"
        results["verdict"] = "✅ Context strongly entails the response"
    elif grounding_score >= 0.45:
        results["hallucination_risk"] = "Medium"
        results["risk_color"] = "orange"
        results["verdict"] = "⚠️ Mixed entailment — some claims may be hallucinated"
    else:
        results["hallucination_risk"] = "High"
        results["risk_color"] = "red"
        results["verdict"] = "🚨 Context contradicts or does not support many claims"

    contradiction_count = sum(
        1 for r in results["sentence_results"] if "Contradiction" in r.get("nli_label", "")
    )
    entailed_count = sum(
        1 for r in results["sentence_results"] if "Entailed" in r.get("nli_label", "")
    )
    results["contradiction_count"] = contradiction_count
    results["entailed_count"] = entailed_count
    results["total_sentences"] = len(response_sentences)
    results["explanation"] = (
        f"NLI Analysis: {entailed_count} entailed, "
        f"{len(response_sentences) - entailed_count - contradiction_count} neutral, "
        f"{contradiction_count} contradicted out of {len(response_sentences)} sentences."
    )

    return results

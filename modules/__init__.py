# ─────────────────────────────────────────────────────────────────────────────
# Module: __init__.py
# Purpose: Make 'modules' a proper Python package
# ─────────────────────────────────────────────────────────────────────────────

from .llm_client import query_ollama, query_ollama_multiple, check_ollama_running, get_ollama_models
from .self_consistency import analyze_self_consistency
from .rag_verifier import verify_with_sbert, verify_with_nli
from .semantic_scorer import compute_semantic_similarity, detect_heuristic_signals, compute_composite_score
from .wikipedia_checker import fever_check_response, check_claim_against_wikipedia
from .wikidata_verifier import verify_entities_in_response, verify_entity
from .vector_rag import VectorRAGStore
from .llm_judge import evaluate_with_llm_judge
from .hallucination_classifier import run_full_classifier, classify_with_heuristics
from .batch_evaluator import run_batch_evaluation, run_single_evaluation

__all__ = [
    "query_ollama", "query_ollama_multiple", "check_ollama_running", "get_ollama_models",
    "analyze_self_consistency",
    "verify_with_sbert", "verify_with_nli",
    "compute_semantic_similarity", "detect_heuristic_signals", "compute_composite_score",
    "fever_check_response", "check_claim_against_wikipedia",
    "verify_entities_in_response", "verify_entity",
    "VectorRAGStore",
    "evaluate_with_llm_judge",
    "run_full_classifier", "classify_with_heuristics",
    "run_batch_evaluation", "run_single_evaluation",
]

# ─────────────────────────────────────────────────────────────────────────────
# Module: batch_evaluator.py
# Purpose: Batch Evaluation Mode — Compare Multiple Prompts or Models
#
# Technique: Run hallucination detection across multiple prompts in one go.
# Produces a comparison table showing which responses are most reliable.
# ─────────────────────────────────────────────────────────────────────────────

import time
import pandas as pd
import numpy as np
from typing import Callable, Optional
from .llm_client import query_ollama, query_huggingface, get_ollama_models
from .self_consistency import analyze_self_consistency
from .rag_verifier import verify_with_sbert
from .semantic_scorer import detect_heuristic_signals, compute_composite_score
from .hallucination_classifier import classify_with_heuristics


def run_single_evaluation(
    prompt: str,
    model: str = "llama3.2",
    backend: str = "ollama",
    context: str = "",
    num_samples: int = 2,
    temperature: float = 0.7,
    hf_token: str = "",
    hf_model: str = "microsoft/Phi-3-mini-4k-instruct",
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Run a complete hallucination analysis for a single prompt+model combination.

    Args:
        prompt: The question/prompt to evaluate
        model: Ollama model name
        backend: "ollama" or "huggingface"
        context: Optional reference context
        num_samples: Samples for self-consistency
        temperature: LLM temperature
        hf_token: HuggingFace API token
        hf_model: HuggingFace model ID
        progress_callback: Optional function to call with status updates

    Returns:
        dict with complete analysis results
    """
    def update(msg):
        if progress_callback:
            progress_callback(msg)

    # 1. Generate main response
    update("Generating response...")
    if backend == "ollama":
        response = query_ollama(prompt, model=model, temperature=temperature, max_tokens=400)
    else:
        response = query_huggingface(prompt, model=hf_model, hf_token=hf_token, temperature=temperature)

    if response.startswith("[ERROR]"):
        return {
            "prompt": prompt,
            "model": model,
            "response": response,
            "error": True,
            "consistency_score": 0.0,
            "grounding_score": 0.0,
            "heuristic_score": 1.0,
            "composite_score": 0.0,
            "hallucination_risk": "Error",
            "risk_color": "gray",
        }

    # 2. Self-consistency
    update("Running self-consistency check...")
    if backend == "ollama":
        all_responses = [response]
        for _ in range(num_samples - 1):
            r = query_ollama(prompt, model=model, temperature=temperature, max_tokens=400)
            all_responses.append(r)
            time.sleep(0.2)
    else:
        all_responses = [response]
        for _ in range(num_samples - 1):
            r = query_huggingface(prompt, model=hf_model, hf_token=hf_token, temperature=temperature)
            all_responses.append(r)

    consistency = analyze_self_consistency(all_responses, use_sbert=False)
    consistency_score = consistency["consistency_score"]

    # 3. RAG grounding (if context provided)
    update("Checking RAG grounding...")
    grounding_score = 0.5
    if context.strip():
        rag_result = verify_with_sbert(response, context, threshold=0.4)
        grounding_score = rag_result.get("grounding_score", 0.5)

    # 4. Heuristic classifier
    update("Running heuristic classifier...")
    heuristic = classify_with_heuristics(response)
    heuristic_score = 1.0 - heuristic["hallucination_score"]  # invert (higher = more factual)

    # 5. Compute composite
    composite = compute_composite_score(
        semantic_score=heuristic_score,
        consistency_score=consistency_score,
        grounding_score=grounding_score,
        weights=(0.30, 0.35, 0.35),
    )

    return {
        "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "model": model if backend == "ollama" else hf_model,
        "backend": backend,
        "response": response,
        "response_length": len(response.split()),
        "consistency_score": round(consistency_score, 3),
        "grounding_score": round(grounding_score, 3),
        "heuristic_score": round(heuristic_score, 3),
        "composite_score": composite["composite_score"],
        "hallucination_risk": composite["hallucination_risk"],
        "risk_color": composite["risk_color"],
        "badge": composite["badge"],
        "verdict": composite["verdict"],
        "hedge_count": detect_heuristic_signals(response)["hedge_count"],
        "error": False,
    }


def run_batch_evaluation(
    prompts: list[str],
    models: list[dict],  # [{"model": "llama3.2", "backend": "ollama"}, ...]
    context: str = "",
    num_samples: int = 2,
    temperature: float = 0.7,
    hf_token: str = "",
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Run batch evaluation across multiple prompts and/or models.

    Args:
        prompts: List of prompts to evaluate
        models: List of model configs [{"model": name, "backend": "ollama"/"huggingface"}]
        context: Optional reference context for all prompts
        num_samples: Self-consistency samples per evaluation
        temperature: LLM temperature
        hf_token: HuggingFace token
        progress_callback: Status update function

    Returns:
        dict with results DataFrame and summary statistics
    """
    results = []
    total = len(prompts) * len(models)
    done = 0

    for prompt in prompts:
        for model_cfg in models:
            done += 1
            model_name = model_cfg.get("model", "llama3.2")
            backend = model_cfg.get("backend", "ollama")
            hf_model = model_cfg.get("hf_model", "microsoft/Phi-3-mini-4k-instruct")

            def cb(msg, p=prompt[:30], m=model_name, d=done, t=total):
                if progress_callback:
                    progress_callback(f"[{d}/{t}] {p}... — {m}: {msg}")

            result = run_single_evaluation(
                prompt=prompt,
                model=model_name,
                backend=backend,
                context=context,
                num_samples=num_samples,
                temperature=temperature,
                hf_token=hf_token,
                hf_model=hf_model,
                progress_callback=cb,
            )
            results.append(result)

    if not results:
        return {"results": [], "df": pd.DataFrame(), "summary": {}}

    df = pd.DataFrame(results)

    # Summary statistics
    summary = {
        "total_evaluations": len(results),
        "avg_composite_score": round(df["composite_score"].mean(), 3),
        "avg_consistency": round(df["consistency_score"].mean(), 3),
        "avg_grounding": round(df["grounding_score"].mean(), 3),
        "high_risk_count": int((df["hallucination_risk"] == "High").sum()),
        "medium_risk_count": int((df["hallucination_risk"] == "Medium").sum()),
        "low_risk_count": int((df["hallucination_risk"] == "Low").sum()),
        "best_prompt": df.loc[df["composite_score"].idxmax(), "prompt"] if len(df) > 0 else "N/A",
        "worst_prompt": df.loc[df["composite_score"].idxmin(), "prompt"] if len(df) > 0 else "N/A",
    }

    # Per-model summary (if multiple models)
    if len(models) > 1:
        model_summary = df.groupby("model")["composite_score"].agg(["mean", "min", "max"]).round(3)
        summary["model_comparison"] = model_summary.to_dict()

    return {
        "results": results,
        "df": df,
        "summary": summary,
    }

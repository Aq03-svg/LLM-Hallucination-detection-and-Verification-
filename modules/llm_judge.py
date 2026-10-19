# ─────────────────────────────────────────────────────────────────────────────
# Module: llm_judge.py
# Purpose: LLM-as-Judge Evaluation (Using local Ollama model as evaluator)
#
# Technique: Use a second LLM (the "judge") to evaluate whether the first
# LLM's response is factually accurate, relevant, and non-hallucinatory.
# Based on MT-Bench and LLM-as-Judge papers.
# ─────────────────────────────────────────────────────────────────────────────

import re
import json
from typing import Optional
from .llm_client import query_ollama, check_ollama_running, query_huggingface


# ── Evaluation Prompts ────────────────────────────────────────────────────────

FACTUALITY_PROMPT = """You are an expert fact-checker evaluating an AI assistant's response for factual accuracy.

**Original Question:** {question}

**AI Response to Evaluate:** {response}

{context_section}

**Your Task:** Evaluate the response on the following criteria and give a JSON output:

1. **factuality_score** (0-10): How factually accurate is the response? 
   - 10 = completely accurate
   - 5 = partially accurate with some errors
   - 0 = completely fabricated/hallucinated

2. **hallucination_detected** (true/false): Does the response contain hallucinations?

3. **specific_issues**: List any specific factual errors or suspicious claims found (empty list if none).

4. **confidence**: How confident are you in this evaluation? ("high", "medium", "low")

5. **verdict**: One-line summary verdict.

Respond ONLY with valid JSON in this exact format:
{{
  "factuality_score": <0-10>,
  "hallucination_detected": <true/false>,
  "specific_issues": ["<issue1>", "<issue2>"],
  "confidence": "<high/medium/low>",
  "verdict": "<one line verdict>"
}}"""

CONTEXT_SECTION_TEMPLATE = """
**Reference Context (use this to verify the response):**
{context}
"""

RELEVANCE_PROMPT = """You are evaluating an AI response for relevance and completeness.

**Question:** {question}
**Response:** {response}

Rate the response from 0-10 on:
- relevance (does it answer the question?)
- completeness (does it cover all aspects?)
- coherence (is it logically consistent?)

Respond ONLY with valid JSON:
{{
  "relevance_score": <0-10>,
  "completeness_score": <0-10>,
  "coherence_score": <0-10>,
  "overall_quality": <0-10>,
  "feedback": "<brief feedback>"
}}"""


# ── Judge Evaluation ──────────────────────────────────────────────────────────

def parse_judge_json(raw_output: str) -> Optional[dict]:
    """
    Safely parse JSON from LLM judge output.
    Handles common LLM formatting issues.
    """
    # Extract JSON block from response
    json_match = re.search(r'\{[\s\S]*\}', raw_output)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        # Try to fix common issues
        text = json_match.group()
        text = re.sub(r',\s*}', '}', text)   # trailing commas
        text = re.sub(r',\s*]', ']', text)
        try:
            return json.loads(text)
        except Exception:
            return None


def evaluate_with_llm_judge(
    question: str,
    response: str,
    context: str = "",
    judge_model: str = "llama3.2",
    judge_backend: str = "ollama",
    hf_token: str = "",
    hf_model: str = "microsoft/Phi-3-mini-4k-instruct",
    temperature: float = 0.1,
) -> dict:
    """
    Use an LLM as a judge to evaluate another LLM's response.

    Args:
        question: The original question asked
        response: The LLM response to evaluate
        context: Optional reference context for verification
        judge_model: Ollama model to use as judge
        judge_backend: "ollama" or "huggingface"
        hf_token: HuggingFace token (if using HF backend)
        hf_model: HuggingFace model ID
        temperature: Low temperature for more consistent judgments

    Returns:
        dict with factuality scores, issues, and overall verdict
    """
    context_section = (
        CONTEXT_SECTION_TEMPLATE.format(context=context[:1000])
        if context.strip() else ""
    )

    # Build factuality evaluation prompt
    fact_prompt = FACTUALITY_PROMPT.format(
        question=question,
        response=response,
        context_section=context_section,
    )

    # Build relevance evaluation prompt
    rel_prompt = RELEVANCE_PROMPT.format(
        question=question,
        response=response,
    )

    def call_judge(prompt: str) -> str:
        if judge_backend == "ollama":
            return query_ollama(
                prompt,
                model=judge_model,
                system_prompt="You are a precise evaluator. Always respond with valid JSON only.",
                temperature=temperature,
                max_tokens=400,
            )
        else:
            return query_huggingface(
                prompt,
                model=hf_model,
                hf_token=hf_token,
                temperature=temperature,
                max_tokens=400,
            )

    # ── Run factuality evaluation ──────────────────────────────────────────────
    fact_raw = call_judge(fact_prompt)
    fact_result = parse_judge_json(fact_raw)

    if fact_result is None:
        fact_result = {
            "factuality_score": 5,
            "hallucination_detected": False,
            "specific_issues": ["Could not parse judge output"],
            "confidence": "low",
            "verdict": "Judge output parsing failed — try again",
            "raw_output": fact_raw,
        }

    # ── Run relevance evaluation ───────────────────────────────────────────────
    rel_raw = call_judge(rel_prompt)
    rel_result = parse_judge_json(rel_raw)

    if rel_result is None:
        rel_result = {
            "relevance_score": 5,
            "completeness_score": 5,
            "coherence_score": 5,
            "overall_quality": 5,
            "feedback": "Could not parse judge output",
        }

    # ── Compute overall judge score ────────────────────────────────────────────
    factuality = fact_result.get("factuality_score", 5)
    quality = rel_result.get("overall_quality", 5)
    composite_10 = factuality * 0.6 + quality * 0.4
    composite_norm = composite_10 / 10.0  # normalize to 0-1

    hallucinated = fact_result.get("hallucination_detected", False)
    issues = fact_result.get("specific_issues", [])

    if composite_norm >= 0.70 and not hallucinated:
        risk = "Low"
        color = "green"
        badge = "✅"
        overall = f"✅ Response is high quality (Judge score: {composite_10:.1f}/10)"
    elif composite_norm >= 0.45 or (hallucinated and composite_norm >= 0.45):
        risk = "Medium"
        color = "orange"
        badge = "⚠️"
        overall = f"⚠️ Response has some issues (Judge score: {composite_10:.1f}/10)"
    else:
        risk = "High"
        color = "red"
        badge = "🚨"
        overall = f"🚨 Response likely hallucinated (Judge score: {composite_10:.1f}/10)"

    return {
        "judge_model": judge_model,
        "judge_backend": judge_backend,
        "factuality": fact_result,
        "relevance": rel_result,
        "factuality_score": factuality,
        "quality_score": quality,
        "composite_score_10": round(composite_10, 2),
        "composite_score_normalized": round(composite_norm, 4),
        "hallucination_detected": hallucinated,
        "specific_issues": issues,
        "hallucination_risk": risk,
        "risk_color": color,
        "badge": badge,
        "overall_verdict": overall,
        "judge_confidence": fact_result.get("confidence", "low"),
        "raw_factuality": fact_raw,
        "raw_relevance": rel_raw,
    }

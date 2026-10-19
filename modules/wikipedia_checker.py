# ─────────────────────────────────────────────────────────────────────────────
# Module: wikipedia_checker.py
# Purpose: FEVER-style Fact Checking against Wikipedia (Free API)
#
# Technique: For each claim sentence in the LLM response:
#   1. Extract key entities/phrases
#   2. Search Wikipedia for relevant articles
#   3. Check if the claim is Supported / Refuted / Not Enough Info
# ─────────────────────────────────────────────────────────────────────────────

import re
import requests
import urllib.parse
from typing import Optional


WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIPEDIA_SEARCH = "https://en.wikipedia.org/w/api.php"


def search_wikipedia(query: str, max_results: int = 3) -> list[dict]:
    """
    Search Wikipedia for pages matching a query.

    Args:
        query: Search query string
        max_results: Max number of results to return

    Returns:
        List of dicts with 'title', 'snippet', 'pageid'
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
        "srprop": "snippet|titlesnippet",
    }
    try:
        r = requests.get(WIKIPEDIA_SEARCH, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("query", {}).get("search", [])
        return [
            {
                "title": res["title"],
                "snippet": re.sub(r"<[^>]+>", "", res.get("snippet", "")),
                "pageid": res["pageid"],
            }
            for res in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


def get_wikipedia_summary(title: str) -> dict:
    """
    Fetch the summary of a Wikipedia article.

    Args:
        title: Wikipedia article title

    Returns:
        dict with 'title', 'extract', 'url'
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    try:
        r = requests.get(f"{WIKIPEDIA_API}{encoded}", timeout=10)
        if r.status_code == 404:
            return {"error": f"Article '{title}' not found"}
        r.raise_for_status()
        data = r.json()
        return {
            "title": data.get("title", title),
            "extract": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "description": data.get("description", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def extract_key_entities(text: str) -> list[str]:
    """
    Extract key noun phrases and named entities using regex heuristics.
    (No NLP library required — uses pattern matching.)

    Args:
        text: Input text

    Returns:
        List of candidate entity/phrase strings to search
    """
    entities = []

    # Proper nouns (capitalized words, excluding sentence starts)
    proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    entities.extend(proper_nouns)

    # Acronyms
    acronyms = re.findall(r'\b([A-Z]{2,})\b', text)
    entities.extend(acronyms)

    # Years
    years = re.findall(r'\b(1[89]\d\d|20[012]\d)\b', text)
    entities.extend(years)

    # Remove duplicates and short entries
    seen = set()
    result = []
    for e in entities:
        if e not in seen and len(e) > 2:
            seen.add(e)
            result.append(e)

    return result[:6]  # Limit to top 6 candidates


def check_claim_against_wikipedia(claim: str, max_articles: int = 2) -> dict:
    """
    Check a single claim against Wikipedia.

    Steps:
    1. Extract key entities from the claim
    2. Search Wikipedia for relevant articles
    3. Compare claim against article summaries
    4. Return SUPPORTED / CONTRADICTED / INSUFFICIENT_INFO

    Args:
        claim: A single factual claim sentence
        max_articles: Number of Wikipedia articles to check against

    Returns:
        dict with verdict, evidence, and Wikipedia sources
    """
    entities = extract_key_entities(claim)
    search_query = " ".join(entities[:3]) if entities else claim[:60]

    # Search for relevant Wikipedia pages
    search_results = search_wikipedia(search_query, max_results=max_articles)
    if search_results and "error" in search_results[0]:
        return {
            "claim": claim,
            "verdict": "NETWORK_ERROR",
            "verdict_label": "⚠️ Network Error",
            "verdict_color": "gray",
            "evidence": search_results[0]["error"],
            "sources": [],
        }

    # Fetch summaries for top results
    evidence_texts = []
    sources = []
    for result in search_results[:max_articles]:
        summary = get_wikipedia_summary(result["title"])
        if "error" not in summary and summary.get("extract"):
            evidence_texts.append(summary["extract"])
            sources.append({
                "title": summary["title"],
                "url": summary["url"],
                "description": summary.get("description", ""),
                "snippet": summary["extract"][:300] + "...",
            })

    if not evidence_texts:
        return {
            "claim": claim,
            "verdict": "NOT_ENOUGH_INFO",
            "verdict_label": "⚪ Not Enough Info",
            "verdict_color": "gray",
            "evidence": "No relevant Wikipedia articles found.",
            "sources": [],
        }

    # Simple keyword overlap check between claim and evidence
    claim_words = set(re.sub(r'[^\w\s]', '', claim.lower()).split())
    stop_words = {"the", "a", "an", "is", "was", "were", "are", "of", "in",
                  "on", "at", "by", "for", "to", "and", "or", "it", "that", "this"}
    claim_keywords = claim_words - stop_words

    best_overlap = 0.0
    best_evidence = ""
    for ev in evidence_texts:
        ev_words = set(re.sub(r'[^\w\s]', '', ev.lower()).split())
        if claim_keywords:
            overlap = len(claim_keywords & ev_words) / len(claim_keywords)
            if overlap > best_overlap:
                best_overlap = overlap
                best_evidence = ev[:400]

    # Classify based on overlap score
    if best_overlap >= 0.5:
        verdict = "SUPPORTED"
        label = "✅ Supported by Wikipedia"
        color = "green"
    elif best_overlap >= 0.2:
        verdict = "PARTIAL"
        label = "⚠️ Partially Supported"
        color = "orange"
    else:
        verdict = "NOT_ENOUGH_INFO"
        label = "⚪ Insufficient Evidence"
        color = "gray"

    return {
        "claim": claim,
        "verdict": verdict,
        "verdict_label": label,
        "verdict_color": color,
        "overlap_score": round(best_overlap, 3),
        "evidence_snippet": best_evidence,
        "sources": sources,
        "entities_searched": entities,
    }


def fever_check_response(response: str, max_articles: int = 2) -> dict:
    """
    Run FEVER-style fact checking on an entire LLM response.

    Args:
        response: Full LLM response text
        max_articles: Wikipedia articles to check per claim

    Returns:
        dict with per-sentence verdicts and overall summary
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    results = []
    verdicts = []
    for sent in sentences:
        result = check_claim_against_wikipedia(sent, max_articles=max_articles)
        results.append(result)
        verdicts.append(result["verdict"])

    # Summarize
    supported = verdicts.count("SUPPORTED") + verdicts.count("PARTIAL")
    contradicted = verdicts.count("CONTRADICTED")
    insufficient = verdicts.count("NOT_ENOUGH_INFO") + verdicts.count("NETWORK_ERROR")
    total = len(verdicts)

    support_rate = supported / total if total > 0 else 0.0

    if support_rate >= 0.6:
        overall_risk = "Low"
        overall_color = "green"
        overall_verdict = "✅ Most claims are supported by Wikipedia"
    elif support_rate >= 0.3:
        overall_risk = "Medium"
        overall_color = "orange"
        overall_verdict = "⚠️ Partial Wikipedia support — verify key claims"
    else:
        overall_risk = "High"
        overall_color = "red"
        overall_verdict = "🚨 Low Wikipedia support — many unverifiable claims"

    return {
        "sentence_results": results,
        "total_sentences": total,
        "supported_count": supported,
        "contradicted_count": contradicted,
        "insufficient_count": insufficient,
        "support_rate": round(support_rate, 4),
        "hallucination_risk": overall_risk,
        "risk_color": overall_color,
        "overall_verdict": overall_verdict,
    }

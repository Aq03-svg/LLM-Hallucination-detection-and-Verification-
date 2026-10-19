# ─────────────────────────────────────────────────────────────────────────────
# Module: wikidata_verifier.py
# Purpose: Entity Verification via Wikidata Knowledge Graph (Free SPARQL API)
#
# Technique: Extract named entities from LLM response, query Wikidata to
# verify their existence and properties (birthdate, occupation, etc.)
# ─────────────────────────────────────────────────────────────────────────────

import re
import requests
from typing import Optional


WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": "LLMHallucinationDetector/1.0 (academic project)",
    "Accept": "application/sparql-results+json",
}


def search_wikidata_entity(name: str) -> list[dict]:
    """
    Search for an entity in Wikidata by name.

    Args:
        name: Entity name to search (e.g., "Albert Einstein")

    Returns:
        List of matching entities with id, label, description
    """
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "limit": 5,
        "format": "json",
    }
    try:
        r = requests.get(WIKIDATA_API, params=params, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("search", []):
            results.append({
                "id": item.get("id"),
                "label": item.get("label"),
                "description": item.get("description", ""),
                "url": f"https://www.wikidata.org/wiki/{item.get('id')}",
                "aliases": item.get("aliases", []),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def get_entity_properties(entity_id: str) -> dict:
    """
    Get key properties of a Wikidata entity via SPARQL.

    Args:
        entity_id: Wikidata entity ID (e.g., "Q937" for Albert Einstein)

    Returns:
        dict with entity properties
    """
    query = f"""
    SELECT ?propLabel ?valueLabel WHERE {{
      wd:{entity_id} ?prop ?value .
      ?propEntity wikibase:directClaim ?prop .
      ?propEntity rdfs:label ?propLabel .
      FILTER(LANG(?propLabel) = "en")
      ?value rdfs:label ?valueLabel .
      FILTER(LANG(?valueLabel) = "en")
    }}
    LIMIT 20
    """
    try:
        r = requests.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        bindings = data.get("results", {}).get("bindings", [])
        props = {}
        for b in bindings:
            prop = b.get("propLabel", {}).get("value", "")
            val  = b.get("valueLabel", {}).get("value", "")
            if prop and val:
                if prop in props:
                    if isinstance(props[prop], list):
                        props[prop].append(val)
                    else:
                        props[prop] = [props[prop], val]
                else:
                    props[prop] = val
        return props
    except Exception as e:
        return {"error": str(e)}


def extract_named_entities(text: str) -> list[str]:
    """
    Extract likely person/organization/place names using regex.

    Args:
        text: Input text

    Returns:
        List of candidate entity names
    """
    # Multi-word proper nouns (e.g., "Albert Einstein", "OpenAI")
    multi_word = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    # Single capitalized words (not at sentence start)
    single = re.findall(r'(?<=[a-z]\s)([A-Z][a-z]{3,})\b', text)
    # Known org patterns
    orgs = re.findall(r'\b(OpenAI|Google|Meta|Microsoft|DeepMind|HuggingFace|Anthropic|'
                      r'MIT|Stanford|Harvard|Oxford|CERN|NASA|IEEE|ACM)\b', text)

    seen = set()
    results = []
    for e in multi_word + single + orgs:
        if e not in seen and len(e) > 3:
            seen.add(e)
            results.append(e)

    return results[:8]


def verify_entity(entity_name: str) -> dict:
    """
    Verify a named entity via Wikidata: check if it exists and get key facts.

    Args:
        entity_name: The entity name from the LLM response

    Returns:
        dict with verification status and Wikidata facts
    """
    search_results = search_wikidata_entity(entity_name)

    if not search_results or "error" in search_results[0]:
        return {
            "entity": entity_name,
            "status": "NOT_FOUND",
            "label": "⚪ Not Found in Wikidata",
            "color": "gray",
            "description": "Could not verify this entity.",
            "wikidata_id": None,
            "properties": {},
        }

    top = search_results[0]
    if "error" in top:
        return {
            "entity": entity_name,
            "status": "ERROR",
            "label": "⚠️ API Error",
            "color": "gray",
            "description": top["error"],
            "wikidata_id": None,
            "properties": {},
        }

    wikidata_id = top["id"]
    description = top.get("description", "No description available")

    # Check if label roughly matches (fuzzy)
    name_lower = entity_name.lower()
    label_lower = top.get("label", "").lower()
    match_score = len(set(name_lower.split()) & set(label_lower.split())) / max(len(name_lower.split()), 1)

    status = "VERIFIED" if match_score >= 0.5 else "PARTIAL_MATCH"
    label = "✅ Verified in Wikidata" if status == "VERIFIED" else "⚠️ Partial Match"
    color = "green" if status == "VERIFIED" else "orange"

    return {
        "entity": entity_name,
        "status": status,
        "label": label,
        "color": color,
        "description": description,
        "wikidata_id": wikidata_id,
        "wikidata_label": top.get("label"),
        "wikidata_url": top.get("url"),
        "match_score": round(match_score, 2),
        "alternatives": search_results[1:3],
    }


def verify_entities_in_response(response: str) -> dict:
    """
    Extract and verify all named entities found in an LLM response.

    Args:
        response: Full LLM response text

    Returns:
        dict with per-entity verdicts and overall summary
    """
    entities = extract_named_entities(response)

    if not entities:
        return {
            "entities_found": [],
            "entity_results": [],
            "total_entities": 0,
            "verified_count": 0,
            "not_found_count": 0,
            "overall_verdict": "⚪ No named entities detected",
            "hallucination_risk": "Unknown",
            "risk_color": "gray",
        }

    entity_results = []
    for entity in entities:
        result = verify_entity(entity)
        entity_results.append(result)

    verified = sum(1 for r in entity_results if r["status"] == "VERIFIED")
    not_found = sum(1 for r in entity_results if r["status"] == "NOT_FOUND")
    total = len(entity_results)

    verify_rate = verified / total if total > 0 else 0.0

    if verify_rate >= 0.7:
        risk = "Low"
        color = "green"
        verdict = "✅ Most entities verified in Wikidata"
    elif verify_rate >= 0.4:
        risk = "Medium"
        color = "orange"
        verdict = "⚠️ Some entities could not be verified"
    else:
        risk = "High"
        color = "red"
        verdict = "🚨 Many entities unverifiable — possible fabrication"

    return {
        "entities_found": entities,
        "entity_results": entity_results,
        "total_entities": total,
        "verified_count": verified,
        "not_found_count": not_found,
        "verify_rate": round(verify_rate, 4),
        "hallucination_risk": risk,
        "risk_color": color,
        "overall_verdict": verdict,
    }

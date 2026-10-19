# ─────────────────────────────────────────────────────────────────────────────
# Module: llm_client.py
# Purpose: Connect to LLMs via Ollama (local) or HuggingFace (free API)
# ─────────────────────────────────────────────────────────────────────────────

import requests
import json
import time
from typing import Optional


OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama_running() -> bool:
    """Check if Ollama server is running locally."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_ollama_models() -> list[str]:
    """List locally available Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def query_ollama(
    prompt: str,
    model: str = "llama3.2",
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """
    Query a local Ollama model.

    Args:
        prompt: User prompt text
        model: Ollama model name (e.g., 'llama3.2', 'mistral', 'gemma2')
        system_prompt: Optional system instruction
        temperature: Sampling temperature (0 = deterministic, 1 = creative)
        max_tokens: Maximum tokens in response

    Returns:
        Generated text response
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to Ollama. Make sure Ollama is running (`ollama serve`)."
    except Exception as e:
        return f"[ERROR] {str(e)}"


def query_ollama_multiple(
    prompt: str,
    model: str = "llama3.2",
    n: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> list[str]:
    """
    Query the same Ollama model multiple times for self-consistency checking.

    Args:
        prompt: User prompt text
        model: Ollama model name
        n: Number of independent responses to generate
        temperature: Sampling temperature (should be > 0 for diversity)
        max_tokens: Maximum tokens per response

    Returns:
        List of n generated responses
    """
    responses = []
    for _ in range(n):
        resp = query_ollama(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
        responses.append(resp)
        time.sleep(0.3)  # slight delay to avoid rate issues
    return responses


def query_huggingface(
    prompt: str,
    model: str = "microsoft/Phi-3-mini-4k-instruct",
    hf_token: str = "",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """
    Query a HuggingFace Inference API (free tier, no local GPU needed).

    Args:
        prompt: User prompt text
        model: HuggingFace model ID
        hf_token: Optional HuggingFace API token (free at hf.co/settings/tokens)
        temperature: Sampling temperature
        max_tokens: Maximum new tokens

    Returns:
        Generated text response
    """
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
        },
    }
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        result = r.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        return str(result)
    except Exception as e:
        return f"[ERROR] HuggingFace API: {str(e)}"

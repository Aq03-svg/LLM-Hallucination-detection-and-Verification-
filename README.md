# 🔬 LLM Hallucination Detection & Verification
### A Walkthrough Project — Generative AI Subject

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

An **interactive Streamlit web app** that demonstrates three state-of-the-art techniques
for detecting and verifying hallucinations in Large Language Model (LLM) outputs —
all using **free, open-source tools** (Ollama + HuggingFace).

---

## 🎯 What is LLM Hallucination?

LLMs sometimes generate text that sounds fluent and confident but is **factually wrong or fabricated**.
This is called a *hallucination*. Examples:
- Citing non-existent papers
- Wrong dates, names, or statistics
- Fabricating facts about real people

This project shows you **how to detect and verify** these hallucinations programmatically.

---

## 🛠️ Detection Techniques

| # | Technique | How It Works |
|---|-----------|-------------|
| 1️⃣ | **Self-Consistency** | Query the LLM N times; inconsistent answers = hallucination |
| 2️⃣ | **RAG-Based Verification** | Compare response to a knowledge base; unsupported claims flagged |
| 3️⃣ | **Semantic Similarity** | Compare to reference answer using SBERT + ROUGE scores |
| ➕ | **Heuristic Signals** | Detect hedging phrases and suspicious claim patterns |

---

## 🚀 Quick Start

### Step 1 — Clone / open the project
```bash
cd "LLM Hallunication detection and Verification"
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — (Option A) Use Ollama for local LLM
```bash
# Download Ollama from https://ollama.com
# Then pull a model and start the server:
ollama pull llama3.2
ollama serve
```

### Step 3 — (Option B) Use HuggingFace (no GPU needed)
Get a free token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
and paste it in the sidebar.

### Step 4 — Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
LLM Hallunication detection and Verification/
├── app.py                    ← Main Streamlit application (5 tabs)
├── requirements.txt          ← Python dependencies
├── README.md
├── modules/
│   ├── __init__.py
│   ├── llm_client.py         ← Ollama + HuggingFace API interface
│   ├── self_consistency.py   ← Technique 1: Self-consistency checking
│   ├── rag_verifier.py       ← Technique 2: RAG-based grounding verification
│   └── semantic_scorer.py    ← Technique 3: Semantic similarity + ROUGE
└── sample_docs/
    └── ai_facts.txt          ← Sample knowledge base (AI/ML facts)
```

---

## 📸 App Tabs

| Tab | Description |
|-----|-------------|
| 📖 **Introduction** | What is hallucination, types, why it matters |
| 🔬 **Live Demo** | Run all 3 detection techniques on your own prompts |
| 📐 **Methods Explained** | Deep-dive into the algorithms with formulas |
| 📊 **Method Comparison** | Side-by-side comparison, research references |
| ℹ️ **About** | Tech stack, project structure, future extensions |

---

## 📚 Research References

1. Wang et al. (2023) — [Self-Consistency Improves CoT Reasoning](https://arxiv.org/abs/2203.11171)
2. Lewis et al. (2020) — [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)
3. Manakul et al. (2023) — [SelfCheckGPT](https://arxiv.org/abs/2303.08896)
4. Min et al. (2023) — [FActScoring](https://arxiv.org/abs/2305.14251)
5. Reimers & Gurevych (2019) — [Sentence-BERT](https://arxiv.org/abs/1908.10084)

---

## 🔮 Future Extensions

- FEVER-style fact checking against Wikipedia
- Entity verification with Wikidata knowledge graphs
- Multi-document RAG with FAISS / ChromaDB vector stores
- LLM-as-judge evaluation pipeline
- Fine-tuned classifier (TruthfulQA, HaluEval datasets)
- Batch evaluation mode

---

*Generative AI Subject — Academic Project*

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   LLM Hallucination Detection & Verification — Interactive Walkthrough      ║
║   Subject: Generative AI                                                    ║
║   Built with: Streamlit + Sentence-Transformers + Ollama / HuggingFace     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run with:  streamlit run app.py
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="LLM Hallucination Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Add modules to path ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from modules.llm_client import (
    check_ollama_running,
    get_ollama_models,
    query_ollama,
    query_ollama_multiple,
    query_huggingface,
)
from modules.self_consistency import analyze_self_consistency
from modules.rag_verifier import verify_with_sbert, split_into_sentences
from modules.semantic_scorer import (
    compute_semantic_similarity,
    detect_heuristic_signals,
    compute_composite_score,
)
# ── Extension modules ─────────────────────────────────────────────────────────
from modules.wikipedia_checker import fever_check_response
from modules.wikidata_verifier import verify_entities_in_response
from modules.vector_rag import VectorRAGStore
from modules.llm_judge import evaluate_with_llm_judge
from modules.hallucination_classifier import run_full_classifier
from modules.batch_evaluator import run_batch_evaluation, run_single_evaluation

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 12px 18px;
    }

    /* Risk badges */
    .badge-low    { background:#1a4731; color:#4ade80; border-radius:6px; padding:4px 10px; font-weight:700; }
    .badge-medium { background:#4a3800; color:#facc15; border-radius:6px; padding:4px 10px; font-weight:700; }
    .badge-high   { background:#4a1515; color:#f87171; border-radius:6px; padding:4px 10px; font-weight:700; }

    /* Section dividers */
    .section-header {
        border-left: 4px solid #6366f1;
        padding-left: 12px;
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 18px;
        font-weight: 600;
    }

    /* Sentence highlight cards */
    .sent-supported  { background:#0d2b1a; border-left:4px solid #4ade80; padding:8px 12px; margin:4px 0; border-radius:4px; }
    .sent-unsupported{ background:#2b0d0d; border-left:4px solid #f87171; padding:8px 12px; margin:4px 0; border-radius:4px; }
    .sent-neutral    { background:#1a1a2b; border-left:4px solid #94a3b8; padding:8px 12px; margin:4px 0; border-radius:4px; }

    /* Code font */
    code { color: #a78bfa; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://em-content.zobj.net/source/google/387/microscope_1f52c.png", width=60)
    st.title("🔬 Hallucination Detector")
    st.caption("LLM Verification Walkthrough — Generative AI Project")
    st.divider()

    # ── LLM Backend ───────────────────────────────────────────────────────────
    st.subheader("🤖 LLM Backend")
    backend = st.radio(
        "Choose LLM source:",
        ["Ollama (Local)", "HuggingFace API (Free)"],
        help="Ollama runs models locally. HuggingFace uses free cloud inference.",
    )

    ollama_ok = False
    selected_model = "llama3.2"

    if backend == "Ollama (Local)":
        ollama_ok = check_ollama_running()
        if ollama_ok:
            st.success("✅ Ollama is running")
            models = get_ollama_models()
            if models:
                selected_model = st.selectbox("Select Ollama model:", models)
            else:
                st.warning("No models pulled yet. Run: `ollama pull llama3.2`")
                selected_model = st.text_input("Model name:", value="llama3.2")
        else:
            st.error("❌ Ollama not detected")
            st.info("Start with: `ollama serve`\nPull a model: `ollama pull llama3.2`")
            selected_model = "llama3.2"
    else:
        hf_token = st.text_input(
            "HuggingFace Token (optional):",
            type="password",
            help="Get free token at hf.co/settings/tokens",
        )
        hf_model = st.selectbox(
            "HF Model:",
            [
                "microsoft/Phi-3-mini-4k-instruct",
                "google/flan-t5-large",
                "HuggingFaceH4/zephyr-7b-beta",
            ],
        )

    st.divider()

    # ── Detection Settings ─────────────────────────────────────────────────────
    st.subheader("⚙️ Detection Settings")

    num_samples = st.slider(
        "Self-consistency samples:",
        min_value=2, max_value=5, value=3,
        help="Number of times the LLM is queried for self-consistency checking",
    )
    temperature = st.slider(
        "Temperature:",
        min_value=0.3, max_value=1.0, value=0.7, step=0.05,
        help="Higher = more diverse (needed for consistency checks)",
    )
    similarity_threshold = st.slider(
        "RAG grounding threshold:",
        min_value=0.2, max_value=0.8, value=0.45, step=0.05,
        help="Min similarity score to consider a sentence 'supported'",
    )

    use_sbert = st.toggle("Use SBERT (requires install)", value=False)
    use_nli   = st.toggle("Use NLI model (heavy, slow)", value=False)

    st.divider()
    st.caption("📚 Generative AI — Academic Project")
    st.caption("All detection runs locally or via free APIs.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — TABS
# ════════════════════════════════════════════════════════════════════════════

tab_intro, tab_demo, tab_methods, tab_compare, tab_about, \
tab_wiki, tab_wiki_ent, tab_vrag, tab_judge, tab_classifier, tab_batch = st.tabs([
    "📖 Introduction",
    "🔬 Live Demo",
    "📐 Methods Explained",
    "📊 Method Comparison",
    "ℹ️ About",
    "🌐 Wikipedia Fact Check",
    "🏷️ Entity Verification",
    "📚 Multi-Doc RAG",
    "⚖️ LLM-as-Judge",
    "🤖 Hallucination Classifier",
    "📋 Batch Evaluation",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — INTRODUCTION
# ────────────────────────────────────────────────────────────────────────────
with tab_intro:
    st.title("🧠 LLM Hallucination Detection & Verification")
    st.markdown(
        """
        > **What is LLM Hallucination?**
        >
        > Large Language Models (LLMs) sometimes generate text that sounds confident and fluent
        > but is **factually incorrect, fabricated, or unsupported**. This is called a *hallucination*.

        ---
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ### 🎯 Why It Matters
            - Medical misinformation → patient harm
            - Legal errors → liability issues
            - Academic dishonesty → wrong facts
            - Business decisions based on false data

            ### 📊 How Common Is It?
            Research shows LLMs hallucinate in **15–40%** of factual queries,
            depending on the domain and model.
            """
        )
    with col2:
        st.markdown(
            """
            ### 🔍 Types of Hallucinations

            | Type | Description | Example |
            |------|-------------|---------|
            | **Factual** | Wrong facts stated as truth | "Einstein won Nobel Prize in 1922 for relativity" |
            | **Entity** | Fabricated names, places, dates | Non-existent papers, fake citations |
            | **Logical** | Self-contradictory statements | Inconsistent reasoning chains |
            | **Contextual** | Ignoring provided context | Answering differently than context suggests |
            """
        )

    st.divider()
    st.markdown("### 🛠️ Detection Techniques in This Project")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(
            """
            **1️⃣ Self-Consistency Checking**

            Ask the same question multiple times.
            Compare answers — inconsistency = likely hallucination.

            *Based on Wang et al. (2023) — "Self-Consistency Improves Chain-of-Thought Reasoning"*
            """
        )
    with col_b:
        st.success(
            """
            **2️⃣ RAG-Based Verification**

            Ground the response against a knowledge base.
            Check which sentences are semantically supported.

            *Based on RAG (Lewis et al. 2020) + NLI grounding*
            """
        )
    with col_c:
        st.warning(
            """
            **3️⃣ Semantic Similarity Scoring**

            Compare response to a reference answer.
            Use ROUGE + cosine similarity to measure divergence.

            *Uses sentence-transformers + ROUGE metrics*
            """
        )

    st.divider()
    st.markdown("### 🚀 Getting Started")
    st.code(
        """
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install & start Ollama (for local LLM)
#    Download from: https://ollama.com
ollama pull llama3.2
ollama serve

# 3. Run the app
streamlit run app.py
        """,
        language="bash",
    )


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — LIVE DEMO
# ────────────────────────────────────────────────────────────────────────────
with tab_demo:
    st.title("🔬 Live Hallucination Detection Demo")

    # ── Prompt Input ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Step 1 — Enter a Prompt</div>', unsafe_allow_html=True)

    example_prompts = {
        "Custom prompt": "",
        "When was ChatGPT released?": "When was ChatGPT released and who made it?",
        "Einstein Nobel Prize": "What did Albert Einstein win the Nobel Prize for, and in what year?",
        "BERT vs GPT": "What is the difference between BERT and GPT language models?",
        "LLM Hallucination causes": "What are the main causes of hallucinations in large language models?",
        "Fabricated fact bait": "Who invented the ReasonForge algorithm and when was it published?",
    }
    selected_example = st.selectbox("Load example prompt:", list(example_prompts.keys()))
    user_prompt = st.text_area(
        "Your prompt:",
        value=example_prompts[selected_example],
        height=100,
        placeholder="Type your question here...",
    )

    # ── Reference / Context ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Step 2 — Provide Context (for RAG & Semantic)</div>', unsafe_allow_html=True)

    context_mode = st.radio(
        "Context source:",
        ["Upload .txt file", "Paste text", "Use sample knowledge base"],
        horizontal=True,
    )
    context_text = ""
    if context_mode == "Upload .txt file":
        uploaded = st.file_uploader("Upload knowledge base (.txt)", type=["txt"])
        if uploaded:
            context_text = uploaded.read().decode("utf-8")
            st.success(f"Loaded {len(context_text)} characters")
    elif context_mode == "Paste text":
        context_text = st.text_area("Paste your reference/context here:", height=150)
    else:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_docs", "ai_facts.txt")
        try:
            with open(sample_path, "r") as f:
                context_text = f.read()
            with st.expander("📄 View loaded sample knowledge base"):
                st.text(context_text)
        except FileNotFoundError:
            st.warning("Sample file not found. Paste context manually.")

    reference_answer = st.text_area(
        "Reference answer (optional — for semantic similarity):",
        height=80,
        placeholder="If you know the correct answer, paste it here for comparison...",
    )

    # ── Run Button ────────────────────────────────────────────────────────────
    st.divider()
    run_btn = st.button("🚀 Run Hallucination Analysis", type="primary", use_container_width=True)

    if run_btn:
        if not user_prompt.strip():
            st.error("Please enter a prompt first.")
        else:
            # ── Step 1: Generate LLM Response ─────────────────────────────────
            st.divider()
            st.markdown("## 🤖 LLM Response Generation")

            with st.spinner("Querying LLM..."):
                if backend == "Ollama (Local)":
                    if not ollama_ok:
                        st.error("Ollama is not running. Please start it first.")
                        st.stop()
                    main_response = query_ollama(
                        user_prompt,
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=512,
                    )
                else:
                    main_response = query_huggingface(
                        user_prompt,
                        model=hf_model,
                        hf_token=hf_token if hf_token else "",
                        temperature=temperature,
                    )

            if main_response.startswith("[ERROR]"):
                st.error(main_response)
                st.stop()

            st.success("✅ LLM Response Generated")
            with st.expander("📝 View Full LLM Response", expanded=True):
                st.markdown(f"> {main_response}")

            # ── TECHNIQUE 1: Self-Consistency ─────────────────────────────────
            st.divider()
            st.markdown("## 1️⃣ Self-Consistency Analysis")
            st.caption(
                "Querying the same LLM multiple times to check answer stability. "
                "Inconsistency = hallucination risk."
            )

            with st.spinner(f"Generating {num_samples} responses for consistency check..."):
                if backend == "Ollama (Local)":
                    all_responses = query_ollama_multiple(
                        user_prompt,
                        model=selected_model,
                        n=num_samples,
                        temperature=temperature,
                    )
                else:
                    all_responses = [
                        query_huggingface(user_prompt, model=hf_model,
                                          hf_token=hf_token if hf_token else "",
                                          temperature=temperature)
                        for _ in range(num_samples)
                    ]

                consistency_result = analyze_self_consistency(
                    all_responses,
                    use_sbert=use_sbert,
                )

            # Display responses
            with st.expander(f"View all {num_samples} responses"):
                for idx, resp in enumerate(all_responses, 1):
                    st.markdown(f"**Response {idx}:**")
                    st.markdown(f"> {resp}")
                    st.divider()

            # Metrics
            col1, col2, col3 = st.columns(3)
            score_pct = int(consistency_result["consistency_score"] * 100)
            col1.metric("Consistency Score", f"{score_pct}%")
            col2.metric("Hallucination Risk", consistency_result["hallucination_risk"])
            col3.metric("Samples Used", consistency_result["num_responses"])

            risk_cls = consistency_result["hallucination_risk"].lower()
            st.markdown(
                f'<div style="margin-top:10px"><span class="badge-{risk_cls}">'
                f'{consistency_result["verdict"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.info(f"**Explanation:** {consistency_result['explanation']}")
            st.caption(f"Method: {consistency_result['method_used']}")

            # Similarity heatmap
            if len(all_responses) >= 2:
                import plotly.express as px
                matrix = consistency_result["similarity_matrix"]
                labels = [f"R{i+1}" for i in range(len(all_responses))]
                fig = px.imshow(
                    matrix,
                    x=labels, y=labels,
                    color_continuous_scale="RdYlGn",
                    zmin=0, zmax=1,
                    title="Pairwise Response Similarity Heatmap",
                    text_auto=".2f",
                )
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            # Store score for composite
            consistency_score_val = consistency_result["consistency_score"]

            # ── TECHNIQUE 2: RAG Verification ─────────────────────────────────
            st.divider()
            st.markdown("## 2️⃣ RAG-Based Grounding Verification")
            st.caption(
                "Checking each sentence of the response against the knowledge base. "
                "Unsupported sentences are flagged as potential hallucinations."
            )

            grounding_score_val = 0.5  # default if no context

            if not context_text.strip():
                st.warning("No context provided. Skipping RAG verification.")
            else:
                with st.spinner("Verifying response against knowledge base..."):
                    rag_result = verify_with_sbert(
                        main_response,
                        context_text,
                        threshold=similarity_threshold,
                    )

                col1, col2, col3 = st.columns(3)
                grounding_pct = int(rag_result.get("grounding_score", 0) * 100)
                supported = rag_result.get("supported_count", 0)
                total = rag_result.get("total_sentences", 0)
                col1.metric("Grounding Score", f"{grounding_pct}%")
                col2.metric("Supported Sentences", f"{supported}/{total}")
                col3.metric("Risk", rag_result.get("hallucination_risk", "N/A"))

                rag_risk_cls = rag_result.get("hallucination_risk", "medium").lower()
                st.markdown(
                    f'<div style="margin-top:10px"><span class="badge-{rag_risk_cls}">'
                    f'{rag_result.get("verdict", "")}</span></div>',
                    unsafe_allow_html=True,
                )
                st.info(f"**Explanation:** {rag_result.get('explanation', '')}")
                st.caption(f"Method: {rag_result.get('method', '')}")

                # Per-sentence breakdown
                st.markdown("#### 📝 Sentence-Level Grounding Breakdown")
                for sr in rag_result.get("sentence_results", []):
                    css_cls = "sent-supported" if sr["supported"] else "sent-unsupported"
                    sim_bar = "█" * int(sr["best_similarity"] * 10) + "░" * (10 - int(sr["best_similarity"] * 10))
                    st.markdown(
                        f'<div class="{css_cls}">'
                        f'<b>{sr["label"]}</b> (sim: {sr["best_similarity"]:.2f}) '
                        f'<code>{sim_bar}</code><br>'
                        f'<i>"{sr["sentence"]}"</i><br>'
                        f'<small>Best match: "{sr["best_matching_context"][:100]}..."</small>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                grounding_score_val = rag_result.get("grounding_score", 0.5)

            # ── TECHNIQUE 3: Semantic Similarity Scoring ──────────────────────
            st.divider()
            st.markdown("## 3️⃣ Semantic Similarity & Heuristic Scoring")
            st.caption(
                "Comparing the response to a reference answer using cosine similarity + ROUGE. "
                "Also detecting hedging phrases and specific-claim patterns."
            )

            semantic_score_val = 0.5  # default if no reference

            # Heuristic signals (always run)
            heuristic = detect_heuristic_signals(main_response)

            col1, col2, col3 = st.columns(3)
            col1.metric("Hedging Phrases", heuristic["hedge_count"])
            col2.metric("Specific Claim Patterns", heuristic["indicator_count"])
            col3.metric("Heuristic Risk", heuristic["risk_level"])

            if heuristic["hedging_phrases_found"]:
                st.markdown("**Hedging phrases detected:**")
                st.code(", ".join(heuristic["hedging_phrases_found"]))

            if heuristic["specific_claim_patterns"]:
                st.markdown("**Specific claim patterns detected:**")
                st.code(", ".join(heuristic["specific_claim_patterns"][:10]))

            # Semantic similarity (if reference provided)
            if reference_answer.strip():
                with st.spinner("Computing semantic similarity..."):
                    sem_result = compute_semantic_similarity(
                        main_response,
                        reference_answer,
                    )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Semantic Similarity", f"{sem_result['semantic_similarity']:.2f}")
                col2.metric("ROUGE-1", f"{sem_result['rouge1']:.2f}")
                col3.metric("ROUGE-2", f"{sem_result['rouge2']:.2f}")
                col4.metric("ROUGE-L", f"{sem_result['rougeL']:.2f}")

                sem_risk_cls = sem_result["hallucination_risk"].lower()
                st.markdown(
                    f'<div style="margin-top:10px"><span class="badge-{sem_risk_cls}">'
                    f'{sem_result["verdict"]}</span></div>',
                    unsafe_allow_html=True,
                )
                st.info(f"**Explanation:** {sem_result['explanation']}")
                semantic_score_val = sem_result["semantic_similarity"]
            else:
                st.info("ℹ️ No reference answer provided. Skipping semantic comparison. "
                        "Provide a reference answer above to enable this check.")

            # ── COMPOSITE SCORE ───────────────────────────────────────────────
            st.divider()
            st.markdown("## 📊 Overall Hallucination Risk Assessment")

            composite = compute_composite_score(
                semantic_score=semantic_score_val,
                consistency_score=consistency_score_val,
                grounding_score=grounding_score_val,
            )

            composite_pct = int(composite["composite_score"] * 100)
            risk_cls = composite["hallucination_risk"].lower()

            st.markdown(
                f"""
                <div style="
                    background: #1e2130;
                    border-radius: 12px;
                    padding: 24px;
                    text-align: center;
                    border: 2px solid {'#4ade80' if risk_cls=='low' else '#facc15' if risk_cls=='medium' else '#f87171'};
                    margin: 20px 0;
                ">
                    <div style="font-size:52px;">{composite['badge']}</div>
                    <div style="font-size:28px; font-weight:700; margin:10px 0;">
                        {composite['verdict']}
                    </div>
                    <div style="font-size:18px; color:#94a3b8;">
                        Composite Confidence Score: <b>{composite_pct}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Breakdown chart
            import plotly.graph_objects as go
            breakdown = composite["breakdown"]
            fig = go.Figure(go.Bar(
                x=["Semantic Similarity", "Self-Consistency", "RAG Grounding"],
                y=[
                    breakdown["semantic_similarity"],
                    breakdown["self_consistency"],
                    breakdown["rag_grounding"],
                ],
                marker_color=["#6366f1", "#22d3ee", "#4ade80"],
                text=[f"{v:.2f}" for v in breakdown.values()],
                textposition="auto",
            ))
            fig.update_layout(
                title="Score Breakdown by Detection Method",
                yaxis=dict(range=[0, 1], title="Score"),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="#0f1117",
                paper_bgcolor="#0f1117",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Final summary table
            st.markdown("### 📋 Detection Summary")
            summary_df = pd.DataFrame([
                {
                    "Method": "1️⃣ Self-Consistency",
                    "Score": f"{int(consistency_score_val*100)}%",
                    "Risk": consistency_result["hallucination_risk"],
                    "Verdict": consistency_result["verdict"],
                },
                {
                    "Method": "2️⃣ RAG Grounding",
                    "Score": f"{int(grounding_score_val*100)}%",
                    "Risk": rag_result.get("hallucination_risk", "N/A") if context_text else "Skipped",
                    "Verdict": rag_result.get("verdict", "No context provided") if context_text else "—",
                },
                {
                    "Method": "3️⃣ Semantic Similarity",
                    "Score": f"{int(semantic_score_val*100)}%",
                    "Risk": sem_result.get("hallucination_risk", "N/A") if reference_answer else "Skipped",
                    "Verdict": sem_result.get("verdict", "No reference provided") if reference_answer else "—",
                },
                {
                    "Method": "🏆 Composite Score",
                    "Score": f"{composite_pct}%",
                    "Risk": composite["hallucination_risk"],
                    "Verdict": composite["verdict"],
                },
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — METHODS EXPLAINED
# ────────────────────────────────────────────────────────────────────────────
with tab_methods:
    st.title("📐 Methods Deep Dive")

    method_tabs = st.tabs([
        "1️⃣ Self-Consistency",
        "2️⃣ RAG Verification",
        "3️⃣ Semantic Scoring",
        "4️⃣ Heuristics",
    ])

    with method_tabs[0]:
        st.markdown(
            """
            ## Self-Consistency Checking

            ### Intuition
            If a model truly "knows" the answer, asking the same question multiple times
            should produce similar answers. **Divergence = uncertainty = potential hallucination**.

            ### Algorithm
            ```
            1. Query LLM(prompt, T=0.7) → response_1
            2. Query LLM(prompt, T=0.7) → response_2
            3. Query LLM(prompt, T=0.7) → response_3
            4. Compute pairwise similarity: sim(r1,r2), sim(r1,r3), sim(r2,r3)
            5. Consistency score = mean(pairwise_similarities)
            6. If consistency < 0.45 → HIGH HALLUCINATION RISK
            ```

            ### Why Temperature > 0?
            Temperature controls randomness. With T=0, the model always gives the same answer
            (deterministic), making consistency checks useless. We need T > 0 so natural variation
            can reveal when the model is uncertain.

            ### Similarity Metrics Used
            | Method | Description | Requires |
            |--------|-------------|---------|
            | **Jaccard** | Word overlap ratio | Nothing (default) |
            | **SBERT Cosine** | Semantic embedding similarity | `sentence-transformers` |

            ### Research Reference
            Wang et al. (2023) — *"Self-Consistency Improves Chain of Thought Reasoning in LLMs"*
            [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
            """
        )

    with method_tabs[1]:
        st.markdown(
            """
            ## RAG-Based Grounding Verification

            ### Intuition
            If a model's claim can be **found (or inferred) from a trusted knowledge source**,
            it's grounded. If not → possible hallucination.

            ### Algorithm
            ```
            1. Split response into individual sentences
            2. Split knowledge base into sentences
            3. For each response sentence:
               a. Embed it and all context sentences with SBERT
               b. Find the most similar context sentence
               c. If max_similarity >= threshold → SUPPORTED
               d. Else → UNSUPPORTED (potential hallucination)
            4. Grounding score = supported / total sentences
            ```

            ### NLI Extension
            Natural Language Inference (NLI) goes further:
            ```
            For each (context, response_sentence) pair:
              - ENTAILMENT   → Context supports the claim ✅
              - NEUTRAL      → Context neither supports nor contradicts ⚪
              - CONTRADICTION → Context contradicts the claim 🚨
            ```

            ### Why This Works
            - Grounded responses come from real information in the context
            - Hallucinated claims have no matching evidence in any context sentence
            - Contradictions are detected when the model says the opposite of what the context says

            ### Research Reference
            Lewis et al. (2020) — *"Retrieval-Augmented Generation for Knowledge-Intensive NLP"*
            [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
            """
        )

    with method_tabs[2]:
        st.markdown(
            """
            ## Semantic Similarity Scoring

            ### Intuition
            Compare the LLM's response to a known-correct reference answer.
            **High similarity → likely correct. Low similarity → likely hallucinated.**

            ### Metrics Used

            #### Cosine Similarity (Semantic)
            ```
            sim(response, reference) = cos(embed(response), embed(reference))
                                     = (A · B) / (|A| × |B|)
            ```
            Range: -1 to 1 (we use 0 to 1 for NLP). Values > 0.65 indicate strong match.

            #### ROUGE Scores
            ROUGE (Recall-Oriented Understudy for Gisting Evaluation):

            | Metric | Formula | What it measures |
            |--------|---------|-----------------|
            | **ROUGE-1** | Unigram (word) overlap | Basic word coverage |
            | **ROUGE-2** | Bigram overlap | Phrase-level accuracy |
            | **ROUGE-L** | Longest Common Subsequence | Structural similarity |

            #### Combined Score
            ```
            combined = 0.6 × semantic_similarity + 0.4 × ROUGE-L
            ```
            - **≥ 0.65** → Low risk
            - **0.35–0.65** → Medium risk
            - **< 0.35** → High risk

            ### Limitation
            Requires a reference answer — not always available in practice.
            Use RAG verification when no reference exists.
            """
        )

    with method_tabs[3]:
        st.markdown(
            """
            ## Heuristic Signals

            ### Intuition
            Certain linguistic patterns correlate with hallucination:
            - **Hedging phrases** → model is uncertain
            - **Specific claims** (names, dates, statistics) → more likely to be fabricated

            ### Hedging Phrases Detected
            ```
            "I think", "I believe", "probably", "possibly", "as far as I know",
            "to the best of my knowledge", "I'm not sure", "it might be", ...
            ```
            When a model hedges, it's signaling uncertainty — which often accompanies hallucination.

            ### Specific Claim Patterns
            ```
            Years (e.g., 2019, 1987)
            Large numbers (e.g., "2.3 million", "45 billion")
            Citations ("according to research", "studies show")
            Proper names ("John Smith", "University of XYZ")
            ```
            Specific details are harder to get right and more commonly fabricated.

            ### Heuristic Risk Score
            ```
            hedge_score     = min(hedge_count / 5, 1.0)
            indicator_score = min(indicator_count / 10, 1.0)
            heuristic_risk  = 0.4 × hedge_score + 0.6 × indicator_score
            ```

            ### Limitation
            Heuristics are approximate — high hedging doesn't always mean hallucination,
            and confident-sounding text can still be wrong.
            Use as a supplementary signal, not the sole metric.
            """
        )


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — METHOD COMPARISON
# ────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.title("📊 Method Comparison & Analysis")

    st.markdown(
        """
        ### When to Use Which Method?

        Each detection technique has strengths and weaknesses. Here's a guide:
        """
    )

    comparison_data = {
        "Method": [
            "Self-Consistency",
            "RAG Verification",
            "Semantic Similarity",
            "Heuristics",
        ],
        "Requires LLM": ["Yes (multiple calls)", "No", "No", "No"],
        "Requires Reference": ["No", "Knowledge Base", "Reference Answer", "No"],
        "Speed": ["Slow (N×LLM calls)", "Fast", "Fast", "Instant"],
        "Accuracy": ["High", "High (with good KB)", "High (with reference)", "Low–Medium"],
        "Best For": [
            "Factual Q&A, open-domain",
            "Document-grounded tasks",
            "Evaluation, benchmarking",
            "Quick screening",
        ],
        "Limitation": [
            "Expensive (N API calls)",
            "Needs relevant knowledge base",
            "Needs ground-truth reference",
            "Approximate only",
        ],
    }

    df_compare = pd.DataFrame(comparison_data)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📈 Hallucination Risk Thresholds")

    import plotly.graph_objects as go
    fig = go.Figure()

    categories = ["0.0–0.44 (High Risk)", "0.45–0.74 (Medium Risk)", "0.75–1.0 (Low Risk)"]
    values = [0.44, 0.30, 0.26]
    colors = ["#f87171", "#facc15", "#4ade80"]

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=["🚨 Hallucination Likely", "⚠️ Uncertain", "✅ Likely Factual"],
        textposition="auto",
    ))

    fig.update_layout(
        title="Risk Classification Bands (applies to all methods)",
        yaxis_title="Score Range Width",
        height=350,
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 🔬 Research Papers & References")
    st.markdown(
        """
        | Paper | Method | Year | Link |
        |-------|--------|------|------|
        | "Self-Consistency Improves CoT Reasoning" | Self-Consistency | 2023 | [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) |
        | "Retrieval-Augmented Generation" (RAG) | RAG Verification | 2020 | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
        | "SelfCheckGPT" | Zero-Resource Hallucination Detection | 2023 | [arXiv:2303.08896](https://arxiv.org/abs/2303.08896) |
        | "ROUGE: A Package for Automatic Evaluation" | ROUGE Scoring | 2004 | [aclanthology](https://aclanthology.org/W04-1013/) |
        | "Sentence-BERT: Sentence Embeddings" | Semantic Similarity | 2019 | [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) |
        | "FActScoring" | Fine-grained Hallucination Evaluation | 2023 | [arXiv:2305.14251](https://arxiv.org/abs/2305.14251) |
        | "Attention Is All You Need" | Transformer Architecture | 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
        """
    )


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — ABOUT
# ────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ## LLM Hallucination Detection & Verification
        ### A Walkthrough Project for Generative AI

        ---

        ### 🎓 Project Overview
        This project demonstrates three state-of-the-art techniques for detecting
        hallucinations in Large Language Model outputs, implemented as an interactive
        Streamlit web application.

        ### 🛠️ Tech Stack
        | Component | Technology |
        |-----------|------------|
        | **Web UI** | Streamlit |
        | **LLM (Local)** | Ollama (llama3.2, mistral, gemma2, etc.) |
        | **LLM (Free Cloud)** | HuggingFace Inference API |
        | **Semantic Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
        | **NLI** | cross-encoder/nli-deberta-v3-small |
        | **N-gram Scoring** | rouge-score |
        | **Visualization** | Plotly |
        | **Data** | Pandas, NumPy |

        ### 📁 Project Structure
        ```
        LLM Hallucination Detection and Verification/
        ├── app.py                    ← Main Streamlit application
        ├── requirements.txt          ← Python dependencies
        ├── modules/
        │   ├── __init__.py
        │   ├── llm_client.py         ← Ollama + HuggingFace API interface
        │   ├── self_consistency.py   ← Technique 1: Self-consistency
        │   ├── rag_verifier.py       ← Technique 2: RAG-based verification
        │   └── semantic_scorer.py    ← Technique 3: Semantic + ROUGE scoring
        └── sample_docs/
            └── ai_facts.txt          ← Sample knowledge base
        ```

        ### 🚀 Quick Start
        ```bash
        # Install dependencies
        pip install -r requirements.txt

        # (Optional) Start Ollama for local LLM
        ollama pull llama3.2
        ollama serve

        # Launch the app
        streamlit run app.py
        ```

        ### 🔮 Future Extensions
        - **FEVER-style fact checking** against Wikipedia
        - **Entity verification** with knowledge graphs (Wikidata)
        - **Multi-document RAG** with vector databases (FAISS, ChromaDB)
        - **LLM-as-judge** using GPT-4 / Claude for evaluation
        - **Fine-tuned hallucination classifier** (trained on TruthfulQA, HaluEval)
        - **Batch evaluation mode** for comparing multiple models

        ---
        *Built for the Generative AI subject — Academic Project*
        """
    )


# ════════════════════════════════════════════════════════════════════════════
# EXTENSION TAB 1 — WIKIPEDIA FACT CHECKING
# ════════════════════════════════════════════════════════════════════════════
with tab_wiki:
    st.title("🌐 FEVER-Style Wikipedia Fact Checking")
    st.markdown(
        """
        > **How it works:** Each sentence of the LLM response is compared against
        > Wikipedia articles. Claims supported by Wikipedia are flagged ✅, others ⚪.
        > Uses the free Wikipedia REST API — no API key needed.
        """
    )

    st.divider()
    wiki_prompt = st.text_area(
        "Enter prompt / question:",
        height=80,
        placeholder="e.g. When was ChatGPT released and by whom?",
        key="wiki_prompt",
    )
    wiki_response = st.text_area(
        "Paste or enter LLM response to fact-check:",
        height=150,
        placeholder="Paste the LLM's response here...",
        key="wiki_response",
    )
    wiki_max_articles = st.slider("Wikipedia articles to check per claim:", 1, 3, 2, key="wiki_articles")

    if st.button("🔍 Run Wikipedia Fact Check", type="primary", use_container_width=True, key="wiki_btn"):
        if not wiki_response.strip():
            st.error("Please paste the LLM response to fact-check.")
        else:
            with st.spinner("🌐 Querying Wikipedia..."):
                wiki_result = fever_check_response(wiki_response, max_articles=wiki_max_articles)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Sentences", wiki_result["total_sentences"])
            col2.metric("Wikipedia-Supported", wiki_result["supported_count"])
            col3.metric("Insufficient Info", wiki_result["insufficient_count"])
            col4.metric("Risk", wiki_result["hallucination_risk"])

            risk_cls = wiki_result["risk_color"]
            st.markdown(
                f'<div style="margin:10px 0"><span class="badge-{wiki_result["hallucination_risk"].lower()}">'
                f'{wiki_result["overall_verdict"]}</span></div>',
                unsafe_allow_html=True,
            )

            st.divider()
            st.markdown("### 📄 Sentence-Level Wikipedia Verdicts")
            for sr in wiki_result["sentence_results"]:
                color_map = {"green": "sent-supported", "orange": "sent-neutral", "gray": "sent-neutral"}
                css_cls = "sent-supported" if sr["verdict"] == "SUPPORTED" else \
                          "sent-unsupported" if sr["verdict"] == "CONTRADICTED" else "sent-neutral"
                st.markdown(
                    f'<div class="{css_cls}">'
                    f'<b>{sr["verdict_label"]}</b>'
                    f'{"  (overlap: " + str(sr.get("overlap_score","")) + ")" if "overlap_score" in sr else ""}<br>'
                    f'<i>"{sr["claim"]}"</i>',
                    unsafe_allow_html=True,
                )
                if sr.get("sources"):
                    for s in sr["sources"][:1]:
                        st.markdown(
                            f'<small>📖 Source: <a href="{s["url"]}" target="_blank">{s["title"]}</a>'
                            f' — {s.get("description","")}</small>',
                            unsafe_allow_html=True,
                        )
                st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# EXTENSION TAB 2 — ENTITY VERIFICATION (WIKIDATA)
# ════════════════════════════════════════════════════════════════════════════
with tab_wiki_ent:
    st.title("🏷️ Entity Verification via Wikidata")
    st.markdown(
        """
        > **How it works:** Named entities (people, organizations, places) are extracted
        > from the LLM response and looked up in the **Wikidata knowledge graph**.
        > Fabricated entities (people/orgs that don't exist) are flagged.
        """
    )

    st.divider()
    ent_response = st.text_area(
        "Paste LLM response for entity verification:",
        height=150,
        placeholder="e.g. 'GPT-4 was developed by OpenAI and released in March 2023...'",
        key="ent_response",
    )

    if st.button("🔍 Verify Entities via Wikidata", type="primary", use_container_width=True, key="ent_btn"):
        if not ent_response.strip():
            st.error("Please paste a response to verify.")
        else:
            with st.spinner("🏷️ Querying Wikidata knowledge graph..."):
                ent_result = verify_entities_in_response(ent_response)

            col1, col2, col3 = st.columns(3)
            col1.metric("Entities Found", ent_result["total_entities"])
            col2.metric("Verified in Wikidata", ent_result["verified_count"])
            col3.metric("Not Found", ent_result["not_found_count"])

            st.markdown(
                f'<div style="margin:10px 0">'
                f'<span class="badge-{ent_result["hallucination_risk"].lower()}">'
                f'{ent_result["overall_verdict"]}</span></div>',
                unsafe_allow_html=True,
            )

            if not ent_result["entity_results"]:
                st.info("No named entities detected in the response.")
            else:
                st.divider()
                st.markdown("### 🏷️ Entity Verification Results")
                for er in ent_result["entity_results"]:
                    css_cls = "sent-supported" if er["status"] == "VERIFIED" else \
                              "sent-neutral" if er["status"] == "PARTIAL_MATCH" else "sent-unsupported"
                    st.markdown(
                        f'<div class="{css_cls}">'
                        f'<b>{er["label"]}</b> — <code>{er["entity"]}</code><br>'
                        f'<i>{er.get("description", "")}</i>',
                        unsafe_allow_html=True,
                    )
                    if er.get("wikidata_url"):
                        st.markdown(
                            f'<small>🔗 <a href="{er["wikidata_url"]}" target="_blank">'
                            f'Wikidata: {er.get("wikidata_label","")}</a></small>',
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# EXTENSION TAB 3 — MULTI-DOCUMENT RAG (FAISS / NUMPY)
# ════════════════════════════════════════════════════════════════════════════
with tab_vrag:
    st.title("📚 Multi-Document RAG Vector Database")
    st.markdown(
        """
        > **How it works:** Upload multiple documents. They are chunked and indexed in a
        > **vector database** (FAISS if installed, else NumPy). Query the LLM response
        > against the full multi-document knowledge base for grounding.
        """
    )

    # Initialize vector store in session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorRAGStore()

    store = st.session_state.vector_store

    st.divider()
    st.markdown("### 📁 Step 1 — Build Your Knowledge Base")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Upload .txt documents (multiple allowed):",
            type=["txt"],
            accept_multiple_files=True,
            key="vrag_upload",
        )
        paste_doc = st.text_area("Or paste a document:", height=100, key="vrag_paste")
        paste_name = st.text_input("Document name:", value="pasted_doc", key="vrag_name")

    with col2:
        stats = store.get_stats()
        st.metric("Documents Indexed", stats["total_documents"])
        st.metric("Total Chunks", stats["total_chunks"])
        st.caption(f"Embedding: {stats['embedding_method']}")
        st.caption(f"FAISS: {'✅ Available' if stats['faiss_available'] else '❌ Using NumPy'}")

        if stats["sources"]:
            st.markdown("**Indexed sources:**")
            for src in stats["sources"]:
                st.markdown(f"- `{src}`")

    col_add1, col_add2, col_clear = st.columns(3)
    with col_add1:
        if st.button("➕ Add Uploaded Files", key="vrag_add_files"):
            if uploaded_files:
                for f in uploaded_files:
                    text = f.read().decode("utf-8")
                    n = store.add_document(f.name, text)
                    st.success(f"Added `{f.name}` — {n} chunks")
    with col_add2:
        if st.button("➕ Add Pasted Doc", key="vrag_add_paste"):
            if paste_doc.strip():
                n = store.add_document(paste_name, paste_doc)
                st.success(f"Added `{paste_name}` — {n} chunks")
    with col_clear:
        if st.button("🗑️ Clear All Documents", key="vrag_clear"):
            st.session_state.vector_store = VectorRAGStore()
            st.rerun()

    # Add sample AI facts doc
    if st.button("📄 Load Sample AI Facts Document", key="vrag_sample"):
        sample_path = os.path.join(os.path.dirname(__file__), "sample_docs", "ai_facts.txt")
        try:
            with open(sample_path) as f:
                text = f.read()
            n = store.add_document("ai_facts.txt", text)
            st.success(f"Loaded sample AI facts — {n} chunks")
        except FileNotFoundError:
            st.error("Sample file not found.")

    st.divider()
    st.markdown("### 🔬 Step 2 — Verify LLM Response")

    vrag_response = st.text_area("LLM Response to verify:", height=120, key="vrag_response")
    vrag_k = st.slider("Top chunks to retrieve:", 1, 10, 3, key="vrag_k")
    vrag_thresh = st.slider("Similarity threshold:", 0.1, 0.9, 0.35, step=0.05, key="vrag_thresh")

    if st.button("🔍 Verify Against Vector DB", type="primary", use_container_width=True, key="vrag_btn"):
        if not vrag_response.strip():
            st.error("Please enter a response to verify.")
        elif store.get_stats()["total_chunks"] == 0:
            st.error("Please add documents to the knowledge base first.")
        else:
            with st.spinner("Searching vector database..."):
                vrag_result = store.verify_response(vrag_response, k=vrag_k, threshold=vrag_thresh)

            col1, col2, col3 = st.columns(3)
            col1.metric("Grounding Score", f"{int(vrag_result['grounding_score']*100)}%")
            col2.metric("Supported", f"{vrag_result['supported_count']}/{vrag_result['total_sentences']}")
            col3.metric("Risk", vrag_result["hallucination_risk"])

            st.markdown(
                f'<div class="badge-{vrag_result["hallucination_risk"].lower()}">'
                f'{vrag_result["verdict"]}</div>',
                unsafe_allow_html=True,
            )
            st.divider()
            st.markdown("### 📄 Sentence Grounding Results")
            for sr in vrag_result["sentence_results"]:
                css_cls = "sent-supported" if sr["supported"] else "sent-unsupported"
                st.markdown(
                    f'<div class="{css_cls}">'
                    f'<b>{sr["label"]}</b> (sim: {sr["best_similarity"]:.2f})<br>'
                    f'<i>"{sr["sentence"]}"</i><br>'
                    f'<small>Source: <code>{sr["best_source"]}</code> — '
                    f'"{sr.get("best_chunk","")[:100]}..."</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# EXTENSION TAB 4 — LLM-AS-JUDGE
# ════════════════════════════════════════════════════════════════════════════
with tab_judge:
    st.title("⚖️ LLM-as-Judge Evaluation")
    st.markdown(
        """
        > **How it works:** A second LLM (the "judge") evaluates the first LLM's response
        > for factuality, relevance, and coherence. It outputs a structured JSON score
        > and flags any specific issues. Based on **MT-Bench** methodology.
        >
        > 🧠 Uses your local Ollama model as the judge — no extra API key needed!
        """
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        judge_question = st.text_area("Original Question:", height=80, key="judge_q",
                                      placeholder="What question was asked to the LLM?")
        judge_response = st.text_area("LLM Response to Evaluate:", height=120, key="judge_resp",
                                      placeholder="Paste the LLM response here...")
    with col2:
        judge_context = st.text_area("Reference Context (optional):", height=100, key="judge_ctx",
                                     placeholder="Paste any ground-truth context for the judge to use...")
        if backend == "Ollama (Local)" and ollama_ok:
            models = get_ollama_models()
            judge_model_sel = st.selectbox("Judge model:", models if models else ["llama3.2"], key="judge_model")
        else:
            judge_model_sel = "llama3.2"
            st.info("Ollama not running — judge uses HuggingFace backend")

    if st.button("⚖️ Run LLM Judge Evaluation", type="primary", use_container_width=True, key="judge_btn"):
        if not judge_question.strip() or not judge_response.strip():
            st.error("Please enter both the question and LLM response.")
        elif backend == "Ollama (Local)" and not ollama_ok:
            st.error("Ollama is not running. Please start it first.")
        else:
            with st.spinner("⚖️ Running LLM judge evaluation..."):
                judge_result = evaluate_with_llm_judge(
                    question=judge_question,
                    response=judge_response,
                    context=judge_context,
                    judge_model=judge_model_sel,
                    judge_backend="ollama" if (backend == "Ollama (Local)" and ollama_ok) else "huggingface",
                    temperature=0.1,
                )

            # Overall verdict banner
            composite = judge_result["composite_score_10"]
            st.markdown(
                f"""
                <div style="background:#1e2130;border-radius:12px;padding:20px;text-align:center;
                    border:2px solid {'#4ade80' if judge_result['hallucination_risk']=='Low' else '#facc15' if judge_result['hallucination_risk']=='Medium' else '#f87171'};margin:15px 0;">
                    <div style="font-size:40px;">{judge_result['badge']}</div>
                    <div style="font-size:22px;font-weight:700;">{judge_result['overall_verdict']}</div>
                    <div style="color:#94a3b8;">Judge confidence: {judge_result['judge_confidence']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Factuality Score", f"{judge_result['factuality_score']}/10")
            col2.metric("Quality Score", f"{judge_result['quality_score']}/10")
            col3.metric("Composite", f"{composite}/10")
            col4.metric("Risk", judge_result["hallucination_risk"])

            fact = judge_result["factuality"]
            rel = judge_result["relevance"]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔍 Factuality Analysis")
                st.metric("Hallucination Detected", "YES 🚨" if fact.get("hallucination_detected") else "NO ✅")
                st.markdown(f"**Verdict:** {fact.get('verdict', '')}")
                if fact.get("specific_issues"):
                    st.markdown("**Issues found:**")
                    for issue in fact["specific_issues"]:
                        st.markdown(f"- ⚠️ {issue}")
            with col2:
                st.markdown("#### 📊 Quality Analysis")
                st.metric("Relevance", f"{rel.get('relevance_score',0)}/10")
                st.metric("Completeness", f"{rel.get('completeness_score',0)}/10")
                st.metric("Coherence", f"{rel.get('coherence_score',0)}/10")
                if rel.get("feedback"):
                    st.info(f"**Feedback:** {rel['feedback']}")

            with st.expander("🔧 View Raw Judge Output"):
                st.text(f"Factuality raw:\n{judge_result['raw_factuality']}\n\nRelevance raw:\n{judge_result['raw_relevance']}")


# ════════════════════════════════════════════════════════════════════════════
# EXTENSION TAB 5 — HALLUCINATION CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════
with tab_classifier:
    st.title("🤖 Hallucination Classifier")
    st.markdown(
        """
        > **How it works:** Combines rule-based heuristics (hedging phrases, fabricated citations,
        > overconfident numbers) with optional **zero-shot NLI classification** to assign a
        > hallucination probability score. Inspired by **TruthfulQA** and **HaluEval** benchmarks.
        """
    )

    st.divider()
    clf_response = st.text_area(
        "Paste LLM response to classify:",
        height=180,
        key="clf_response",
        placeholder="Paste any LLM output here to classify its hallucination risk...",
    )
    use_zs = st.toggle("Use Zero-Shot NLI classifier (requires `transformers` + slow)", value=False, key="clf_zs")

    if st.button("🤖 Run Classifier", type="primary", use_container_width=True, key="clf_btn"):
        if not clf_response.strip():
            st.error("Please paste a response to classify.")
        else:
            with st.spinner("Running hallucination classifier..."):
                clf_result = run_full_classifier(clf_response, use_zero_shot=use_zs)

            heuristic = clf_result["heuristic"]
            combined_score = clf_result["combined_hallucination_score"]

            # Overall banner
            st.markdown(
                f'<div class="badge-{clf_result["final_risk"].lower()}" style="font-size:18px;padding:10px 16px;">'
                f'{clf_result["final_verdict"]}</div>',
                unsafe_allow_html=True,
            )

            st.divider()
            st.markdown("### 📊 Heuristic Analysis (Rule-Based)")

            col1, col2, col3 = st.columns(3)
            col1.metric("Hallucination Score", f"{int(heuristic['hallucination_score']*100)}%")
            col2.metric("Risk Level", heuristic["hallucination_risk"])
            col3.metric("Factual Signals", heuristic["factual_signals_found"])

            import plotly.graph_objects as go
            cat_scores = heuristic["category_scores"]
            fig = go.Figure(go.Bar(
                x=list(cat_scores.keys()),
                y=list(cat_scores.values()),
                marker_color=["#f87171", "#facc15", "#fb923c", "#c084fc", "#60a5fa"],
                text=list(cat_scores.values()),
                textposition="auto",
            ))
            fig.update_layout(
                title="Hallucination Signal Category Scores",
                height=300,
                plot_bgcolor="#0f1117",
                paper_bgcolor="#0f1117",
                font_color="#e2e8f0",
                xaxis_tickangle=-15,
            )
            st.plotly_chart(fig, use_container_width=True)

            if heuristic["triggers_found"]:
                st.markdown("**⚠️ Triggers detected:**")
                for t in heuristic["triggers_found"]:
                    st.markdown(f"- `{t}`")
            else:
                st.success("No hallucination triggers detected.")

            if use_zs and "zero_shot" in clf_result:
                st.divider()
                st.markdown("### 🧠 Zero-Shot NLI Classification")
                zs = clf_result["zero_shot"]
                if "error" in zs:
                    st.error(zs["error"])
                else:
                    st.markdown(f"**Result:** {zs.get('verdict','')}")
                    if zs.get("all_labels"):
                        for label, score in zip(zs["all_labels"], zs["all_scores"]):
                            st.progress(float(score), text=f"{label}: {score:.0%}")


# ════════════════════════════════════════════════════════════════════════════
# EXTENSION TAB 6 — BATCH EVALUATION
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.title("📋 Batch Evaluation Mode")
    st.markdown(
        """
        > **How it works:** Run hallucination detection on **multiple prompts at once**.
        > Compare results across prompts (or across different models) and get aggregate statistics.
        > Perfect for benchmarking reliability of a model.
        """
    )

    st.divider()
    st.markdown("### 📝 Step 1 — Define Prompts")

    default_prompts = """When was ChatGPT released?
Who invented the transformer architecture?
What is retrieval-augmented generation?
What did Einstein win the Nobel Prize for?
Who invented the ReasonForge algorithm in 2018?"""

    batch_prompts_raw = st.text_area(
        "Enter prompts (one per line):",
        value=default_prompts,
        height=150,
        key="batch_prompts",
    )

    st.markdown("### ⚙️ Step 2 — Configure Models")

    col1, col2 = st.columns(2)
    with col1:
        batch_models_input = []
        if backend == "Ollama (Local)" and ollama_ok:
            available = get_ollama_models()
            selected_batch_models = st.multiselect(
                "Select Ollama models to compare:",
                options=available if available else ["llama3.2"],
                default=available[:1] if available else ["llama3.2"],
                key="batch_model_select",
            )
            for m in selected_batch_models:
                batch_models_input.append({"model": m, "backend": "ollama"})
        else:
            st.info("Ollama not running — using HuggingFace backend")
            batch_models_input = [{"model": "hf_model", "backend": "huggingface",
                                   "hf_model": "microsoft/Phi-3-mini-4k-instruct"}]

    with col2:
        batch_context = st.text_area(
            "Reference context (optional, applied to all prompts):",
            height=100,
            key="batch_context",
        )
        batch_samples = st.slider("Consistency samples:", 2, 3, 2, key="batch_samples")

    run_batch_btn = st.button("🚀 Run Batch Evaluation", type="primary", use_container_width=True, key="batch_btn")

    if run_batch_btn:
        prompts_list = [p.strip() for p in batch_prompts_raw.strip().split("\n") if p.strip()]
        if not prompts_list:
            st.error("Please enter at least one prompt.")
        elif not batch_models_input:
            st.error("Please select at least one model.")
        elif backend == "Ollama (Local)" and not ollama_ok:
            st.error("Ollama is not running.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_evals = len(prompts_list) * len(batch_models_input)
            done_count = [0]

            def batch_progress(msg):
                done_count[0] = min(done_count[0] + 1, total_evals)
                progress_bar.progress(done_count[0] / max(total_evals, 1))
                status_text.text(msg)

            with st.spinner(f"Running {total_evals} evaluations..."):
                batch_result = run_batch_evaluation(
                    prompts=prompts_list,
                    models=batch_models_input,
                    context=batch_context,
                    num_samples=batch_samples,
                    temperature=temperature,
                    hf_token=hf_token if backend == "HuggingFace API (Free)" else "",
                    progress_callback=batch_progress,
                )

            progress_bar.progress(1.0)
            status_text.text("✅ Batch evaluation complete!")

            summary = batch_result["summary"]
            df = batch_result["df"]

            st.divider()
            st.markdown("### 📊 Batch Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Evaluations", summary["total_evaluations"])
            col2.metric("Avg Composite Score", f"{summary['avg_composite_score']:.0%}")
            col3.metric("High Risk Count", summary["high_risk_count"])
            col4.metric("Low Risk Count", summary["low_risk_count"])

            # Risk distribution chart
            import plotly.graph_objects as go
            risk_counts = [summary["low_risk_count"], summary["medium_risk_count"], summary["high_risk_count"]]
            fig_pie = go.Figure(go.Pie(
                labels=["Low Risk ✅", "Medium Risk ⚠️", "High Risk 🚨"],
                values=risk_counts,
                marker_colors=["#4ade80", "#facc15", "#f87171"],
                hole=0.4,
            ))
            fig_pie.update_layout(
                title="Risk Distribution",
                height=300,
                paper_bgcolor="#0f1117",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("### 📋 Detailed Results Table")
            display_cols = ["prompt", "model", "composite_score", "consistency_score",
                            "grounding_score", "hallucination_risk", "badge", "hedge_count"]
            display_df = df[[c for c in display_cols if c in df.columns]].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Show best and worst
            if len(df) > 1:
                st.markdown("### 🏆 Best vs Worst Responses")
                best_idx = df["composite_score"].idxmax()
                worst_idx = df["composite_score"].idxmin()
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Best Response** (score: {df.loc[best_idx,'composite_score']:.0%})")
                    st.markdown(f"**Prompt:** {df.loc[best_idx,'prompt']}")
                    with st.expander("View response"):
                        st.markdown(f"> {df.loc[best_idx,'response']}")
                with col2:
                    st.error(f"**Riskiest Response** (score: {df.loc[worst_idx,'composite_score']:.0%})")
                    st.markdown(f"**Prompt:** {df.loc[worst_idx,'prompt']}")
                    with st.expander("View response"):
                        st.markdown(f"> {df.loc[worst_idx,'response']}")

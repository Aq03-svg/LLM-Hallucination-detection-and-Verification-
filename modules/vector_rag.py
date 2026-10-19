# ─────────────────────────────────────────────────────────────────────────────
# Module: vector_rag.py
# Purpose: Multi-Document RAG with FAISS Vector Database
#
# Technique: Build a persistent vector index from multiple documents,
# then retrieve the most relevant chunks for any query and verify
# LLM responses against retrieved evidence.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import pickle
import numpy as np
from typing import Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks for embedding.

    Args:
        text: Input document text
        chunk_size: Approximate words per chunk
        overlap: Words to overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk.strip())
    return chunks


# ── Embedding (SBERT or TF-IDF fallback) ─────────────────────────────────────

class TFIDFEmbedder:
    """
    Lightweight TF-IDF based embedder (no model download needed).
    Falls back to this when sentence-transformers is unavailable.
    """

    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.fitted = False

    def fit(self, texts: list[str]):
        from math import log
        N = len(texts)
        df = {}
        for text in texts:
            words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
            for w in words:
                df[w] = df.get(w, 0) + 1
        self.vocab = {w: i for i, w in enumerate(sorted(df.keys()))}
        self.idf = {w: log(N / (df[w] + 1)) for w in df}
        self.fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            self.fit(texts)
        V = len(self.vocab)
        matrix = np.zeros((len(texts), V), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.sub(r'[^\w\s]', '', text.lower()).split()
            tf = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1
            for w, count in tf.items():
                if w in self.vocab:
                    j = self.vocab[w]
                    matrix[i][j] = (count / len(words)) * self.idf.get(w, 0)
        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms


# ── Vector Store ─────────────────────────────────────────────────────────────

class VectorRAGStore:
    """
    Multi-document vector store using FAISS (or numpy fallback).

    Usage:
        store = VectorRAGStore()
        store.add_document("doc1.txt", text1)
        store.add_document("doc2.txt", text2)
        results = store.search("What is GPT?", k=5)
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.chunks: list[str] = []
        self.metadata: list[dict] = []   # {source, chunk_id}
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.embedding_model_name = embedding_model
        self._sbert_model = None
        self._tfidf = TFIDFEmbedder()
        self.dim = 384  # MiniLM default
        self.use_sbert = SBERT_AVAILABLE and FAISS_AVAILABLE

    def _get_embedder(self):
        if self.use_sbert:
            if self._sbert_model is None:
                self._sbert_model = SentenceTransformer(self.embedding_model_name)
            return self._sbert_model
        return self._tfidf

    def _embed(self, texts: list[str]) -> np.ndarray:
        embedder = self._get_embedder()
        if self.use_sbert:
            return embedder.encode(texts, show_progress_bar=False)
        else:
            if not self._tfidf.fitted and self.chunks:
                self._tfidf.fit(self.chunks + texts)
            return self._tfidf.encode(texts)

    def add_document(self, source_name: str, text: str, chunk_size: int = 200) -> int:
        """
        Add a document to the vector store.

        Args:
            source_name: Document identifier (filename or title)
            text: Document text content
            chunk_size: Words per chunk

        Returns:
            Number of chunks added
        """
        new_chunks = chunk_text(text, chunk_size=chunk_size)
        for i, chunk in enumerate(new_chunks):
            self.chunks.append(chunk)
            self.metadata.append({"source": source_name, "chunk_id": i})
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self):
        """Rebuild FAISS index from all current chunks."""
        if not self.chunks:
            return

        embeddings = self._embed(self.chunks)
        self.embeddings = embeddings.astype(np.float32)
        self.dim = embeddings.shape[1]

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dim)  # Inner product (cosine after normalize)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = self.embeddings / norms
            self.index.add(normalized)
        # If no FAISS, we'll use numpy dot product in search()

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for the most relevant chunks for a query.

        Args:
            query: Search query
            k: Number of top results to return

        Returns:
            List of dicts with chunk text, source, and similarity score
        """
        if not self.chunks:
            return []

        query_emb = self._embed([query]).astype(np.float32)

        if FAISS_AVAILABLE and self.index is not None:
            norm = np.linalg.norm(query_emb)
            if norm > 0:
                query_emb = query_emb / norm
            k = min(k, len(self.chunks))
            scores, indices = self.index.search(query_emb, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append({
                        "chunk": self.chunks[idx],
                        "source": self.metadata[idx]["source"],
                        "chunk_id": self.metadata[idx]["chunk_id"],
                        "similarity": round(float(score), 4),
                    })
        else:
            # Numpy fallback
            if self.embeddings is None:
                return []
            norms_db = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms_db[norms_db == 0] = 1
            normalized_db = self.embeddings / norms_db
            norm_q = np.linalg.norm(query_emb)
            if norm_q > 0:
                query_emb = query_emb / norm_q
            sims = (normalized_db @ query_emb.T).squeeze()
            if sims.ndim == 0:
                sims = np.array([float(sims)])
            top_k = min(k, len(sims))
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append({
                    "chunk": self.chunks[int(idx)],
                    "source": self.metadata[int(idx)]["source"],
                    "chunk_id": self.metadata[int(idx)]["chunk_id"],
                    "similarity": round(float(sims[int(idx)]), 4),
                })

        return results

    def verify_response(self, response: str, k: int = 3, threshold: float = 0.4) -> dict:
        """
        Verify an LLM response against the vector store.

        For each sentence, retrieve top-k relevant chunks and check support.

        Args:
            response: LLM response to verify
            k: Chunks to retrieve per sentence
            threshold: Similarity threshold for "supported"

        Returns:
            dict with per-sentence verdicts and overall grounding score
        """
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        sentence_results = []
        for sent in sentences:
            top_chunks = self.search(sent, k=k)
            if not top_chunks:
                sentence_results.append({
                    "sentence": sent,
                    "supported": False,
                    "best_similarity": 0.0,
                    "best_chunk": "No documents in store",
                    "best_source": "N/A",
                    "label": "⚪ No data",
                })
                continue

            best = top_chunks[0]
            supported = best["similarity"] >= threshold
            sentence_results.append({
                "sentence": sent,
                "supported": supported,
                "best_similarity": best["similarity"],
                "best_chunk": best["chunk"][:200] + "...",
                "best_source": best["source"],
                "top_sources": [r["source"] for r in top_chunks],
                "label": "✅ Supported" if supported else "🚨 Unsupported",
            })

        n = len(sentence_results)
        supported_count = sum(1 for r in sentence_results if r["supported"])
        grounding = supported_count / n if n > 0 else 0.0

        if grounding >= 0.75:
            risk = "Low"; color = "green"
            verdict = "✅ Response is well-grounded in your documents"
        elif grounding >= 0.45:
            risk = "Medium"; color = "orange"
            verdict = "⚠️ Partially grounded — some claims not supported"
        else:
            risk = "High"; color = "red"
            verdict = "🚨 Poorly grounded — most claims not found in documents"

        return {
            "sentence_results": sentence_results,
            "grounding_score": round(grounding, 4),
            "supported_count": supported_count,
            "total_sentences": n,
            "hallucination_risk": risk,
            "risk_color": color,
            "verdict": verdict,
            "total_chunks_in_store": len(self.chunks),
            "use_faiss": FAISS_AVAILABLE,
            "use_sbert": self.use_sbert,
        }

    def get_stats(self) -> dict:
        """Return store statistics."""
        sources = list(set(m["source"] for m in self.metadata))
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(sources),
            "sources": sources,
            "embedding_method": f"SBERT ({self.embedding_model_name})" if self.use_sbert else "TF-IDF (fallback)",
            "faiss_available": FAISS_AVAILABLE,
        }

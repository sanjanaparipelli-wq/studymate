# app.py
"""
StudyMate — Streamlit Q&A from uploaded PDFs (hackathon-ready)

How to run:
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt
3) (optional) set IBM watsonx env vars:
   WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID
4) streamlit run app.py
"""
from __future__ import annotations
import os
import io
import uuid
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import streamlit as st

# PDF processing
import fitz  # PyMuPDF

# Embeddings & vector search
from sentence_transformers import SentenceTransformer

# faiss (CPU)
try:
    import faiss
except Exception as e:
    faiss = None
    print("faiss import failed:", e)

# Optional IBM watsonx (may not be installed)
try:
    from ibm_watsonx_ai.foundation_models import Model
    from ibm_watsonx_ai import Credentials
except Exception:
    Model = None
    Credentials = None

# ---------------------------
# Data classes
# ---------------------------
@dataclass
class TextChunk:
    text: str
    source: str
    page: int
    chunk_id: str

@dataclass
class RetrievalResult:
    chunk: TextChunk
    score: float

# ---------------------------
# Utilities
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load SentenceTransformer once per session."""
    return SentenceTransformer(model_name)

def extract_text_from_pdf(file_bytes: bytes, source_name: str) -> List[tuple]:
    """Return list of (page_text, page_num) using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        # Normalize/cleanup
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        pages.append((text, i + 1))
    doc.close()
    return pages

def chunk_text(pages: List[tuple], source_name: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[TextChunk]:
    """Char-based sliding window chunking across the document text (fast & simple)."""
    chunks: List[TextChunk] = []
    # Concatenate pages with separators so we can still roughly map back to pages
    page_texts = [p for p, _ in pages]
    page_nums = [n for _, n in pages]
    full_text = "\n\n".join(page_texts)
    if not full_text:
        return []

    start = 0
    L = len(full_text)
    while start < L:
        end = min(start + chunk_size, L)
        segment = full_text[start:end].strip()
        # Heuristic mapping to page: proportion of position
        try:
            frac = start / max(1, L)
            idx = min(len(page_nums) - 1, int(frac * len(page_nums)))
            page_guess = page_nums[idx]
        except Exception:
            page_guess = 1
        chunks.append(TextChunk(text=segment, source=source_name, page=page_guess, chunk_id=str(uuid.uuid4())))
        if end == L:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks

def _normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """L2-normalize row vectors. Avoid dividing by 0."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms

def embed_chunks(chunks: List[TextChunk], embedder: SentenceTransformer) -> np.ndarray:
    texts = [c.text for c in chunks]
    emb = embedder.encode(texts, show_progress_bar=False)
    emb = np.array(emb, dtype="float32")
    emb = _normalize_embeddings(emb)
    return emb

def build_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss is not available. Install faiss-cpu.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)
    return index

def search_index(query: str, embedder: SentenceTransformer, index, chunks: List[TextChunk], k: int = 5) -> List[RetrievalResult]:
    q_emb = embedder.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    q_emb = _normalize_embeddings(q_emb)
    if index is None:
        return []
    scores, idxs = index.search(q_emb, k)
    results: List[RetrievalResult] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append(RetrievalResult(chunk=chunks[int(idx)], score=float(score)))
    return results

def format_prompt(question: str, retrieved: List[RetrievalResult]) -> str:
    parts = []
    for r in retrieved:
        meta = f"Source: {r.chunk.source}, Page: {r.chunk.page}"
        parts.append(f"[{meta}]\n{r.chunk.text}")
    context = "\n\n---\n\n".join(parts) if parts else ""
    system = ("You are StudyMate, an academic assistant. Answer using ONLY the CONTEXT provided. "
              "Cite sources inline like (Source, p. Page). If the answer is not present, say you cannot find it.")
    prompt = f"{system}\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nRESPOND:\n- Be concise and structured.\n- Cite sources for claims."
    return prompt

# ---------------------------
# Watsonx integration (optional)
# ---------------------------
@st.cache_resource(show_spinner=False)
def init_watsonx_model():
    """Initialize watsonx model if credentials and SDK are present. Returns model instance or None."""
    if Model is None or Credentials is None:
        return None
    api_key = os.getenv("WATSONX_API_KEY")
    url = os.getenv("WATSONX_URL")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not (api_key and url and project_id):
        return None
    try:
        creds = Credentials(url=url, api_key=api_key)
        model = Model(
            model_id="mistralai/mixtral-8x7b-instruct-v01",
            credentials=creds,
            project_id=project_id,
            params={"decoding_method": "greedy", "max_new_tokens": 512, "temperature": 0.2},
        )
        return model
    except Exception as e:
        print("watsonx init error:", e)
        return None

def generate_answer_with_watsonx(prompt: str) -> Optional[str]:
    model = init_watsonx_model()
    if model is None:
        return None
    try:
        resp = model.generate_text(prompt)
        # Defensive parsing for different SDK shapes
        if isinstance(resp, dict):
            # common shape: {"results": [{"generated_text": "..."}], ...}
            results = resp.get("results") or resp.get("completions") or []
            if isinstance(results, list) and results:
                first = results[0]
                text = first.get("generated_text") or first.get("text") or first.get("content")
                if text:
                    return text.strip()
            # fallback stringify
            return json.dumps(resp)[:4000]
        if isinstance(resp, str):
            return resp.strip()
        return None
    except Exception as e:
        print("watsonx generation error:", e)
        return None

def fallback_extractive_answer(question: str, retrieved: List[RetrievalResult]) -> str:
    if not retrieved:
        return "No relevant passages found in the index."
    lines = ["**Provisional answer (LLM offline): relevant excerpts below:**", ""]
    for r in retrieved:
        snippet = r.chunk.text.replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        lines.append(f"- {snippet} ({r.chunk.source}, p. {r.chunk.page})")
    return "\n\n".join(lines)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")
st.title("📚 StudyMate — Conversational Q&A from PDFs")
st.write("Upload PDFs, build an index, and ask natural-language questions. Answers cite the source pages.")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size (chars)", 400, 2000, 1000, step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 50, 800, 200, step=10)
    top_k = st.slider("Top-K passages", 1, 10, 5)
    embed_model_name = st.selectbox("Embedding model", [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    ], index=0)
    st.markdown("---")
    st.write("watsonx: detected" if (Model is not None and Credentials is not None) else "watsonx: SDK not installed or not configured")

uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

# session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embedder_name" not in st.session_state:
    st.session_state.embedder_name = None

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Build / Rebuild Index", disabled=(not uploaded_files)):
        if faiss is None:
            st.error("faiss is not installed. Install faiss-cpu and restart.")
        else:
            with st.spinner("Extracting text and building index..."):
                all_chunks = []
                for f in uploaded_files:
                    bytes_data = f.read()
                    pages = extract_text_from_pdf(bytes_data, f.name)
                    file_chunks = chunk_text(pages, f.name, chunk_size, chunk_overlap)
                    all_chunks.extend(file_chunks)
                if not all_chunks:
                    st.warning("No text extracted from uploaded PDFs.")
                else:
                    embedder = load_embedding_model(embed_model_name)
                    embeddings = embed_chunks(all_chunks, embedder)
                    index = build_faiss_index(embeddings)
                    st.session_state.chunks = all_chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.faiss_index = index
                    st.session_state.embedder_name = embed_model_name
                    st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")

with col2:
    if st.session_state.faiss_index is not None:
        st.metric("Indexed chunks", len(st.session_state.chunks))

st.markdown("---")
question = st.text_input("Ask a question about your PDFs", placeholder="e.g., What does theorem 1 say? How is algorithm X analyzed?")
ask_btn = st.button("Ask")

if ask_btn and question.strip():
    if st.session_state.faiss_index is None:
        st.warning("Please upload PDFs and click Build / Rebuild Index first.")
    else:
        embedder = load_embedding_model(st.session_state.embedder_name or embed_model_name)
        with st.spinner("Retrieving top passages..."):
            results = search_index(question, embedder, st.session_state.faiss_index, st.session_state.chunks, k=top_k)
        if not results:
            st.info("No relevant passages found.")
        else:
            st.subheader("Answer")
            prompt = format_prompt(question, results)
            with st.spinner("Generating grounded answer..."):
                llm_ans = generate_answer_with_watsonx(prompt)
            if llm_ans:
                st.markdown(llm_ans)
            else:
                st.markdown(fallback_extractive_answer(question, results))

            st.markdown("**Sources / Evidence**")
            for i, r in enumerate(results, start=1):
                with st.expander(f"{i}. {r.chunk.source} — p. {r.chunk.page} (score={r.score:.3f})"):
                    st.write(r.chunk.text)

st.markdown("---")
st.markdown("Built for hackathons — feel free to adapt the prompt, LLM backend, or embedding model.")

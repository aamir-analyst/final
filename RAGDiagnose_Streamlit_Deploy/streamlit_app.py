import os
import re
from collections import Counter
from typing import List, Dict, Any

import streamlit as st
from pypdf import PdfReader
from groq import Groq

# 1. Page Configuration
st.set_page_config(
    page_title="RAGDiagnose | RAG Evaluation & Diagnosis",
    page_icon="🩺",
    layout="wide",
)

# 2. Custom CSS styling
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#14213d,#284b63); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: .4rem 0 0 0; opacity: .9;}
.metric-card {border: 1px solid #e6eaf0; border-radius: 12px; padding: 14px;}
.small-muted {color: #64748b; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🩺 RAGDiagnose</h1>
  <p>Retrieval-Augmented Generation • Source-grounded answers • Heuristic evaluation • Root-cause insights</p>
</div>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        value=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
        type="password",
        help="Paste your Groq API key here, or add GROQ_API_KEY in Streamlit Secrets."
    )
    model = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    chunk_size = st.slider("Chunk size (characters)", 400, 1600, 900, 100)
    overlap = st.slider("Chunk overlap (characters)", 0, 300, 120, 20)
    top_k = st.slider("Retrieved chunks (Top-K)", 1, 6, 3)
    st.caption("Demo build: runs entirely on Streamlit Cloud; no separate Render backend required.")

# 4. Session State Initialization
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None

# 5. Core Functions
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_chunks(text: str, size: int, overlap_chars: int, filename: str, page_num: int) -> List[Dict[str, Any]]:
    text = clean_text(text)
    if not text:
        return []
    step = max(1, size - overlap_chars)
    out = []
    for start in range(0, len(text), step):
        part = text[start:start + size].strip()
        if len(part) >= 40:
            out.append({
                "text": part,
                "source": filename,
                "page": page_num,
                "chunk_id": len(out) + 1,
            })
    return out

def tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())

def retrieve(question: str, chunks: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    q_tokens = tokens(question)
    if not q_tokens or not chunks:
        return []
    q_counts = Counter(q_tokens)
    scored = []
    for chunk in chunks:
        doc_tokens = tokens(chunk["text"])
        if not doc_tokens:
            continue
        d_counts = Counter(doc_tokens)
        overlap_score = sum(min(q_counts[t], d_counts[t]) for t in q_counts)
        coverage = overlap_score / max(1, sum(q_counts.values()))
        # Mildly reward chunks containing exact query phrases/terms.
        phrase_bonus = 0.15 if question.lower().strip() in chunk["text"].lower() else 0

import os
import re
from collections import Counter
from typing import List, Dict, Any

import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

st.set_page_config(
    page_title="RAGDiagnose | RAG Evaluation & Diagnosis",
    page_icon="🩺",
    layout="wide",
)

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
  <h1>🩺 RAGDiagnose (Gemini Powered)</h1>
  <p>Retrieval-Augmented Generation • Source-grounded answers • Heuristic evaluation • Root-cause insights</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    # Yahan aapki API key default set kar di gayi hai
    api_key = st.text_input(
        "Gemini API Key",
        value="AQ.Ab8RN6JMe9HnE0kI7PHCZklGyOPm_Uj3eCwTNE5vrHn19bo9iw",
        type="password",
        help="Paste your Gemini API key here."
    )
    
    model_name = st.selectbox(
        "Gemini model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
        index=0
    )
    
    chunk_size = st.slider("Chunk size (characters)", 400, 1600, 900, 100)
    overlap = st.slider("Chunk overlap (characters)", 0, 300, 120, 20)
    top_k = st.slider("Retrieved chunks (Top-K)", 1, 6, 3)

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None

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
        phrase_bonus = 0.15 if question.lower().strip() in chunk["text"].lower() else 0.0
        score = coverage + phrase_bonus
        if score > 0:
            scored.append({**chunk, "score": min(score, 1.0)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]

def make_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        f"[Source {i+1}: {c['source']}, page {c['page']}]\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    return f"""You are RAGDiagnose, a document-grounded assistant.
Answer the user's question using ONLY the supplied context.
If the context does not contain enough information, clearly say: "The uploaded documents do not provide enough information to answer this."
Do not invent facts. Keep the answer clear and concise. Cite supporting sources inline like [Source 1, page 2].

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

def diagnose(question: str, retrieved: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
    if not retrieved:
        return {
            "retrieval_status": "Retrieval failure",
            "grounding_risk": "High",
            "root_cause": "No chunks matched the question.",
            "recommendation": "Try more specific terms from the document."
        }
    top_score = retrieved[0]["score"]
    if top_score < 0.12:

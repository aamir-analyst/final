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
    
    # PERMANENT FIX: Stable Groq models that do not cause 404 errors
    model = st.selectbox(
        "Groq model",
        [
            "llama3-8b-8192", 
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
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
            "root_cause": "No chunks matched the question. The query may use different terminology, or the uploaded document may not contain the answer.",
            "recommendation": "Try more specific terms from the document, upload a more relevant PDF, or improve chunking/retrieval."
        }
    top_score = retrieved[0]["score"]
    if top_score < 0.12:
        status, risk = "Weak retrieval", "High"
        cause = "Retrieved context has low lexical overlap with the query; relevant evidence may be missing."
        recommendation = "Rephrase the question using document terminology and inspect the retrieved chunks before trusting the answer."
    elif top_score < 0.28:
        status, risk = "Moderate retrieval", "Medium"
        cause = "Some query terms match the retrieved context, but relevance may be incomplete."
        recommendation = "Increase Top-K or adjust chunk size/overlap, then compare the answer and citations."
    else:
        status, risk = "Relevant context found", "Low–Medium"
        cause = "The retrieved context shares several terms with the question. Lexical matching is only a heuristic and does not prove semantic relevance."
        recommendation = "Verify that the cited source actually supports each claim; use a labelled evaluation set for stronger validation."
    if len(answer.strip()) < 25:
        risk = "High"
        cause += " The generated answer is unusually short and may indicate an incomplete response."
    return {
        "retrieval_status": status,
        "grounding_risk": risk,
        "root_cause": cause,
        "recommendation": recommendation,
    }

def read_pdf(uploaded_file) -> List[Dict[str, Any]]:
    reader = PdfReader(uploaded_file)
    all_chunks = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        all_chunks.extend(split_chunks(text, chunk_size, overlap, uploaded_file.name, idx))
    return all_chunks

# 6. UI Tabs
tabs = st.tabs(["📊 Overview", "📚 Knowledge Base", "💬 RAG Chat", "🧪 Evaluation Lab", "🩺 Root Cause Analysis"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(set(c["source"] for c in st.session_state.chunks)))
    c2.metric("Indexed chunks", len(st.session_state.chunks))
    c3.metric("Questions asked", len(st.session_state.history))
    if st.session_state.last_eval:
        c4.metric("Latest retrieval score", f"{st.session_state.last_eval['score']:.2f}")
    else:
        c4.metric("Latest retrieval score", "—")
    st.subheader("System status")
    status_col1, status_col2 = st.columns(2)
    status_col1.success("Streamlit app is running")
    status_col2.info("Groq API key: configured" if api_key else "Groq API key: not configured")
    st.markdown("#### What

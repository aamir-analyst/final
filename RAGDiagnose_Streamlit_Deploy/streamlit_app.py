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
    api_key = st.text_input(
        "Gemini API Key",
        value=st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")),
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
        status, risk = "Weak retrieval", "High"
        cause = "Retrieved context has low lexical overlap."
        recommendation = "Rephrase the question."
    elif top_score < 0.28:
        status, risk = "Moderate retrieval", "Medium"
        cause = "Some query terms match."
        recommendation = "Increase Top-K."
    else:
        status, risk = "Relevant context found", "Low–Medium"
        cause = "Context shares several terms."
        recommendation = "Verify citations."
    if len(answer.strip()) < 25:
        risk = "High"
        cause += " The generated answer is unusually short."
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
    status_col2.info("Gemini API key: configured" if api_key else "Gemini API key: not configured")

with tabs[1]:
    st.subheader("Upload PDF knowledge")
    files = st.file_uploader("Choose one or more text-based PDF files", type=["pdf"], accept_multiple_files=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("📥 Process uploaded PDFs", type="primary", use_container_width=True):
            if not files:
                st.warning("Please select at least one PDF.")
            else:
                combined = []
                for f in files:
                    combined.extend(read_pdf(f))
                st.session_state.chunks = combined
                st.session_state.documents = [f.name for f in files]
                st.session_state.history = []
                st.session_state.last_eval = None
                st.success(f"Processed {len(files)} file(s) into {len(combined)} text chunks.")
    with col_b:
        if st.button("🗑️ Clear knowledge base", use_container_width=True):
            st.session_state.chunks = []
            st.session_state.documents = []
            st.session_state.history = []
            st.session_state.last_eval = None
            st.rerun()

with tabs[2]:
    st.subheader("Ask questions grounded in your PDFs")
    question = st.text_area("Your question", placeholder="Ask something...", height=90)
    if st.button("Generate grounded answer", type="primary", disabled=not question.strip()):
        if not st.session_state.chunks:
            st.error("Upload and process a PDF first.")
        elif not api_key:
            st.error("Add your Gemini API key in the sidebar.")
        else:
            with st.spinner("Retrieving context and generating answer…"):
                retrieved = retrieve(question, st.session_state.chunks, top_k)
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name)
                    if retrieved:
                        prompt = make_prompt(question, retrieved)
                        response = model.generate_content(prompt)
                        answer = response.text
                    else:
                        answer = "The uploaded documents do not provide enough information."
                    
                    diag = diagnose(question, retrieved, answer)
                    score = retrieved[0]["score"] if retrieved else 0.0
                    record = {
                        "question": question,
                        "answer": answer,
                        "retrieved": retrieved,
                        "diagnosis": diag,
                        "score": score,
                        "model": model_name,
                    }
                    st.session_state.history.append(record)
                    st.session_state.last_eval = {"score": score, **diag}
                    st.session_state.current_result = record
                except Exception as e:
                    st.error(f"Gemini API request failed: {e}")
                    
    result = st.session_state.get("current_result")
    if result:
        st.markdown("### Answer")
        st.markdown(result["answer"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Top retrieval score", f"{result['score']:.2f}")
        m2.metric("Retrieval status", result["diagnosis"]["retrieval_status"])
        m3.metric("Grounding risk", result["diagnosis"]["grounding_risk"])
        if not result["retrieved"]:
            st.warning("No matching evidence found.")
        for i, item in enumerate(result["retrieved"], start=1):
            with st.expander(f"Source {i} · {item['source']} · Page {item['page']}"):
                st.write(item["text"])

with tabs[3]:
    st.subheader("Evaluation Lab")
    if not st.session_state.history:
        st.info("Ask at least one question in RAG Chat.")
    else:
        scores = [r["score"] for r in st.session_state.history]
        st.metric("Average top retrieval score", f"{(sum(scores) / len(scores)):.2f}")
        for idx, record in enumerate(reversed(st.session_state.history[-10:]), start=1):
            with st.expander(f"{idx}. {record['question'][:100]}"):
                st.write("**Answer:**", record["answer"])

with tabs[4]:
    st.subheader("Heuristic root-cause diagnosis")
    if not st.session_state.history:
        st.info("Ask a question in RAG Chat to see diagnosis.")
    else:
        latest = st.session_state.history[-1]
        st.write("**Likely issue:**", latest["diagnosis"]["root_cause"])
        st.write("**Recommendation:**", latest["diagnosis"]["recommendation"])

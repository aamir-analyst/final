import os
import re
import math
from collections import Counter
from typing import List, Dict, Any

import streamlit as st
from pypdf import PdfReader
from groq import Groq

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
  <h1>🩺 RAGDiagnose</h1>
  <p>Retrieval-Augmented Generation • Source-grounded answers • Heuristic evaluation • Root-cause insights</p>
</div>
""", unsafe_allow_html=True)

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
        # Mildly reward chunks containing exact query phrases/terms.
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
    st.markdown("#### What this demo includes")
    st.markdown("""
    - PDF upload and page-aware text chunking
    - Lightweight lexical retrieval (no embedding model download required)
    - Groq-powered answers constrained to retrieved context
    - Source snippets and page references
    - Heuristic retrieval/grounding-risk diagnosis
    - Simple evaluation history for live demos
    """)
    st.warning("Evaluation scores and root-cause labels are heuristic demo signals, not a formal benchmark or proof of factual correctness.")

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
                errors = []
                for f in files:
                    try:
                        extracted = read_pdf(f)
                        combined.extend(extracted)
                        if not extracted:
                            errors.append(f"{f.name}: no extractable text found (may be scanned/image-only PDF).")
                    except Exception as e:
                        errors.append(f"{f.name}: {e}")
                st.session_state.chunks = combined
                st.session_state.documents = [f.name for f in files]
                st.session_state.history = []
                st.session_state.last_eval = None
                st.success(f"Processed {len(files)} file(s) into {len(combined)} text chunks.")
                for err in errors:
                    st.warning(err)
    with col_b:
        if st.button("🗑️ Clear knowledge base", use_container_width=True):
            st.session_state.chunks = []
            st.session_state.documents = []
            st.session_state.history = []
            st.session_state.last_eval = None
            st.rerun()
    if st.session_state.chunks:
        st.markdown("#### Indexed documents")
        for doc in sorted(set(c["source"] for c in st.session_state.chunks)):
            doc_chunks = [c for c in st.session_state.chunks if c["source"] == doc]
            st.write(f"**{doc}** — {len(doc_chunks)} chunks")
        with st.expander("Preview indexed chunks"):
            for c in st.session_state.chunks[:12]:
                st.markdown(f"**{c['source']} · Page {c['page']} · Chunk {c['chunk_id']}**")
                st.write(c["text"][:700] + ("…" if len(c["text"]) > 700 else ""))
                st.divider()

with tabs[2]:
    st.subheader("Ask questions grounded in your PDFs")
    if not st.session_state.chunks:
        st.info("First open Knowledge Base and process at least one PDF.")
    question = st.text_area("Your question", placeholder="Ask something that should be answered by the uploaded PDF…", height=90)
    if st.button("Generate grounded answer", type="primary", disabled=not question.strip(), use_container_width=False):
        if not st.session_state.chunks:
            st.error("Upload and process a PDF first.")
        elif not api_key:
            st.error("Add your Groq API key in the sidebar or Streamlit Secrets.")
        else:
            with st.spinner("Retrieving context and generating answer…"):
                retrieved = retrieve(question, st.session_state.chunks, top_k)
                try:
                    client = Groq(api_key=api_key)
                    if retrieved:
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "Follow the grounding instructions exactly. Never invent information absent from context."},
                                {"role": "user", "content": make_prompt(question, retrieved)},
                            ],
                            temperature=0.1,
                            max_tokens=900,
                        )
                        answer = completion.choices[0].message.content or "No answer was returned."
                    else:
                        answer = "The uploaded documents do not provide enough information to answer this. No relevant text chunks were retrieved."
                    diag = diagnose(question, retrieved, answer)
                    score = retrieved[0]["score"] if retrieved else 0.0
                    record = {
                        "question": question,
                        "answer": answer,
                        "retrieved": retrieved,
                        "diagnosis": diag,
                        "score": score,
                        "model": model,
                    }
                    st.session_state.history.append(record)
                    st.session_state.last_eval = {"score": score, **diag}
                    st.session_state.current_result = record
                except Exception as e:
                    st.error(f"Groq request failed: {e}")
    result = st.session_state.get("current_result")
    if result:
        st.markdown("### Answer")
        st.markdown(result["answer"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Top retrieval score", f"{result['score']:.2f}")
        m2.metric("Retrieval status", result["diagnosis"]["retrieval_status"])
        m3.metric("Grounding risk", result["diagnosis"]["grounding_risk"])
        st.markdown("### Retrieved evidence")
        if not result["retrieved"]:
            st.warning("No matching evidence found.")
        for i, item in enumerate(result["retrieved"], start=1):
            with st.expander(f"Source {i} · {item['source']} · Page {item['page']} · Score {item['score']:.2f}", expanded=(i == 1)):
                st.write(item["text"])
        st.caption(f"Model: {result['model']} · The retrieval score is a lexical heuristic, not a calibrated probability.")

with tabs[3]:
    st.subheader("Evaluation Lab")
    st.markdown("Run a quick heuristic review of the current question history. For formal evaluation, use a labelled question/ground-truth dataset.")
    if not st.session_state.history:
        st.info("Ask at least one question in RAG Chat to populate evaluation results.")
    else:
        scores = [r["score"] for r in st.session_state.history]
        avg_score = sum(scores) / len(scores)
        high_risk = sum(1 for r in st.session_state.history if r["diagnosis"]["grounding_risk"] == "High")
        e1, e2, e3 = st.columns(3)
        e1.metric("Questions evaluated", len(scores))
        e2.metric("Average top retrieval score", f"{avg_score:.2f}")
        e3.metric("High-risk responses", high_risk)
        st.markdown("#### Recent query results")
        for idx, record in enumerate(reversed(st.session_state.history[-20:]), start=1):
            with st.expander(f"{idx}. {record['question'][:100]} — score {record['score']:.2f}"):
                st.write("**Answer:**", record["answer"])
                st.write("**Retrieval status:**", record["diagnosis"]["retrieval_status"])
                st.write("**Grounding risk:**", record["diagnosis"]["grounding_risk"])
                st.write("**Potential root cause:**", record["diagnosis"]["root_cause"])

with tabs[4]:
    st.subheader("Heuristic root-cause diagnosis")
    if not st.session_state.history:
        st.info("Ask a question in RAG Chat to see diagnosis.")
    else:
        latest = st.session_state.history[-1]
        d = latest["diagnosis"]
        a, b = st.columns(2)
        a.metric("Retrieval status", d["retrieval_status"])
        b.metric("Grounding risk", d["grounding_risk"])
        st.markdown("#### Likely issue")
        st.write(d["root_cause"])
        st.markdown("#### Recommended next action")
        st.write(d["recommendation"])
        st.markdown("#### Debug trace")
        st.write(f"Question: {latest['question']}")
        st.write(f"Retrieved chunks: {len(latest['retrieved'])}")
        st.write(f"Top lexical score: {latest['score']:.3f}")
        st.caption("This diagnosis is rule-based and intended for a hackathon demonstration; it does not replace human review or robust RAG evaluation.")

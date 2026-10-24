import io
import re
import os
import html
import math
import hashlib
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Evaluation Pipeline",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 4px;
    }

    .sub-title {
        color: #667085;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #4f46e5;
    }

    .metric-label {
        color: #667085;
        font-size: 13px;
        margin-top: 3px;
    }

    .chunk-box {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 14px;
    }

    .good {
        color: #15803d;
        font-weight: 700;
    }

    .bad {
        color: #b91c1c;
        font-weight: 700;
    }

    .warning {
        color: #b45309;
        font-weight: 700;
    }

    .small-text {
        font-size: 13px;
        color: #667085;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_from_file(uploaded_file) -> str:
    """
    Supports TXT, PDF and DOCX.
    """
    file_name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if file_name.endswith(".txt") or file_name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")

    if file_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            pages = []

            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)

            return "\n".join(pages)

        except Exception as exc:
            raise ValueError(f"PDF read error: {exc}")

    if file_name.endswith(".docx"):
        try:
            from docx import Document

            document = Document(io.BytesIO(data))
            paragraphs = [p.text for p in document.paragraphs]
            return "\n".join(paragraphs)

        except Exception as exc:
            raise ValueError(f"DOCX read error: {exc}")

    raise ValueError(
        "Unsupported file format. Please upload PDF, TXT, MD or DOCX."
    )


# ============================================================
# TEXT CLEANING AND CHUNKING
# ============================================================

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100
) -> List[str]:
    """
    Character-based chunking with overlap.
    """
    text = clean_text(text)

    if not text:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# RETRIEVER
# ============================================================

class TfidfRetriever:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
        )

        if not chunks:
            self.matrix = None
        else:
            self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.chunks or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "chunk_id": int(index),
                    "score": float(scores[index]),
                    "text": self.chunks[index],
                }
            )

        return results


# ============================================================
# SIMPLE EXTRACTIVE GENERATOR
# ============================================================

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def normalize_tokens(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"\b[a-zA-Z0-9\u0900-\u097F]+\b", text)


def sentence_relevance(sentence: str, query: str) -> float:
    sentence_tokens = set(normalize_tokens(sentence))
    query_tokens = set(normalize_tokens(query))

    if not sentence_tokens or not query_tokens:
        return 0.0

    overlap = len(sentence_tokens.intersection(query_tokens))
    return overlap / math.sqrt(len(sentence_tokens) * len(query_tokens))


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict],
    max_sentences: int = 3
) -> str:
    """
    Extractive answer generator.
    It only uses retrieved context, so hallucination risk is low.
    """
    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded documents."

    all_sentences = []

    for item in retrieved_chunks:
        sentences = split_sentences(item["text"])

        for sentence in sentences:
            score = sentence_relevance(sentence, query)
            all_sentences.append(
                {
                    "text": sentence,
                    "score": score,
                    "chunk_id": item["chunk_id"],
                }
            )

    all_sentences.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    seen = set()

    for item in all_sentences:
        sentence = item["text"].strip()
        sentence_key = sentence.lower()

        if sentence_key in seen:
            continue

        if len(sentence) < 20:
            continue

        selected.append(sentence)
        seen.add(sentence_key)

        if len(selected) >= max_sentences:
            break

    if not selected:
        fallback = retrieved_chunks[0]["text"]
        return fallback[:700]

    return " ".join(selected)


# ============================================================
# EVALUATION HELPERS
# ============================================================

def lexical_tokens(text: str) -> set:
    return set(normalize_tokens(text))


def token_overlap_score(a: str, b: str) -> float:
    a_tokens = lexical_tokens(a)
    b_tokens = lexical_tokens(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens.intersection(b_tokens)) / len(b_tokens)


def context_relevance(query: str, retrieved_chunks: List[Dict]) -> float:
    if not retrieved_chunks:
        return 0.0

    query_tokens = lexical_tokens(query)

    if not query_tokens:
        return 0.0

    scores = []

    for item in retrieved_chunks:
        chunk_tokens = lexical_tokens(item["text"])
        if not chunk_tokens:
            scores.append(0.0)
            continue

        overlap = len(query_tokens.intersection(chunk_tokens))
        scores.append(overlap / len(query_tokens))

    return float(min(1.0, max(scores)))


def faithfulness_score(answer: str, context: str) -> float:
    """
    Sentence-level support score.
    A sentence is supported if a meaningful amount of its content
    is present in retrieved context.
    """
    answer_sentences = split_sentences(answer)

    if not answer_sentences:
        return 0.0

    scores = []

    for sentence in answer_sentences:
        sentence_tokens = lexical_tokens(sentence)
        context_tokens = lexical_tokens(context)

        if not sentence_tokens:
            continue

        overlap = len(sentence_tokens.intersection(context_tokens))
        scores.append(overlap / len(sentence_tokens))

    if not scores:
        return 0.0

    return float(min(1.0, sum(scores) / len(scores)))


def answer_accuracy(answer: str, expected_answer: str) -> float:
    if not expected_answer.strip():
        return 0.0

    answer_tokens = lexical_tokens(answer)
    expected_tokens = lexical_tokens(expected_answer)

    if not answer_tokens or not expected_tokens:
        return 0.0

    intersection = answer_tokens.intersection(expected_tokens)

    precision = len(intersection) / len(answer_tokens)
    recall = len(intersection) / len(expected_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(min(1.0, f1))


def get_root_cause(
    context_score: float,
    faithfulness: float,
    accuracy: float,
    retrieval_threshold: float = 0.35,
    faithfulness_threshold: float = 0.65,
    accuracy_threshold: float = 0.60,
) -> Tuple[str, str]:
    """
    Root-cause classification:
    - Retrieval failure
    - Generator hallucination
    - Both
    - Passed
    """
    retrieval_failed = context_score < retrieval_threshold
    generation_failed = (
        faithfulness < faithfulness_threshold
        or accuracy < accuracy_threshold
    )

    if retrieval_failed and generation_failed:
        return (
            "Both retrieval and generation issue",
            "Retrieved context has weak query overlap and the final answer is also insufficiently supported or inaccurate.",
        )

    if retrieval_failed:
        return (
            "Retriever failure",
            "The retrieved chunks do not contain enough query-relevant information.",
        )

    if faithfulness < faithfulness_threshold:
        return (
            "Generator hallucination",
            "Retrieved context appears relevant, but the answer contains unsupported information.",
        )

    if accuracy < accuracy_threshold:
        return (
            "Generator accuracy failure",
            "Retrieved context is relevant and mostly supported, but the answer does not match the expected answer sufficiently.",
        )

    return (
        "Passed",
        "Retriever and generator both passed the configured evaluation thresholds.",
    )


# ============================================================
# SYNTHETIC QUESTION GENERATION
# ============================================================

def generate_synthetic_tests(
    chunks: List[str],
    max_questions: int = 10
) -> List[Dict]:
    """
    Creates simple synthetic QA pairs from document sentences.
    This is deterministic and does not require an LLM/API.
    """
    tests = []

    for chunk_id, chunk in enumerate(chunks):
        sentences = split_sentences(chunk)

        for sentence in sentences:
            words = normalize_tokens(sentence)

            if len(words) < 8:
                continue

            # Select important words for a keyword-style question.
            selected_words = words[: min(5, len(words))]
            keyword_text = " ".join(selected_words)

            question = (
                f"What information is mentioned about: {keyword_text}?"
            )

            tests.append(
                {
                    "question": question,
                    "expected_answer": sentence,
                    "source_chunk_id": chunk_id,
                }
            )

            if len(tests) >= max_questions:
                return tests

    return tests


# ============================================================
# UI HELPERS
# ============================================================

def display_metric(label: str, value: float):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value:.0%}</div>
            <div class="metric-label">{html.escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_badge(score: float, threshold: float = 0.6) -> str:
    if score >= threshold:
        return f'<span class="good">{score:.0%}</span>'
    return f'<span class="bad">{score:.0%}</span>'


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧪 RAG Evaluation Pipeline</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">Retriever and Generator Tested Separately</div>',
    unsafe_allow_html=True,
)

st.info(
    "यह prototype retrieval और answer generation को अलग-अलग evaluate करता है "
    "और हर failure के लिए probable root cause बताता है।"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="PDF, TXT, MD और DOCX files supported हैं।",
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=300,
        max_value=1500,
        value=700,
        step=50,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=300,
        value=100,
        step=25,
    )

    top_k = st.slider(
        "Top K retrieved chunks",
        min_value=1,
        max_value=8,
        value=3,
    )

    max_answer_sentences = st.slider(
        "Answer sentences",
        min_value=1,
        max_value=6,
        value=3,
    )

    st.divider()

    st.markdown("### Evaluation thresholds")

    retrieval_threshold = st.slider(
        "Retrieval threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )

    faithfulness_threshold = st.slider(
        "Faithfulness threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )

    accuracy_threshold = st.slider(
        "Accuracy threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

if not uploaded_files:
    st.markdown("## Getting Started")
    st.write(
        "左 sidebar से कम-से-कम एक document upload करें। "
        "इसके बाद query enter करके retrieval और generation evaluate करें।"
    )

    st.markdown(
        """
        ### Supported features

        - PDF, TXT, MD और DOCX document ingestion
        - TF-IDF semantic-style retrieval
        - Retrieved chunk scores
        - Extractive answer generation
        - Context relevance score
        - Faithfulness score
        - Answer accuracy score
        - Root-cause diagnosis
        - Synthetic test-set evaluation
        """
    )

    st.stop()


all_text = []
file_summary = []

for uploaded_file in uploaded_files:
    try:
        text = extract_text_from_file(uploaded_file)
        text = clean_text(text)

        if text:
            all_text.append(text)
            file_summary.append(
                {
                    "file": uploaded_file.name,
                    "characters": len(text),
                    "status": "Loaded",
                }
            )
        else:
            file_summary.append(
                {
                    "file": uploaded_file.name,
                    "characters": 0,
                    "status": "Empty document",
                }
            )

    except Exception as exc:
        file_summary.append(
            {
                "file": uploaded_file.name,
                "characters": 0,
                "status": f"Error: {exc}",
            }
        )

combined_text = "\n\n".join(all_text)
chunks = split_into_chunks(
    combined_text,
    chunk_size=chunk_size,
    overlap=chunk_overlap,
)

if not chunks:
    st.error("Documents से कोई readable text नहीं मिला।")
    st.stop()

retriever = TfidfRetriever(chunks)


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

st.markdown("## 📄 Document Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    display_metric("Documents", len(uploaded_files))

with c2:
    display_metric("Chunks", len(chunks))

with c3:
    display_metric("Characters", min(len(combined_text) / 100000, 1.0))

with c4:
    display_metric("Retriever Ready", 1.0 if retriever.matrix is not None else 0.0)

with st.expander("View uploaded file status"):
    st.dataframe(
        pd.DataFrame(file_summary),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# QUERY AND MANUAL EVALUATION
# ============================================================

st.markdown("## 🔍 Query Evaluation")

default_query = (
    "What is the main information contained in the uploaded documents?"
)

query = st.text_input(
    "Enter your question",
    value=default_query,
)

expected_answer = st.text_area(
    "Expected answer / ground truth",
    placeholder=(
        "Optional: expected answer लिखें। "
        "Accuracy score calculate करने के लिए इसका उपयोग होगा।"
    ),
    height=100,
)

run_query = st.button(
    "Run RAG Evaluation",
    type="primary",
    use_container_width=True,
)

if run_query:
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    retrieved = retriever.search(query, top_k=top_k)

    context = "\n\n".join([item["text"] for item in retrieved])
    answer = generate_answer(
        query=query,
        retrieved_chunks=retrieved,
        max_sentences=max_answer_sentences,
    )

    retrieval_score = context_relevance(query, retrieved)
    faithfulness = faithfulness_score(answer, context)

    if expected_answer.strip():
        accuracy = answer_accuracy(answer, expected_answer)
    else:
        accuracy = 0.0

    root_cause, explanation = get_root_cause(
        context_score=retrieval_score,
        faithfulness=faithfulness,
        accuracy=accuracy if expected_answer.strip() else faithfulness,
        retrieval_threshold=retrieval_threshold,
        faithfulness_threshold=faithfulness_threshold,
        accuracy_threshold=accuracy_threshold,
    )

    st.session_state["last_result"] = {
        "query": query,
        "expected_answer": expected_answer,
        "retrieved": retrieved,
        "context": context,
        "answer": answer,
        "retrieval_score": retrieval_score,
        "faithfulness": faithfulness,
        "accuracy": accuracy,
        "root_cause": root_cause,
        "explanation": explanation,
    }


if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.markdown("### Evaluation Scores")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        display_metric("Context Relevance", result["retrieval_score"])

    with m2:
        display_metric("Faithfulness", result["faithfulness"])

    with m3:
        display_metric(
            "Answer Accuracy",
            result["accuracy"] if result["expected_answer"].strip() else 0.0,
        )

    with m4:
        passed = (
            result["retrieval_score"] >= retrieval_threshold
            and result["faithfulness"] >= faithfulness_threshold
            and (
                not result["expected_answer"].strip()
                or result["accuracy"] >= accuracy_threshold
            )
        )
        display_metric("Overall", 1.0 if passed else 0.0)

    st.markdown("### 🤖 Generated Answer")
    st.success(result["answer"])

    st.markdown("### 🧠 Root-Cause Diagnosis")

    if result["root_cause"] == "Passed":
        st.success(f"✅ {result['root_cause']}")
    elif "Both" in result["root_cause"]:
        st.error(f"❌ {result['root_cause']}")
    else:
        st.warning(f"⚠️ {result['root_cause']}")

    st.write(result["explanation"])

    st.markdown("### 📚 Retrieved Context")

    if not result["retrieved"]:
        st.error("No chunks were retrieved.")
    else:
        for item in result["retrieved"]:
            st.markdown(
                f"""
                <div class="chunk-box">
                    <b>Rank:</b> {item['rank']}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Chunk ID:</b> {item['chunk_id']}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Similarity:</b> {item['score']:.3f}
                    <br><br>
                    {html.escape(item['text'])}
                </div>
                """,
                unsafe_allow_html=True,
            )

    report_row = {
        "query": result["query"],
        "generated_answer": result["answer"],
        "context_relevance": round(result["retrieval_score"], 4),
        "faithfulness": round(result["faithfulness"], 4),
        "answer_accuracy": round(result["accuracy"], 4),
        "root_cause": result["root_cause"],
        "diagnosis": result["explanation"],
    }

    report_df = pd.DataFrame([report_row])

    st.download_button(
        "⬇️ Download Current Evaluation CSV",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="rag_evaluation_report.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# RETRIEVER INSPECTION
# ============================================================

st.markdown("## 🧩 Retriever Inspection")

inspection_query = st.text_input(
    "Inspect retrieved chunks for another query",
    value="",
)

if inspection_query.strip():
    inspection_results = retriever.search(
        inspection_query,
        top_k=top_k,
    )

    for item in inspection_results:
        st.markdown(
            f"""
            <div class="chunk-box">
                <b>Rank {item['rank']}</b>
                &nbsp; | &nbsp;
                <b>Similarity:</b> {item['score']:.3f}
                <br><br>
                {html.escape(item['text'])}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SYNTHETIC TEST SET EVALUATION
# ============================================================

st.markdown("## 🧪 Automated Synthetic Test Evaluation")

st.write(
    "यह section uploaded documents के sentences से deterministic synthetic "
    "questions बनाकर पूरे pipeline को batch में test करता है।"
)

synthetic_count = st.slider(
    "Number of synthetic test cases",
    min_value=3,
    max_value=30,
    value=10,
)

run_batch = st.button(
    "Generate and Run Synthetic Tests",
    use_container_width=True,
)

if run_batch:
    synthetic_tests = generate_synthetic_tests(
        chunks=chunks,
        max_questions=synthetic_count,
    )

    if not synthetic_tests:
        st.warning("Synthetic tests generate नहीं हो सके।")
    else:
        batch_rows = []

        progress = st.progress(0.0)

        for index, test in enumerate(synthetic_tests):
            test_query = test["question"]
            expected = test["expected_answer"]

            retrieved = retriever.search(
                test_query,
                top_k=top_k,
            )

            context = "\n\n".join(
                [item["text"] for item in retrieved]
            )

            answer = generate_answer(
                query=test_query,
                retrieved_chunks=retrieved,
                max_sentences=max_answer_sentences,
            )

            retrieval_score = context_relevance(
                test_query,
                retrieved,
            )

            faithfulness = faithfulness_score(
                answer,
                context,
            )

            accuracy = answer_accuracy(
                answer,
                expected,
            )

            root_cause, explanation = get_root_cause(
                context_score=retrieval_score,
                faithfulness=faithfulness,
                accuracy=accuracy,
                retrieval_threshold=retrieval_threshold,
                faithfulness_threshold=faithfulness_threshold,
                accuracy_threshold=accuracy_threshold,
            )

            batch_rows.append(
                {
                    "test_id": index + 1,
                    "question": test_query,
                    "expected_answer": expected,
                    "generated_answer": answer,
                    "retrieval_score": round(retrieval_score, 4),
                    "faithfulness": round(faithfulness, 4),
                    "accuracy": round(accuracy, 4),
                    "root_cause": root_cause,
                    "diagnosis": explanation,
                }
            )

            progress.progress((index + 1) / len(synthetic_tests))

        batch_df = pd.DataFrame(batch_rows)
        st.session_state["batch_df"] = batch_df

if "batch_df" in st.session_state:
    batch_df = st.session_state["batch_df"]

    avg_retrieval = batch_df["retrieval_score"].mean()
    avg_faithfulness = batch_df["faithfulness"].mean()
    avg_accuracy = batch_df["accuracy"].mean()

    b1, b2, b3 = st.columns(3)

    with b1:
        display_metric("Avg Retrieval", avg_retrieval)

    with b2:
        display_metric("Avg Faithfulness", avg_faithfulness)

    with b3:
        display_metric("Avg Accuracy", avg_accuracy)

    st.dataframe(
        batch_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Full Synthetic Evaluation CSV",
        data=batch_df.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_rag_evaluation_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### Root-Cause Distribution")

    root_cause_counts = (
        batch_df["root_cause"]
        .value_counts()
        .rename_axis("root_cause")
        .reset_index(name="count")
    )

    st.bar_chart(
        root_cause_counts.set_index("root_cause")
    )


# ============================================================
# CHUNK VIEWER
# ============================================================

with st.expander("View all document chunks"):
    chunk_df = pd.DataFrame(
        [
            {
                "chunk_id": index,
                "characters": len(chunk),
                "text": chunk,
            }
            for index, chunk in enumerate(chunks)
        ]
    )

    st.dataframe(
        chunk_df,
        use_container_width=True,
        hide_index=True,
    )

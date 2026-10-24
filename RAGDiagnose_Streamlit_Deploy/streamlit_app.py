import io
import re
import html
import math
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAGDiagnose",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL FRONTEND STYLE
# ============================================================

st.markdown(
    """
    <style>
    /* ------------------------------
       Global application colors
    ------------------------------ */

    :root {
        --primary: #6366f1;
        --primary-dark: #4338ca;
        --text-main: #111827;
        --text-soft: #475467;
        --text-muted: #667085;
        --card-bg: #ffffff;
        --page-bg: #f5f7fb;
        --border: #e4e7ec;
        --success: #027a48;
        --warning: #b54708;
    }

    /* Main background */
    .stApp {
        background:
            linear-gradient(
                135deg,
                #f8faff 0%,
                #f4f6fb 45%,
                #eef2ff 100%
            );
    }

    /* Main content width */
    .block-container {
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Force readable text */
    .stApp,
    .stApp p,
    .stApp li,
    .stApp label,
    .stApp span,
    .stApp div {
        color: var(--text-main);
    }

    /* Main headings */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 2.25rem !important;
    }

    h2 {
        font-size: 1.65rem !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #ffffff 0%,
                #f8f9fc 100%
            );
        border-right: 1px solid #e4e7ec;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #111827 !important;
    }

    /* Hero section */
    .hero {
        background:
            linear-gradient(
                135deg,
                #312e81 0%,
                #4f46e5 50%,
                #7c3aed 100%
            );
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 22px;
        box-shadow: 0 18px 45px rgba(79, 70, 229, 0.22);
    }

    .hero-title {
        color: #ffffff !important;
        font-size: 2.35rem;
        font-weight: 850;
        margin: 0;
    }

    .hero-subtitle {
        color: #e0e7ff !important;
        font-size: 1rem;
        margin-top: 8px;
    }

    /* Section heading */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 11px;
        color: #111827 !important;
        font-size: 1.55rem;
        font-weight: 800;
        margin: 30px 0 14px 0;
    }

    .section-icon {
        width: 38px;
        height: 38px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #e0e7ff;
        border-radius: 12px;
        font-size: 21px;
    }

    /* Information card */
    .info-card {
        background: #eef4ff;
        border: 1px solid #bfd3ff;
        border-left: 5px solid #4f46e5;
        color: #1e3a8a !important;
        padding: 15px 18px;
        border-radius: 12px;
        line-height: 1.55;
        margin: 12px 0 22px 0;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff !important;
        border: 1px solid #e4e7ec;
        border-radius: 17px;
        padding: 21px 12px;
        text-align: center;
        min-height: 125px;
        box-shadow: 0 7px 20px rgba(16, 24, 40, 0.06);
    }

    .metric-value {
        color: #4f46e5 !important;
        font-size: 2rem;
        font-weight: 850;
        line-height: 1.1;
    }

    .metric-label {
        color: #475467 !important;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 9px;
    }

    /* Retrieved text cards */
    .chunk-card {
        background: #ffffff !important;
        border: 1px solid #d9def0;
        border-left: 5px solid #6366f1;
        color: #111827 !important;
        border-radius: 13px;
        padding: 17px;
        margin: 12px 0;
        box-shadow: 0 5px 14px rgba(16, 24, 40, 0.05);
        line-height: 1.65;
    }

    .chunk-card,
    .chunk-card *,
    .chunk-card b {
        color: #111827 !important;
    }

    .chunk-meta {
        color: #4f46e5 !important;
        font-weight: 750;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }

    /* Result card */
    .answer-card {
        background: #ffffff !important;
        border: 1px solid #c7d2fe;
        border-radius: 15px;
        padding: 20px;
        color: #111827 !important;
        line-height: 1.7;
        box-shadow: 0 7px 20px rgba(16, 24, 40, 0.05);
    }

    .answer-card,
    .answer-card * {
        color: #111827 !important;
    }

    /* Input fields */
    input,
    textarea {
        color: #111827 !important;
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 9px !important;
    }

    input:focus,
    textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px #c7d2fe !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px !important;
        font-weight: 700 !important;
        min-height: 42px;
    }

    .stButton > button[kind="primary"] {
        background: #4f46e5 !important;
        color: #ffffff !important;
        border: 0 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #3730a3 !important;
    }

    .stDownloadButton > button {
        background: #4f46e5 !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary p {
        color: #111827 !important;
        font-weight: 700 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] p {
        color: inherit !important;
    }

    /* Divider */
    hr {
        border-color: #e4e7ec !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )
        except Exception as exc:
            raise ValueError(f"Unable to read the PDF file: {exc}")

    if name.endswith(".docx"):
        try:
            from docx import Document

            document = Document(io.BytesIO(data))
            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )
        except Exception as exc:
            raise ValueError(f"Unable to read the DOCX file: {exc}")

    raise ValueError(
        "Unsupported file format. Please upload PDF, TXT, MD, or DOCX files."
    )


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def create_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    text = clean_text(text)

    if not text:
        return []

    overlap = min(overlap, chunk_size - 1)
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

class Retriever:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12000,
        )
        self.matrix = (
            self.vectorizer.fit_transform(chunks)
            if chunks
            else None
        )

    def search(self, query: str, top_k: int) -> List[Dict]:
        if not self.chunks or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "rank": rank,
                "chunk_id": int(index),
                "score": float(scores[index]),
                "text": self.chunks[index],
            }
            for rank, index in enumerate(indices, 1)
        ]


# ============================================================
# ANSWER GENERATION
# ============================================================

def tokens(text: str) -> set:
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))


def sentences(text: str) -> List[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", text.strip())
        if item.strip()
    ]


def generate_answer(
    query: str,
    retrieved: List[Dict],
    max_sentences: int,
) -> str:
    if not retrieved:
        return "No relevant information was found in the uploaded documents."

    query_tokens = tokens(query)
    candidates = []

    for item in retrieved:
        for sentence in sentences(item["text"]):
            sentence_tokens = tokens(sentence)

            if not sentence_tokens:
                continue

            overlap = len(query_tokens & sentence_tokens)
            score = overlap / math.sqrt(
                len(query_tokens) * len(sentence_tokens)
            ) if query_tokens else 0

            candidates.append((score, sentence))

    candidates.sort(reverse=True)

    selected = []
    used = set()

    for _, sentence in candidates:
        key = sentence.lower()

        if key in used or len(sentence) < 20:
            continue

        selected.append(sentence)
        used.add(key)

        if len(selected) >= max_sentences:
            break

    return " ".join(selected) if selected else retrieved[0]["text"][:800]


# ============================================================
# EVALUATION
# ============================================================

def relevance_score(query: str, retrieved: List[Dict]) -> float:
    if not retrieved:
        return 0.0

    query_tokens = tokens(query)

    if not query_tokens:
        return 0.0

    scores = []

    for item in retrieved:
        item_tokens = tokens(item["text"])
        scores.append(
            len(query_tokens & item_tokens) / len(query_tokens)
        )

    return min(1.0, max(scores))


def faithfulness_score(answer: str, context: str) -> float:
    answer_sentences = sentences(answer)
    context_tokens = tokens(context)

    if not answer_sentences:
        return 0.0

    scores = []

    for sentence in answer_sentences:
        sentence_tokens = tokens(sentence)

        if sentence_tokens:
            scores.append(
                len(sentence_tokens & context_tokens)
                / len(sentence_tokens)
            )

    return min(1.0, sum(scores) / len(scores)) if scores else 0.0


def accuracy_score(answer: str, expected: str) -> float:
    answer_tokens = tokens(answer)
    expected_tokens = tokens(expected)

    if not answer_tokens or not expected_tokens:
        return 0.0

    common = answer_tokens & expected_tokens

    precision = len(common) / len(answer_tokens)
    recall = len(common) / len(expected_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def diagnose(
    retrieval: float,
    faithfulness: float,
    accuracy: float,
    has_expected: bool,
    retrieval_limit: float,
    faithfulness_limit: float,
    accuracy_limit: float,
):
    retrieval_failed = retrieval < retrieval_limit
    faithfulness_failed = faithfulness < faithfulness_limit
    accuracy_failed = has_expected and accuracy < accuracy_limit

    if retrieval_failed and (faithfulness_failed or accuracy_failed):
        return (
            "Retrieval and Generation Issue",
            "The retrieved context is weak and the generated answer is not sufficiently supported or accurate.",
        )

    if retrieval_failed:
        return (
            "Retriever Issue",
            "The retrieved chunks have low relevance to the user query.",
        )

    if faithfulness_failed:
        return (
            "Generation Faithfulness Issue",
            "The answer contains information that is not sufficiently supported by the retrieved context.",
        )

    if accuracy_failed:
        return (
            "Answer Accuracy Issue",
            "The answer does not sufficiently match the expected answer.",
        )

    return (
        "Passed",
        "The retrieval and answer generation pipeline passed the configured thresholds.",
    )


# ============================================================
# UI HELPERS
# ============================================================

def section(title: str, icon: str):
    st.markdown(
        f"""
        <div class="section-heading">
            <span class="section-icon">{icon}</span>
            <span>{html.escape(title)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric(label: str, value: float):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value:.0%}</div>
            <div class="metric-label">{html.escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_chunk(item: Dict):
    safe_text = html.escape(item["text"])

    st.markdown(
        f"""
        <div class="chunk-card">
            <div class="chunk-meta">
                Rank {item["rank"]} &nbsp;•&nbsp;
                Chunk {item["chunk_id"]} &nbsp;•&nbsp;
                Similarity {item["score"]:.3f}
            </div>
            <div>{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧪 RAG Evaluation Pipeline</div>
        <div class="hero-subtitle">
            Evaluate retrieval quality, answer faithfulness, and answer accuracy
            in one place.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        This application evaluates the retriever and answer generator separately
        and identifies the probable root cause of each evaluation failure.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Configuration")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="PDF, TXT, MD, and DOCX files are supported.",
    )

    st.markdown("### Chunking Settings")

    chunk_size = st.slider(
        "Chunk Size",
        300,
        1500,
        700,
        50,
    )

    overlap = st.slider(
        "Chunk Overlap",
        0,
        300,
        100,
        25,
    )

    top_k = st.slider(
        "Retrieved Chunks",
        1,
        8,
        3,
    )

    max_sentences = st.slider(
        "Maximum Answer Sentences",
        1,
        6,
        3,
    )

    st.divider()
    st.markdown("### Evaluation Thresholds")

    retrieval_limit = st.slider(
        "Retrieval Threshold",
        0.0,
        1.0,
        0.35,
        0.05,
    )

    faithfulness_limit = st.slider(
        "Faithfulness Threshold",
        0.0,
        1.0,
        0.65,
        0.05,
    )

    accuracy_limit = st.slider(
        "Accuracy Threshold",
        0.0,
        1.0,
        0.60,
        0.05,
    )


# ============================================================
# EMPTY STATE
# ============================================================

if not uploaded_files:
    section("Getting Started", "🚀")

    st.markdown(
        """
        Upload at least one document from the sidebar to start evaluating
        your RAG pipeline.
        """
    )

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:
        st.info("📄 Upload PDF, TXT, MD, or DOCX files.")

    with feature_col2:
        st.info("🔍 Search and inspect relevant document chunks.")

    with feature_col3:
        st.info("📊 Evaluate retrieval and answer quality.")

    st.stop()


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = []
file_status = []

for uploaded_file in uploaded_files:
    try:
        text = clean_text(extract_text(uploaded_file))

        if text:
            documents.append(text)
            file_status.append(
                {
                    "File": uploaded_file.name,
                    "Characters": len(text),
                    "Status": "Loaded",
                }
            )
        else:
            file_status.append(
                {
                    "File": uploaded_file.name,
                    "Characters": 0,
                    "Status": "No readable text",
                }
            )

    except Exception as exc:
        file_status.append(
            {
                "File": uploaded_file.name,
                "Characters": 0,
                "Status": f"Error: {exc}",
            }
        )

combined_text = "\n\n".join(documents)

chunks = create_chunks(
    combined_text,
    chunk_size,
    overlap,
)

if not chunks:
    st.error("No readable text was found in the uploaded documents.")
    st.stop()

retriever = Retriever(chunks)


# ============================================================
# DOCUMENT OVERVIEW
# ============================================================

section("Document Overview", "📄")

overview1, overview2, overview3, overview4 = st.columns(4)

with overview1:
    metric("Documents", 1.0)

with overview2:
    metric("Chunks Available", 1.0 if chunks else 0.0)

with overview3:
    metric("Text Available", 1.0 if combined_text else 0.0)

with overview4:
    metric("Retriever Ready", 1.0 if retriever.matrix is not None else 0.0)

with st.expander("View Uploaded File Status"):
    st.dataframe(
        pd.DataFrame(file_status),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# QUERY EVALUATION
# ============================================================

section("Query Evaluation", "🔍")

query = st.text_input(
    "Enter Your Question",
    placeholder="Example: What is the main topic of these documents?",
)

expected_answer = st.text_area(
    "Expected Answer / Ground Truth",
    placeholder=(
        "Optional: enter the expected answer to calculate answer accuracy."
    ),
    height=110,
)

run_evaluation = st.button(
    "Run RAG Evaluation",
    type="primary",
    use_container_width=True,
)

if run_evaluation:
    if not query.strip():
        st.warning("Please enter a question first.")
        st.stop()

    retrieved = retriever.search(query, top_k)
    context = "\n\n".join(item["text"] for item in retrieved)

    answer = generate_answer(
        query,
        retrieved,
        max_sentences,
    )

    retrieval = relevance_score(query, retrieved)
    faithfulness = faithfulness_score(answer, context)

    has_expected = bool(expected_answer.strip())

    accuracy = (
        accuracy_score(answer, expected_answer)
        if has_expected
        else 0.0
    )

    root_cause, explanation = diagnose(
        retrieval,
        faithfulness,
        accuracy,
        has_expected,
        retrieval_limit,
        faithfulness_limit,
        accuracy_limit,
    )

    st.session_state["evaluation"] = {
        "query": query,
        "answer": answer,
        "retrieved": retrieved,
        "retrieval": retrieval,
        "faithfulness": faithfulness,
        "accuracy": accuracy,
        "root_cause": root_cause,
        "explanation": explanation,
        "expected": expected_answer,
    }


# ============================================================
# EVALUATION RESULT
# ============================================================

if "evaluation" in st.session_state:
    result = st.session_state["evaluation"]

    section("Evaluation Results", "📊")

    result1, result2, result3, result4 = st.columns(4)

    passed = (
        result["retrieval"] >= retrieval_limit
        and result["faithfulness"] >= faithfulness_limit
        and (
            not result["expected"].strip()
            or result["accuracy"] >= accuracy_limit
        )
    )

    with result1:
        metric("Context Relevance", result["retrieval"])

    with result2:
        metric("Faithfulness", result["faithfulness"])

    with result3:
        metric("Answer Accuracy", result["accuracy"])

    with result4:
        metric("Overall Result", 1.0 if passed else 0.0)

    st.markdown("### 🤖 Generated Answer")

    st.markdown(
        f"""
        <div class="answer-card">
            {html.escape(result["answer"])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 Root-Cause Diagnosis")

    if result["root_cause"] == "Passed":
        st.success(f"✅ {result['root_cause']}")
    elif "Retrieval" in result["root_cause"]:
        st.warning(f"⚠️ {result['root_cause']}")
    else:
        st.error(f"❌ {result['root_cause']}")

    st.write(result["explanation"])

    st.markdown("### 📚 Retrieved Context")

    if result["retrieved"]:
        for item in result["retrieved"]:
            show_chunk(item)
    else:
        st.warning("No relevant context was retrieved.")

    report = pd.DataFrame(
        [
            {
                "Query": result["query"],
                "Generated Answer": result["answer"],
                "Context Relevance": round(result["retrieval"], 4),
                "Faithfulness": round(result["faithfulness"], 4),
                "Answer Accuracy": round(result["accuracy"], 4),
                "Root Cause": result["root_cause"],
                "Explanation": result["explanation"],
            }
        ]
    )

    st.download_button(
        "Download Evaluation Report",
        report.to_csv(index=False).encode("utf-8"),
        "rag_evaluation_report.csv",
        "text/csv",
        use_container_width=True,
    )


# ============================================================
# RETRIEVER INSPECTION
# ============================================================

section("Retriever Inspection", "🧩")

inspection_query = st.text_input(
    "Inspect Retrieved Chunks",
    placeholder="Enter another question to inspect retrieval...",
)

if inspection_query.strip():
    inspection_results = retriever.search(
        inspection_query,
        top_k,
    )

    if inspection_results:
        for item in inspection_results:
            show_chunk(item)
    else:
        st.info("No chunks were retrieved for this query.")


# ============================================================
# ALL CHUNKS
# ============================================================

section("Document Chunks", "🗂️")

with st.expander("View All Document Chunks"):
    all_chunks = pd.DataFrame(
        [
            {
                "Chunk ID": index,
                "Characters": len(chunk),
                "Text": chunk,
            }
            for index, chunk in enumerate(chunks)
        ]
    )

    st.dataframe(
        all_chunks,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#667085 !important;
        font-size:13px;
        padding:12px;
    ">
        RAG Evaluation Pipeline • Retrieval and Generation Analysis
    </div>
    """,
    unsafe_allow_html=True,
)

import io
import re
import math
import html
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
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
            color: #111827 !important;
            margin-bottom: 4px;
        }

        .sub-title {
            color: #4b5563 !important;
            font-size: 15px;
            margin-bottom: 25px;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #111827 !important;
        }

        .metric-card {
            background-color: #f8fafc !important;
            border: 1px solid #d1d5db;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
        }

        .metric-value {
            font-size: 28px;
            font-weight: 800;
            color: #4f46e5 !important;
        }

        .metric-label {
            color: #374151 !important;
            font-size: 13px;
            margin-top: 3px;
        }

        .chunk-box {
            background-color: #f8fafc !important;
            color: #111827 !important;
            border-left: 4px solid #6366f1;
            border-top: 1px solid #d1d5db;
            border-right: 1px solid #d1d5db;
            border-bottom: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 10px;
            font-size: 14px;
            line-height: 1.6;
            word-wrap: break-word;
            overflow-wrap: anywhere;
        }

        .chunk-box,
        .chunk-box *,
        .chunk-box b {
            color: #111827 !important;
        }

        input,
        textarea {
            color: #111827 !important;
            background-color: #ffffff !important;
        }

        [data-baseweb="select"] * {
            color: #111827 !important;
        }

        [data-testid="stDataFrame"] {
            color: #111827 !important;
        }

        .stButton button {
            color: #111827 !important;
            background-color: #f3f4f6 !important;
            border: 1px solid #d1d5db !important;
        }

        .stButton button:hover {
            color: #4f46e5 !important;
            border-color: #4f46e5 !important;
        }

        .stDownloadButton button {
            color: #ffffff !important;
            background-color: #4f46e5 !important;
            border: none !important;
        }

        section[data-testid="stSidebar"] {
            color: #111827 !important;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #111827 !important;
        }

        [data-testid="stAlert"] p {
            color: inherit !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE TEXT EXTRACTION
# ============================================================

def extract_text_from_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    file_data = uploaded_file.getvalue()

    if file_name.endswith((".txt", ".md")):
        return file_data.decode("utf-8", errors="ignore")

    if file_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(file_data))
            pages = []

            for page in reader.pages:
                pages.append(page.extract_text() or "")

            return "\n".join(pages)

        except Exception as exc:
            raise ValueError(f"Unable to read the PDF file: {exc}")

    if file_name.endswith(".docx"):
        try:
            from docx import Document

            document = Document(io.BytesIO(file_data))
            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        except Exception as exc:
            raise ValueError(f"Unable to read the DOCX file: {exc}")

    raise ValueError(
        "Unsupported file format. Please upload PDF, TXT, MD, or DOCX files."
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
    overlap: int = 100,
) -> List[str]:
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
        self.vectorizer = None
        self.matrix = None

        if chunks:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10000,
            )
            self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.chunks or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "rank": rank,
                "chunk_id": int(index),
                "score": float(scores[index]),
                "text": self.chunks[index],
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]


# ============================================================
# ANSWER GENERATION
# ============================================================

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def normalize_tokens(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def sentence_relevance(sentence: str, query: str) -> float:
    sentence_tokens = set(normalize_tokens(sentence))
    query_tokens = set(normalize_tokens(query))

    if not sentence_tokens or not query_tokens:
        return 0.0

    overlap = len(sentence_tokens.intersection(query_tokens))

    return overlap / math.sqrt(
        len(sentence_tokens) * len(query_tokens)
    )


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict],
    max_sentences: int = 3,
) -> str:
    if not retrieved_chunks:
        return (
            "I could not find relevant information in the uploaded documents."
        )

    candidates = []

    for chunk in retrieved_chunks:
        for sentence in split_sentences(chunk["text"]):
            candidates.append(
                {
                    "text": sentence,
                    "score": sentence_relevance(sentence, query),
                }
            )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = []
    seen = set()

    for item in candidates:
        sentence = item["text"].strip()
        key = sentence.lower()

        if key in seen or len(sentence) < 20:
            continue

        selected.append(sentence)
        seen.add(key)

        if len(selected) >= max_sentences:
            break

    if not selected:
        return retrieved_chunks[0]["text"][:700]

    return " ".join(selected)


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def lexical_tokens(text: str) -> set:
    return set(normalize_tokens(text))


def context_relevance(
    query: str,
    retrieved_chunks: List[Dict],
) -> float:
    if not retrieved_chunks:
        return 0.0

    query_tokens = lexical_tokens(query)

    if not query_tokens:
        return 0.0

    scores = []

    for chunk in retrieved_chunks:
        chunk_tokens = lexical_tokens(chunk["text"])

        if not chunk_tokens:
            scores.append(0.0)
            continue

        overlap = len(query_tokens.intersection(chunk_tokens))
        scores.append(overlap / len(query_tokens))

    return float(min(1.0, max(scores)))


def faithfulness_score(
    answer: str,
    context: str,
) -> float:
    answer_sentences = split_sentences(answer)
    context_tokens = lexical_tokens(context)

    if not answer_sentences:
        return 0.0

    scores = []

    for sentence in answer_sentences:
        sentence_tokens = lexical_tokens(sentence)

        if not sentence_tokens:
            continue

        overlap = len(sentence_tokens.intersection(context_tokens))
        scores.append(overlap / len(sentence_tokens))

    if not scores:
        return 0.0

    return float(min(1.0, sum(scores) / len(scores)))


def answer_accuracy(
    answer: str,
    expected_answer: str,
) -> float:
    answer_tokens = lexical_tokens(answer)
    expected_tokens = lexical_tokens(expected_answer)

    if not answer_tokens or not expected_tokens:
        return 0.0

    common_tokens = answer_tokens.intersection(expected_tokens)

    precision = len(common_tokens) / len(answer_tokens)
    recall = len(common_tokens) / len(expected_tokens)

    if precision + recall == 0:
        return 0.0

    return float(
        2 * precision * recall / (precision + recall)
    )


def get_root_cause(
    retrieval_score: float,
    faithfulness: float,
    accuracy: float,
    has_ground_truth: bool,
    retrieval_threshold: float,
    faithfulness_threshold: float,
    accuracy_threshold: float,
) -> Tuple[str, str]:
    retrieval_failed = retrieval_score < retrieval_threshold
    generation_failed = faithfulness < faithfulness_threshold

    if has_ground_truth:
        generation_failed = (
            generation_failed
            or accuracy < accuracy_threshold
        )

    if retrieval_failed and generation_failed:
        return (
            "Both Retrieval and Generation Issue",
            "The retrieved context has weak query relevance and the generated answer is either unsupported or inaccurate.",
        )

    if retrieval_failed:
        return (
            "Retriever Failure",
            "The retrieved chunks do not contain enough information related to the query.",
        )

    if faithfulness < faithfulness_threshold:
        return (
            "Generator Hallucination",
            "The retrieved context is relevant, but the generated answer contains unsupported information.",
        )

    if has_ground_truth and accuracy < accuracy_threshold:
        return (
            "Generator Accuracy Failure",
            "The retrieved context is relevant and supported, but the generated answer does not sufficiently match the expected answer.",
        )

    return (
        "Passed",
        "The retriever and generator passed the configured evaluation thresholds.",
    )


# ============================================================
# SYNTHETIC TEST GENERATION
# ============================================================

def generate_synthetic_tests(
    chunks: List[str],
    max_questions: int,
) -> List[Dict]:
    tests = []

    for chunk_id, chunk in enumerate(chunks):
        for sentence in split_sentences(chunk):
            words = normalize_tokens(sentence)

            if len(words) < 8:
                continue

            keywords = " ".join(words[:5])

            tests.append(
                {
                    "question": (
                        f"What information is mentioned about {keywords}?"
                    ),
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


def render_chunk(chunk: Dict, show_chunk_id: bool = True):
    chunk_id_html = ""

    if show_chunk_id:
        chunk_id_html = (
            f"<b>Chunk ID:</b> {chunk['chunk_id']}"
            "&nbsp;&nbsp;|&nbsp;&nbsp;"
        )

    safe_text = html.escape(chunk["text"])

    st.markdown(
        f"""
        <div class="chunk-box">
            <b>Rank:</b> {chunk['rank']}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            {chunk_id_html}
            <b>Similarity:</b> {chunk['score']:.3f}
            <br><br>
            {safe_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    "This prototype evaluates the retriever and answer generator separately. "
    "It also identifies the probable root cause of each evaluation failure."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="PDF, TXT, MD, and DOCX files are supported.",
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=1500,
        value=700,
        step=50,
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=300,
        value=100,
        step=25,
    )

    top_k = st.slider(
        "Number of Retrieved Chunks",
        min_value=1,
        max_value=8,
        value=3,
    )

    max_answer_sentences = st.slider(
        "Maximum Answer Sentences",
        min_value=1,
        max_value=6,
        value=3,
    )

    st.divider()
    st.markdown("### Evaluation Thresholds")

    retrieval_threshold = st.slider(
        "Retrieval Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )

    faithfulness_threshold = st.slider(
        "Faithfulness Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )

    accuracy_threshold = st.slider(
        "Accuracy Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
    )


# ============================================================
# DOCUMENT LOADING
# ============================================================

if not uploaded_files:
    st.markdown("## Getting Started")

    st.write(
        "Upload at least one document using the sidebar. "
        "Then enter a query to evaluate retrieval and answer generation."
    )

    st.markdown(
        """
        ### Supported Features

        - PDF, TXT, MD, and DOCX document ingestion
        - TF-IDF-based retrieval
        - Retrieved chunk similarity scores
        - Extractive answer generation
        - Context relevance scoring
        - Faithfulness scoring
        - Answer accuracy scoring
        - Root-cause diagnosis
        - Synthetic test-set evaluation
        """
    )

    st.stop()


all_text = []
file_summary = []

for uploaded_file in uploaded_files:
    try:
        extracted_text = clean_text(
            extract_text_from_file(uploaded_file)
        )

        if extracted_text:
            all_text.append(extracted_text)

            file_summary.append(
                {
                    "File": uploaded_file.name,
                    "Characters": len(extracted_text),
                    "Status": "Loaded",
                }
            )
        else:
            file_summary.append(
                {
                    "File": uploaded_file.name,
                    "Characters": 0,
                    "Status": "Empty document",
                }
            )

    except Exception as exc:
        file_summary.append(
            {
                "File": uploaded_file.name,
                "Characters": 0,
                "Status": f"Error: {exc}",
            }
        )


combined_text = "\n\n".join(all_text)

chunks = split_into_chunks(
    combined_text,
    chunk_size=chunk_size,
    overlap=chunk_overlap,
)

if not chunks:
    st.error("No readable text was found in the uploaded documents.")
    st.stop()

retriever = TfidfRetriever(chunks)


# ============================================================
# DOCUMENT OVERVIEW
# ============================================================

st.markdown("## 📄 Document Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    display_metric("Documents", 1.0)

with col2:
    display_metric("Chunks", 1.0 if chunks else 0.0)

with col3:
    display_metric("Text Available", 1.0 if combined_text else 0.0)

with col4:
    display_metric(
        "Retriever Ready",
        1.0 if retriever.matrix is not None else 0.0,
    )

with st.expander("View Uploaded File Status"):
    st.dataframe(
        pd.DataFrame(file_summary),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MANUAL QUERY EVALUATION
# ============================================================

st.markdown("## 🔍 Query Evaluation")

query = st.text_input(
    "Enter Your Question",
    value="What is the main information contained in the uploaded documents?",
)

expected_answer = st.text_area(
    "Expected Answer / Ground Truth",
    placeholder=(
        "Optional: enter the expected answer. "
        "This will be used to calculate answer accuracy."
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

    retrieved_chunks = retriever.search(
        query=query,
        top_k=top_k,
    )

    context = "\n\n".join(
        item["text"]
        for item in retrieved_chunks
    )

    generated_answer = generate_answer(
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_sentences=max_answer_sentences,
    )

    retrieval_score = context_relevance(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    faithfulness = faithfulness_score(
        answer=generated_answer,
        context=context,
    )

    has_ground_truth = bool(expected_answer.strip())

    accuracy = (
        answer_accuracy(
            answer=generated_answer,
            expected_answer=expected_answer,
        )
        if has_ground_truth
        else 0.0
    )

    root_cause, diagnosis = get_root_cause(
        retrieval_score=retrieval_score,
        faithfulness=faithfulness,
        accuracy=accuracy,
        has_ground_truth=has_ground_truth,
        retrieval_threshold=retrieval_threshold,
        faithfulness_threshold=faithfulness_threshold,
        accuracy_threshold=accuracy_threshold,
    )

    st.session_state["last_result"] = {
        "query": query,
        "expected_answer": expected_answer,
        "retrieved_chunks": retrieved_chunks,
        "generated_answer": generated_answer,
        "retrieval_score": retrieval_score,
        "faithfulness": faithfulness,
        "accuracy": accuracy,
        "root_cause": root_cause,
        "diagnosis": diagnosis,
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.markdown("### Evaluation Scores")

    score1, score2, score3, score4 = st.columns(4)

    overall_passed = (
        result["retrieval_score"] >= retrieval_threshold
        and result["faithfulness"] >= faithfulness_threshold
        and (
            not result["expected_answer"].strip()
            or result["accuracy"] >= accuracy_threshold
        )
    )

    with score1:
        display_metric(
            "Context Relevance",
            result["retrieval_score"],
        )

    with score2:
        display_metric(
            "Faithfulness",
            result["faithfulness"],
        )

    with score3:
        display_metric(
            "Answer Accuracy",
            result["accuracy"],
        )

    with score4:
        display_metric(
            "Overall Result",
            1.0 if overall_passed else 0.0,
        )

    st.markdown("### 🤖 Generated Answer")
    st.success(result["generated_answer"])

    st.markdown("### 🧠 Root-Cause Diagnosis")

    if result["root_cause"] == "Passed":
        st.success(f"✅ {result['root_cause']}")
    elif "Both" in result["root_cause"]:
        st.error(f"❌ {result['root_cause']}")
    else:
        st.warning(f"⚠️ {result['root_cause']}")

    st.write(result["diagnosis"])

    st.markdown("### 📚 Retrieved Context")

    if not result["retrieved_chunks"]:
        st.error("No relevant chunks were retrieved.")
    else:
        for item in result["retrieved_chunks"]:
            render_chunk(item, show_chunk_id=True)

    report = pd.DataFrame(
        [
            {
                "query": result["query"],
                "generated_answer": result["generated_answer"],
                "context_relevance": round(
                    result["retrieval_score"],
                    4,
                ),
                "faithfulness": round(
                    result["faithfulness"],
                    4,
                ),
                "answer_accuracy": round(
                    result["accuracy"],
                    4,
                ),
                "root_cause": result["root_cause"],
                "diagnosis": result["diagnosis"],
            }
        ]
    )

    st.download_button(
        "⬇️ Download Current Evaluation CSV",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="rag_evaluation_report.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# RETRIEVER INSPECTION
# ============================================================

st.markdown("## 🧩 Retriever Inspection")

inspection_query = st.text_input(
    "Inspect Retrieved Chunks for Another Query",
    value="",
)

if inspection_query.strip():
    inspection_results = retriever.search(
        query=inspection_query,
        top_k=top_k,
    )

    if not inspection_results:
        st.warning("No chunks were retrieved.")
    else:
        for item in inspection_results:
            render_chunk(item, show_chunk_id=False)


# ============================================================
# SYNTHETIC TEST EVALUATION
# ============================================================

st.markdown("## 🧪 Automated Synthetic Test Evaluation")

st.write(
    "This section creates deterministic synthetic questions from document "
    "sentences and evaluates the complete pipeline in batch mode."
)

synthetic_count = st.slider(
    "Number of Synthetic Test Cases",
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
        st.warning("Synthetic test cases could not be generated.")
    else:
        rows = []
        progress_bar = st.progress(0.0)

        for index, test_case in enumerate(synthetic_tests):
            test_query = test_case["question"]
            expected = test_case["expected_answer"]

            retrieved_chunks = retriever.search(
                query=test_query,
                top_k=top_k,
            )

            context = "\n\n".join(
                item["text"]
                for item in retrieved_chunks
            )

            generated_answer = generate_answer(
                query=test_query,
                retrieved_chunks=retrieved_chunks,
                max_sentences=max_answer_sentences,
            )

            retrieval_score = context_relevance(
                query=test_query,
                retrieved_chunks=retrieved_chunks,
            )

            faithfulness = faithfulness_score(
                answer=generated_answer,
                context=context,
            )

            accuracy = answer_accuracy(
                answer=generated_answer,
                expected_answer=expected,
            )

            root_cause, diagnosis = get_root_cause(
                retrieval_score=retrieval_score,
                faithfulness=faithfulness,
                accuracy=accuracy,
                has_ground_truth=True,
                retrieval_threshold=retrieval_threshold,
                faithfulness_threshold=faithfulness_threshold,
                accuracy_threshold=accuracy_threshold,
            )

            rows.append(
                {
                    "test_id": index + 1,
                    "question": test_query,
                    "expected_answer": expected,
                    "generated_answer": generated_answer,
                    "retrieval_score": round(
                        retrieval_score,
                        4,
                    ),
                    "faithfulness": round(
                        faithfulness,
                        4,
                    ),
                    "accuracy": round(
                        accuracy,
                        4,
                    ),
                    "root_cause": root_cause,
                    "diagnosis": diagnosis,
                }
            )

            progress_bar.progress(
                (index + 1) / len(synthetic_tests)
            )

        st.session_state["batch_df"] = pd.DataFrame(rows)


# ============================================================
# BATCH RESULT
# ============================================================

if "batch_df" in st.session_state:
    batch_df = st.session_state["batch_df"]

    average_retrieval = batch_df["retrieval_score"].mean()
    average_faithfulness = batch_df["faithfulness"].mean()
    average_accuracy = batch_df["accuracy"].mean()

    batch1, batch2, batch3 = st.columns(3)

    with batch1:
        display_metric(
            "Average Retrieval",
            average_retrieval,
        )

    with batch2:
        display_metric(
            "Average Faithfulness",
            average_faithfulness,
        )

    with batch3:
        display_metric(
            "Average Accuracy",
            average_accuracy,
        )

    st.dataframe(
        batch_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Synthetic Evaluation CSV",
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

with st.expander("View All Document Chunks"):
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

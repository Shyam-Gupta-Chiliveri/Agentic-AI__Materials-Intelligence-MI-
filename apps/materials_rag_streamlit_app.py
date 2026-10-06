import streamlit as st
import os
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger("pypdf").setLevel(logging.ERROR)

from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import time

load_dotenv(Path(__file__).parent.parent / ".env")

st.set_page_config(
    page_title="Materials Tech RAG System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header  { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .answer-box  { background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px;
                   border-left: 5px solid #1f77b4; margin: 1rem 0; }
    .source-box  { background-color: #fff9e6; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

BASE_DIR      = Path(__file__).parent.parent
DOCS_DIR      = BASE_DIR / "ISO_DIN standards_Thesis_Documents"
FAISS_DIR     = BASE_DIR / "faiss_index_local"
MAX_PAGES_PDF = 30   # cap per PDF — keeps first-build under 3 min

@st.cache_resource(show_spinner="⚙️ Loading embedding model…")
def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=False)
def build_or_load_index(_embeddings):
    if FAISS_DIR.exists():
        return FAISS.load_local(str(FAISS_DIR), _embeddings,
                                allow_dangerous_deserialization=True)

    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    all_docs = []

    bar = st.progress(0, text="📚 Building knowledge base…")
    for i, pdf in enumerate(pdf_files):
        bar.progress((i + 1) / len(pdf_files),
                     text=f"📄 {i+1}/{len(pdf_files)}: {pdf.name[:55]}")
        try:
            pages  = PyPDFLoader(str(pdf)).load_and_split(splitter)
            pages  = pages[:MAX_PAGES_PDF * 4]   # ~4 chunks/page × 30 pages
            all_docs.extend(pages)
        except Exception:
            pass
    bar.empty()

    if not all_docs:
        return None

    db = FAISS.from_documents(all_docs, _embeddings)
    db.save_local(str(FAISS_DIR))
    return db

def _extract_answer(msg):
    """GPT-OSS often puts text in reasoning_content or content blocks, not .content."""
    content = msg.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        content = "\n".join(parts)
    content = (content or "").strip()
    if not content:
        extra = msg.additional_kwargs or {}
        content = (extra.get("reasoning_content") or extra.get("reasoning") or "").strip()
    return content


def ask_question(question, db, llm, top_k=5):
    try:
        with st.spinner("🔍 Searching knowledge base…"):
            results = db.similarity_search_with_score(question, k=top_k)

        if not results:
            return "No relevant documents found.", []

        context = "\n\n".join(doc.page_content for doc, _ in results)
        sources = [
            f"{Path(doc.metadata.get('source','?')).name} (Page {doc.metadata.get('page',0)+1})"
            for doc, _ in results
        ]

        prompt = f"""You are a materials science lab instructor. Use the textbook excerpts below to answer the student's question.

Write a clear, practical answer in complete sentences (at least 120 words).
Cover the relevant lab steps if the question is about sample preparation
(sectioning, mounting, grinding, polishing, etching, cleaning, safety).
Use details from the excerpts. If the excerpts are incomplete, still answer
from what they do contain and say what is missing.

Textbook excerpts:
{context}

Student question: {question}

Answer:"""

        with st.spinner("💭 Generating answer…"):
            msg = llm.invoke(prompt)
            answer = _extract_answer(msg)

        if not answer:
            answer = (
                "The knowledge base found relevant textbook pages, but the model "
                "returned an empty answer. Try searching again, or ask a more "
                "specific question such as: 'What are the steps for metallographic "
                "sample preparation: cutting, mounting, grinding, polishing and etching?'"
            )

        return answer, sources
    except Exception as e:
        return f"Error: {str(e)}", []

def main():
    st.markdown('<h1 class="main-header">🔬 Materials Technology RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about ISO/DIN standards, metallography, hardness testing, and material properties</p>', unsafe_allow_html=True)

    groq_key = os.getenv("GROQ_API_KEY", "")

    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Sources to retrieve", 1, 10, 5)

        st.markdown("---")
        st.header("📊 System Info")
        pdf_count   = len(list(DOCS_DIR.glob("*.pdf"))) if DOCS_DIR.exists() else 0
        index_ready = FAISS_DIR.exists()
        st.info(f"📄 PDFs: {pdf_count}")
        if index_ready:
            st.success("✅ Knowledge base ready (cached)")
        else:
            st.warning(f"⚠️ Will index up to {MAX_PAGES_PDF} pages/PDF on first run")

        if index_ready and st.button("🔄 Rebuild Knowledge Base"):
            import shutil; shutil.rmtree(str(FAISS_DIR))
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.header("💡 Sample Questions")
        samples = [
            "What are differences between Brinell and Vickers hardness testing?",
            "How is grain size measured according to ISO 643?",
            "What are mechanical properties of EN-AC44300 aluminum alloy?",
            "What is austenitic grain size?",
            "Explain SEM imaging techniques",
        ]
        for i, q in enumerate(samples, 1):
            if st.button(f"{i}. {q[:40]}…", key=f"s{i}"):
                st.session_state.question = q

        st.markdown("---")
        st.markdown("### 🛠️ Technologies")
        st.markdown("""
- **LLM**: GPT-OSS 20B via Groq *(free)*
- **Embeddings**: BGE-small *(local)*
- **Vector DB**: FAISS *(local)*
- **Framework**: LangChain
        """)

    st.markdown("---")

    if not groq_key:
        st.error("GROQ_API_KEY is not set on the server. The RAG app cannot answer until that environment variable is provided.")
        return

    embeddings = get_embeddings()
    with st.spinner("🗄️ Loading knowledge base…"):
        db = build_or_load_index(embeddings)

    if db is None:
        st.error(f"❌ No PDFs found in `{DOCS_DIR}`")
        return

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        groq_api_key=groq_key,
        max_tokens=4096,
        reasoning_format="parsed",
        reasoning_effort="low",
    )

    question = st.text_input(
        "🔍 Ask your question:",
        value=st.session_state.get("question", ""),
        placeholder="e.g., What is the Vickers hardness test procedure?",
        key="q_input"
    )

    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        search = st.button("🚀 Search", type="primary", use_container_width=True)

    if search and question:
        st.session_state.question = question
        t0 = time.time()
        answer, sources = ask_question(question, db, llm, top_k)
        elapsed = time.time() - t0

        st.markdown("### 💡 Answer")
        st.markdown(answer)

        if sources:
            st.markdown("### 📚 Sources")
            st.markdown('<div class="source-box">', unsafe_allow_html=True)
            for i, s in enumerate(sources, 1):
                st.markdown(f"**{i}.** {s}")
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("⏱️ Response Time", f"{elapsed:.2f}s")
        c2.metric("📄 Sources Used",  len(sources))
        c3.metric("💬 Answer Length", f"{len(answer.split())} words")

    elif search:
        st.warning("⚠️ Please enter a question!")

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;'>🔬 Materials Technology RAG System | Local FAISS · LangChain · Groq (free)</div>",
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()

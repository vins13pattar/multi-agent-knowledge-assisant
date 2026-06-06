"""
main.py — Streamlit Chat UI for the Multi-Agent Knowledge Assistant.

Provides document upload (PDF/TXT), URL ingestion via Firecrawl,
chat interface with streaming responses, agent trace visibility,
and session memory management.
"""

import os
import uuid
import tempfile
import logging

import streamlit as st

from app.supervisor import run_query, get_chat_history
from app.rag import ingest_document, ingest_url, get_ingested_documents

# ═══════════════════════════════════════════════════════════════════════════
#  Page Configuration
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Multi-Agent Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [APP] %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Session State Initialization
# ═══════════════════════════════════════════════════════════════════════════

if "thread_id" not in st.session_state:
    q_params = st.query_params
    if "thread_id" in q_params:
        st.session_state.thread_id = q_params["thread_id"]
    else:
        st.session_state.thread_id = str(uuid.uuid4())
        st.query_params["thread_id"] = st.session_state.thread_id

if "chat_history" not in st.session_state:
    st.session_state.chat_history = get_chat_history(st.session_state.thread_id)

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = get_ingested_documents()

if "last_trace" not in st.session_state:
    st.session_state.last_trace = None


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&family=Fira+Code:wght@400;500&display=swap');

    /* Global Font Override & Theme Adjustments */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Headers styling */
    h1, h2, h3, h4, h5, h6, [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }

    /* Agent trace styling */
    .agent-trace {
        background-color: #0d0e15 !important;
        border-left: 3px solid #8b5cf6 !important;
        border-radius: 8px;
        padding: 14px;
        font-family: 'Fira Code', monospace;
        font-size: 0.85em;
        color: #cbd5e1;
    }
    .trace-intent {
        color: #38bdf8;
        font-weight: bold;
    }
    .trace-chunks {
        color: #34d399;
    }
    .trace-score {
        color: #fb923c;
    }
    /* Citation styling */
    .citation {
        color: #8b5cf6;
        font-weight: 600;
        font-size: 0.85em;
        cursor: pointer;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Primary Button Styling (New Chat) */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(124, 58, 237, 0.5) !important;
    }

    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.01);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Sidebar — Document Upload + URL Ingestion + Settings
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧠 Knowledge Assistant")
    st.caption("RAG + Multi-Agent System")

    if st.button("➕ New Chat", key="new_chat_btn", type="primary", use_container_width=True):
        st.session_state.chat_history = []
        new_thread = str(uuid.uuid4())
        st.session_state.thread_id = new_thread
        st.query_params["thread_id"] = new_thread
        st.session_state.last_trace = None
        st.rerun()

    st.divider()

    # ── Document Upload ─────────────────────────────────────────────────
    st.markdown("##### 📁 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        help="Documents are chunked, embedded, and stored in ChromaDB",
        key="file_uploader",
    )

    if uploaded_file is not None:
        if uploaded_file.name not in [d["name"] for d in st.session_state.ingested_docs]:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    # Save to temp file for processing
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    chunks = ingest_document(tmp_path, file_type="auto")
                    os.unlink(tmp_path)  # clean up

                    st.session_state.ingested_docs.append({
                        "name": uploaded_file.name,
                        "chunks": len(chunks),
                        "type": "file",
                    })
                    st.success(f"✅ {uploaded_file.name} — {len(chunks)} chunks stored")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {str(e)}")

    # ── URL Ingestion ───────────────────────────────────────────────────
    st.markdown("##### 🌐 Ingest from URL")
    url_input = st.text_input(
        "Enter URL",
        placeholder="https://example.com/article",
        key="url_input",
    )
    if st.button("🔥 Ingest URL", key="ingest_url_btn", use_container_width=True):
        if url_input:
            if url_input not in [d["name"] for d in st.session_state.ingested_docs]:
                with st.spinner(f"Scraping & ingesting {url_input}..."):
                    try:
                        chunks = ingest_url(url_input)
                        st.session_state.ingested_docs.append({
                            "name": url_input,
                            "chunks": len(chunks),
                            "type": "url",
                        })
                        st.success(f"✅ URL ingested — {len(chunks)} chunks stored")
                    except Exception as e:
                        st.error(f"❌ URL ingestion failed: {str(e)}")
            else:
                st.info("URL already ingested")
        else:
            st.warning("Please enter a URL")

    # ── Ingested Documents List ─────────────────────────────────────────
    if st.session_state.ingested_docs:
        st.divider()
        st.markdown("##### 📚 Ingested Documents")
        for doc in st.session_state.ingested_docs:
            icon = "📄" if doc["type"] == "file" else "🌐"
            name = doc["name"][:35] + "..." if len(doc["name"]) > 35 else doc["name"]
            st.caption(f"{icon} {name} ({doc['chunks']} chunks)")

    # ── Memory Management ───────────────────────────────────────────────
    st.divider()
    st.markdown("##### ⚙️ Settings")

    if st.button("🗑️ Reset Session", key="clear_memory", use_container_width=True):
        st.session_state.chat_history = []
        new_thread = str(uuid.uuid4())
        st.session_state.thread_id = new_thread
        st.query_params["thread_id"] = new_thread
        st.session_state.last_trace = None
        st.success("Session reset completed")
        st.rerun()

    st.caption(f"Session: `{st.session_state.thread_id[:8]}...`")


# ═══════════════════════════════════════════════════════════════════════════
#  Main Chat Interface
# ═══════════════════════════════════════════════════════════════════════════

# Header
st.markdown("## 💬 Chat with your Documents")

if not st.session_state.ingested_docs:
    st.info("👈 Upload a document or ingest a URL to get started.")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Agents are thinking..."):
            try:
                result = run_query(
                    query=prompt,
                    thread_id=st.session_state.thread_id,
                )
                response = result["final_answer"]
                st.session_state.last_trace = result
            except Exception as e:
                logger.error(f"Query failed: {e}", exc_info=True)
                response = f"⚠️ An error occurred: {str(e)}"
                st.session_state.last_trace = None

        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # ── Agent Trace Expander ────────────────────────────────────────────
    if st.session_state.last_trace:
        trace = st.session_state.last_trace

        with st.expander("🔍 Agent Trace", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Intent:** `{trace.get('intent', 'unknown')}`")
                mcp = trace.get("mcp_context", {})
                st.markdown(f"**MCP Status:** `{mcp.get('status', 'n/a')}`")

            with col2:
                chunks = trace.get("retrieved_chunks", [])
                st.markdown(f"**Chunks Retrieved:** `{len(chunks)}`")

            # Show retrieved chunks with scores
            if chunks:
                st.markdown("---")
                st.markdown("**📄 Retrieved Chunks:**")
                for i, chunk in enumerate(chunks):
                    score = chunk.get("similarity_score", 0)
                    source = chunk.get("source", "unknown")
                    idx = chunk.get("chunk_index", 0)
                    text_preview = chunk.get("text", "")[:300]

                    score_color = "🟢" if score >= 0.8 else "🟡" if score >= 0.65 else "🔴"

                    with st.expander(
                        f"{score_color} [{source} · {idx}] — Score: {score:.3f}",
                        expanded=False,
                    ):
                        st.text(text_preview + ("..." if len(chunk.get("text", "")) > 300 else ""))

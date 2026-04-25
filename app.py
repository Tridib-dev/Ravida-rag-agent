"""
app.py
------
Entry point for the Ravida Streamlit application.

Run with:
    streamlit run app.py

Architecture overview
─────────────────────
app.py          ← orchestrates startup, upload detection, and the chat loop
│
├── core/
│   ├── config.py             ← constants, env vars, LLM factories
│   ├── vector_store.py       ← Chroma + FastEmbed lifecycle
│   ├── document_processor.py ← PDF → chunks → vector store
│   ├── tools.py              ← search_query LangChain tool factory
│   └── agent.py              ← agent creation + streaming events
│
├── utils/
│   ├── session_state.py      ← typed accessors for st.session_state
│   └── question_generator.py ← Gemini-powered example question generation
│
└── ui/
    ├── components.py         ← reusable Streamlit widgets
    └── chat.py               ← full chat-turn handler (stream + render)
"""

import streamlit as st


# ── Core layer ────────────────────────────────────────────────────────────────
from core.vector_store import get_embedder, reset_vector_store
from core.document_processor import ingest
from core.tools import make_search_tool
from core.agent import build_agent


# ── Utilities ─────────────────────────────────────────────────────────────────
from utils.session_state import (
    get_current_file,
    set_current_file,
    is_file_processed,
    mark_file_processed,
    reset_file_state,
    set_agent,
    get_example_questions,
    set_example_questions,
)
from utils.question_generator import generate_example_questions

# ── UI layer ──────────────────────────────────────────────────────────────────
from ui.components import (
    render_header,
    render_file_uploader,
    render_chat_history,
    render_followup_questions,
    show_success,
    show_embedder_error,
)
from ui.chat import answer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ravida – Medical RAG Agent", page_icon="🤖")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
render_header()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model (session-cached; shown once with a spinner)
# ─────────────────────────────────────────────────────────────────────────────
try:
    embedder = get_embedder()
except Exception:
    show_embedder_error()
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = render_file_uploader()
if not uploaded_file:
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Detect a new file upload – reset all file-specific state
# ─────────────────────────────────────────────────────────────────────────────
if get_current_file() != uploaded_file.name:
    set_current_file(uploaded_file.name)
    reset_file_state()
    reset_vector_store(embedder)


# ─────────────────────────────────────────────────────────────────────────────
# Process the PDF (only once per file)
# ─────────────────────────────────────────────────────────────────────────────
if not is_file_processed():
    with st.spinner(f"Processing {uploaded_file.name}…"):
        # 1. Ingest: load → split → embed
        from core.vector_store import get_vector_store

        vector_store = get_vector_store()
        chunks = ingest(uploaded_file, vector_store)

        # 2. Generate example questions (once per file)
        if get_example_questions() is None:
            questions = generate_example_questions(chunks)
            set_example_questions(questions)

        # 3. Build the agent with the populated vector store
        search_tool = make_search_tool(vector_store)
        agent = build_agent(tools=[search_tool])
        set_agent(agent)

        mark_file_processed()

    show_success(uploaded_file.name)


# ─────────────────────────────────────────────────────────────────────────────
# Chat interface
# ─────────────────────────────────────────────────────────────────────────────
render_chat_history()

if is_file_processed():
    user_input = st.chat_input("Ask a question about the medical report:")
    if user_input:
        answer(user_input)

followup_q = render_followup_questions()
if followup_q:
    answer(followup_q)
    st.rerun()

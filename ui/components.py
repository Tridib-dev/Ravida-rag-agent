"""
ui/components.py
----------------
Reusable Streamlit UI components.

Each function is responsible for rendering exactly one piece of the interface.
The main app assembles these components; none of them contain business logic.
"""

import random
import streamlit as st

from utils.session_state import (
    get_history,
    clear_history,
    get_example_questions,
    is_file_processed,
)


# ── Header ────────────────────────────────────────────────────────────────────


def render_header() -> None:
    """Top bar: title on the left, 'Clear chat' button on the right."""

    def _on_clear():
        clear_history()
        st.toast("Chat cleared", icon="🧹")

    with st.container(horizontal=True):
        c1, _, c3 = st.columns(3)
        with c1:
            c1.title("🤖 Ravida")
            c1.caption("RAG Powered Research Agent")
        with c3:
            c3.space("stretch")
            c3.button("Clear chat", on_click=_on_clear)
            st.space("small")

    st.divider()


# ── File uploader ─────────────────────────────────────────────────────────────


def render_file_uploader():
    """
    Render the PDF file-uploader widget inside a bordered container.

    Returns
    -------
    The uploaded file object, or None if nothing has been uploaded yet.
    """
    with st.container(border=True, width="stretch", horizontal=False):
        uploaded = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            key="pdf_uploader",
            accept_multiple_files=False,
        )
        if not uploaded:
            st.warning("Please upload a PDF file to proceed.")
    return uploaded


# ── Chat history ──────────────────────────────────────────────────────────────


def render_chat_history() -> None:
    """Re-render all messages stored in session history."""
    for msg in get_history():
        st.chat_message(msg["role"]).markdown(msg["content"])


# ── Suggested follow-up questions ─────────────────────────────────────────────


def render_followup_questions() -> None:
    """
    Show a random sample of 4 example questions as clickable buttons.

    Parameters
    ----------
    on_question_click : Callable[[str], None]
        Called with the question text when the user clicks a button.
    """
    questions = get_example_questions()
    if not questions:
        return

    history = get_history()
    last_role = history[-1]["role"] if history else None

    if not is_file_processed() or last_role != "assistant":
        return

    with st.container(border=True):
        st.markdown("**💡 Related questions:**")
        sample = random.sample(questions, min(4, len(questions)))
        for idx, question in enumerate(sample):
            if st.button(
                f"↗ {question}",
                key=f"followup_{idx}",
                use_container_width=True,
            ):
                clicked = question
                return clicked


# ── Processing status ─────────────────────────────────────────────────────────


def show_processing_spinner(file_name: str):
    """Return a context manager that shows a spinner while the PDF is processed."""
    return st.spinner(f"Processing {file_name}…")


def show_success(file_name: str) -> None:
    st.success(f"[{file_name}] Processed successfully!")


def show_embedder_error() -> None:
    st.error("Failed to load embedding model. Check your internet connection.")

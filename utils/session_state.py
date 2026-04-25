"""
utils/session_state.py
-----------------------
Thin wrappers around st.session_state to give the UI clean, named accessors
instead of raw dict-style lookups scattered throughout the codebase.

Every key used in session_state is defined here, making it trivial to audit
what state the app holds at any point.
"""

import streamlit as st


# ── Chat history ──────────────────────────────────────────────────────────────

def get_history() -> list[dict]:
    return st.session_state.setdefault("history", [])


def append_message(role: str, content: str) -> None:
    get_history().append({"role": role, "content": content})


def clear_history() -> None:
    st.session_state.history = []


# ── File tracking ─────────────────────────────────────────────────────────────

def get_current_file() -> str | None:
    return st.session_state.get("current_file")


def set_current_file(name: str) -> None:
    st.session_state.current_file = name


def is_file_processed() -> bool:
    return st.session_state.get("file_processed", False)


def mark_file_processed() -> None:
    st.session_state.file_processed = True


def reset_file_state() -> None:
    """Clear all file-specific state (called on new upload)."""
    clear_history()
    st.session_state.pop("file_processed", None)
    st.session_state.pop("agent", None)
    st.session_state.pop("example_questions", None)


# ── Agent ─────────────────────────────────────────────────────────────────────

def get_agent():
    return st.session_state.get("agent")


def set_agent(agent) -> None:
    st.session_state.agent = agent


# ── Example questions ─────────────────────────────────────────────────────────

def get_example_questions() -> list[str] | None:
    return st.session_state.get("example_questions")


def set_example_questions(questions: list[str]) -> None:
    st.session_state.example_questions = questions

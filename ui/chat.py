import streamlit as st

from core.agent import (
    stream_agent_response,
    ToolCallEvent,
    ToolResultEvent,
    TextChunkEvent,
)
from utils.session_state import get_agent, append_message

_CONTEXT_ERRORS = (
    "message too large",
    "context_length_exceeded",
    "maximum context length",
)


def answer(question: str) -> None:
    append_message("user", question)
    st.chat_message("user").markdown(question)

    agent = get_agent()
    if agent is None:
        st.error("Agent not initialised. Please upload a PDF first.")
        return

    try:
        final_text = _stream_and_render(agent, question)
        append_message("assistant", final_text)
    except Exception as exc:
        msg = str(exc)
        if any(m in msg for m in _CONTEXT_ERRORS):
            st.error("Conversation too long. Click **Clear chat** to start fresh.")
        else:
            st.error(f"An error occurred: {msg}")


def _stream_and_render(agent, question: str) -> str:
    execution_steps: list[str] = []
    final_text = ""

    with st.chat_message("assistant"):
        with st.status("🔍 Thinking…", expanded=True) as status:
            for event in stream_agent_response(agent, question):
                if isinstance(event, ToolCallEvent):
                    status.update(label=f"🔎 Searching: {event.query}…")
                    execution_steps.append(f"• 🔎 Searching: {event.query}")

                elif isinstance(event, ToolResultEvent):
                    status.update(label="📄 Processing results…")
                    if (
                        not execution_steps
                        or "Got search results" not in execution_steps[-1]
                    ):
                        execution_steps.append("• 📄 Got search results, analysing…")

                elif isinstance(event, TextChunkEvent):
                    status.update(label="✍️ Writing answer…")
                    final_text += event.content

            if execution_steps:
                for step in execution_steps:
                    st.markdown(step)
            else:
                st.markdown("• Analysed prompt directly, no search needed")

        status.update(label="✅ Done!", state="complete", expanded=False)
        st.markdown(final_text)

    return final_text

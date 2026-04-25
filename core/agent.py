"""
core/agent.py
-------------
Creates the LangGraph ReAct agent and exposes a streaming helper.

Responsibilities
────────────────
• Build the agent with `create_agent` + `MemorySaver`.
• Provide `stream_agent_response()` – a generator that yields structured
  events (tool-call, tool-result, text-chunk) so the UI layer can display
  real-time progress without knowing anything about LangGraph internals.

"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Generator

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from core.config import AGENT_SYSTEM_PROMPT, get_agent_llm


# ── Event dataclasses ─────────────────────────────────────────────────────────


@dataclass
class ToolCallEvent:
    query: str


@dataclass
class ToolResultEvent:
    pass


@dataclass
class TextChunkEvent:
    content: str


AgentEvent = ToolCallEvent | ToolResultEvent | TextChunkEvent


# ── Agent factory ─────────────────────────────────────────────────────────────


def build_agent(tools: list):
    llm = get_agent_llm()
    return create_agent(
        model=llm,
        checkpointer=MemorySaver(),
        system_prompt=AGENT_SYSTEM_PROMPT,
        tools=tools,
    )


# ── Streaming helper ──────────────────────────────────────────────────────────


def stream_agent_response(
    agent,
    question: str,
) -> Generator[AgentEvent, None, None]:
    """
    Yield typed events in real time as the agent works.

    Fresh thread_id per call — prevents MemorySaver context overflow.
    No retry here — caller (chat.py) handles retries around the for loop.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    input_msg = {"messages": [{"role": "user", "content": question}]}

    stream = agent.stream(input_msg, config, stream_mode="messages")

    for chunk in stream:
        msg = chunk[0]
        msg_type = type(msg).__name__

        if (
            msg_type == "AIMessageChunk"
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        ):
            query = msg.tool_calls[0].get("args", {}).get("query", "")
            yield ToolCallEvent(query=query)

        elif msg_type == "ToolMessage":
            yield ToolResultEvent()

        elif msg_type == "AIMessageChunk" and msg.content:
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            if content:
                yield TextChunkEvent(content=content)

"""Graph for processing email against existing user memories."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent import prompts, utils
from memory_agent.context import Context
from memory_agent.memory_backends import format_memory, memory_backend_from_runtime


logger = logging.getLogger(__name__)

MAX_RETRIEVAL_QUERY_CHARS = 400
MAX_ANALYSIS_BODY_CHARS = 1_500
MAX_RELATED_MEMORY_CHARS = 1_000


@dataclass(kw_only=True)
class EmailState:
    """State for the email processing graph."""

    email_id: str
    """Stable external ID for the email."""

    thread_id: str | None = None
    """Stable conversation/thread ID for related emails."""

    sender: str
    """Email sender address or display name."""

    subject: str
    """Email subject."""

    body: str
    """Raw email body text or HTML."""

    received_at: str | None = None
    """When the email was received, if known."""

    normalized_body: str = ""
    """Plain text body after lightweight cleanup."""

    related_memories: list[str] = field(default_factory=list)
    """Existing memories retrieved for context."""

    email_preferences: list[str] = field(default_factory=list)
    """User preference memories relevant to email replies."""

    summary: str | None = None
    classification: str | None = None
    key_info: dict[str, Any] = field(default_factory=dict)
    action: str | None = None
    draft_reply: str | None = None
    reply_confirmation_required: bool = False
    reply_status: str | None = None


def _strip_html(value: str) -> str:
    """Convert simple HTML-ish email content to compact plain text."""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate_text(value: str, limit: int) -> str:
    """Keep model and embedding inputs inside small context windows."""
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n...[truncated]"


def _normalize_subject(subject: str) -> str:
    """Produce a stable fallback thread key from a mail subject."""
    normalized = subject.strip().lower()
    normalized = re.sub(r"^(\s*(re|fw|fwd)\s*:\s*)+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized or "untitled"


def _derive_thread_id(state: EmailState) -> str:
    """Use explicit thread IDs first, falling back to normalized subjects."""
    return state.thread_id or f"subject:{_normalize_subject(state.subject)}"


def _coerce_analysis(value: str) -> dict[str, Any]:
    """Parse model output into the JSON object expected by the graph."""
    try:
        return cast(dict[str, Any], json.loads(value))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if not match:
            raise
        return cast(dict[str, Any], json.loads(match.group(0)))


async def normalize_email(state: EmailState) -> dict[str, str]:
    """Clean the raw email body before model analysis."""
    return {
        "normalized_body": _strip_html(state.body),
        "received_at": state.received_at or datetime.now().isoformat(),
        "thread_id": _derive_thread_id(state),
    }


async def retrieve_related_memories(
    state: EmailState, runtime: Runtime[Context]
) -> dict[str, list[str]]:
    """Fetch memories that may help interpret the email."""
    memory_backend = memory_backend_from_runtime(
        runtime.context,
        cast(BaseStore | None, runtime.store),
    )
    if memory_backend is None:
        return {"related_memories": []}

    query = _truncate_text(
        "\n".join([state.sender, state.subject, state.normalized_body]),
        MAX_RETRIEVAL_QUERY_CHARS,
    )
    memories = await memory_backend.search(
        user_id=runtime.context.user_id,
        query=query,
        limit=8,
    )

    formatted = [format_memory(mem) for mem in memories]
    return {"related_memories": formatted}


async def retrieve_email_preferences(
    state: EmailState, runtime: Runtime[Context]
) -> dict[str, list[str]]:
    """Fetch user preference memories for reply drafting context."""
    memory_backend = memory_backend_from_runtime(
        runtime.context,
        cast(BaseStore | None, runtime.store),
    )
    if memory_backend is None:
        return {"email_preferences": []}

    query = _truncate_text(
        "\n".join([state.sender, state.subject, "email reply preferences"]),
        MAX_RETRIEVAL_QUERY_CHARS,
    )
    memories = await memory_backend.search(
        user_id=runtime.context.user_id,
        query=query,
        limit=8,
    )
    preferences = []
    for memory in memories:
        content = str(memory.value.get("content", ""))
        context = str(memory.value.get("context", ""))
        if "preference" not in f"{content} {context}".lower():
            continue
        preferences.append(format_memory(memory))
    return {"email_preferences": preferences}


async def analyze_email(
    state: EmailState, runtime: Runtime[Context]
) -> dict[str, Any]:
    """Classify the email and decide the next action."""
    related = _truncate_text(
        "\n".join(state.related_memories),
        MAX_RELATED_MEMORY_CHARS,
    ) or "None"
    system_prompt = prompts.EMAIL_ANALYSIS_PROMPT.format(
        time=datetime.now().isoformat(),
        related_memories=related,
        email_preferences=_truncate_text("\n".join(state.email_preferences), MAX_RELATED_MEMORY_CHARS)
        or "None",
    )
    body = _truncate_text(state.normalized_body, MAX_ANALYSIS_BODY_CHARS)
    user_prompt = f"""Email ID: {state.email_id}
Thread ID: {state.thread_id}
From: {state.sender}
Received at: {state.received_at}
Subject: {state.subject}

Body:
{body}
"""

    llm = utils.load_chat_model(runtime.context.model)
    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    content = response.content
    if not isinstance(content, str):
        content = json.dumps(content)

    try:
        analysis = _coerce_analysis(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse email analysis JSON: %s", content)
        analysis = {
            "summary": state.subject,
            "classification": "unknown",
            "key_info": {},
            "action": "ask_user",
            "draft_reply": None,
        }

    return {
        "summary": analysis.get("summary"),
        "classification": analysis.get("classification", "unknown"),
        "key_info": analysis.get("key_info") or {},
        "action": analysis.get("action", "ask_user"),
        "draft_reply": analysis.get("draft_reply"),
    }


async def decide_action(state: EmailState) -> dict[str, str]:
    """Normalize the action into the supported set."""
    supported = {"draft_reply", "create_todo", "archive", "ignore", "ask_user"}
    action = state.action if state.action in supported else "ask_user"
    return {"action": action}


async def draft_reply(state: EmailState, runtime: Runtime[Context]) -> dict[str, str]:
    """Create a reply draft when the analysis says a response is useful."""
    if state.draft_reply:
        return {"draft_reply": state.draft_reply}

    llm = utils.load_chat_model(runtime.context.model)
    body = _truncate_text(state.normalized_body, MAX_ANALYSIS_BODY_CHARS)
    response = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": "Draft a concise, helpful email reply. Return only the draft body.",
            },
            {
                "role": "user",
                "content": (
                    f"From: {state.sender}\n"
                    f"Subject: {state.subject}\n"
                    f"Summary: {state.summary}\n"
                    f"Key info: {json.dumps(state.key_info, ensure_ascii=False)}\n\n"
                    f"Email body:\n{body}"
                ),
            },
        ]
    )
    return {"draft_reply": str(response.content)}


async def request_reply_confirmation(state: EmailState) -> dict[str, str | bool | None]:
    """Mark drafted replies as waiting for explicit user approval."""
    if not state.draft_reply:
        return {
            "reply_confirmation_required": False,
            "reply_status": None,
        }
    return {
        "reply_confirmation_required": True,
        "reply_status": "pending_user_confirmation",
    }


def route_after_decision(state: EmailState):
    """Route to a drafting step only when a reply is needed."""
    if state.action == "draft_reply":
        return "draft_reply"
    return END


builder = StateGraph(EmailState, context_schema=Context)

builder.add_node(normalize_email)
builder.add_node(retrieve_related_memories)
builder.add_node(retrieve_email_preferences)
builder.add_node(analyze_email)
builder.add_node(decide_action)
builder.add_node(draft_reply)
builder.add_node(request_reply_confirmation)

builder.add_edge("__start__", "normalize_email")
builder.add_edge("normalize_email", "retrieve_related_memories")
builder.add_edge("retrieve_related_memories", "retrieve_email_preferences")
builder.add_edge("retrieve_email_preferences", "analyze_email")
builder.add_edge("analyze_email", "decide_action")
builder.add_conditional_edges("decide_action", route_after_decision, ["draft_reply", END])
builder.add_edge("draft_reply", "request_reply_confirmation")
builder.add_edge("request_reply_confirmation", END)

email_graph = builder.compile()
email_graph.name = "EmailMemoryAgent"


__all__ = ["EmailState", "builder", "email_graph"]

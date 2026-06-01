"""Graph for processing email and saving durable user memories."""

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

from memory_agent import prompts, tools, utils
from memory_agent.context import Context


logger = logging.getLogger(__name__)

MAX_RETRIEVAL_QUERY_CHARS = 400
MAX_ANALYSIS_BODY_CHARS = 1_500
MAX_RELATED_MEMORY_CHARS = 1_000
MAX_THREAD_CONTEXT_CHARS = 2_000
MAX_THREAD_HISTORY_ITEMS = 12
MIN_MEMORY_CONFIDENCE = 0.75
LONG_TERM_MEMORY_TYPES = {"preference", "contact", "project", "rule", "account"}
LONG_TERM_MEMORY_SCOPES = {"user", "contact", "project", "account"}
TRANSIENT_MEMORY_PATTERNS = (

    r"\bnewsletter|unsubscribe|receipt|invoice|tracking number\b",
)


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

    thread_context: list[str] = field(default_factory=list)
    """Earlier emails from the same conversation thread."""

    summary: str | None = None
    classification: str | None = None
    key_info: dict[str, Any] = field(default_factory=dict)
    memories_to_save: list[dict[str, str]] = field(default_factory=list)
    stored_memories: list[str] = field(default_factory=list)
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


def _thread_namespace(user_id: str, thread_id: str) -> tuple[str, str, str]:
    """Namespace used for per-user, per-email-thread history."""
    return ("email_threads", user_id, thread_id)


def _coerce_analysis(value: str) -> dict[str, Any]:
    """Parse model output into the JSON object expected by the graph."""
    try:
        return cast(dict[str, Any], json.loads(value))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if not match:
            raise
        return cast(dict[str, Any], json.loads(match.group(0)))


def _memory_items(value: Any) -> list[dict[str, Any]]:
    """Normalize model-produced memory items into tool arguments."""
    if not isinstance(value, list):
        return []

    memories: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            memories.append(
                {
                    "content": item.strip(),
                    "context": "Extracted from email.",
                    "type": "unknown",
                    "confidence": 0.0,
                    "scope": "unknown",
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        content = str(item.get("content", "")).strip()
        if not content:
            continue

        context = str(item.get("context", "")).strip() or "Extracted from email."
        memories.append(
            {
                "content": content,
                "context": context,
                "type": str(item.get("type", "")).strip().lower(),
                "confidence": _coerce_confidence(item.get("confidence")),
                "scope": str(item.get("scope", "")).strip().lower(),
            }
        )

    return memories


def _coerce_confidence(value: Any) -> float:
    """Convert model confidence values to a conservative numeric score."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _is_long_term_memory_candidate(memory: dict[str, Any], classification: str | None) -> bool:
    """Return whether a model-proposed memory passes deterministic save gates."""
    if classification in {"newsletter", "junk"}:
        return False

    content = str(memory.get("content", "")).strip()
    if not content:
        return False

    if memory.get("type") not in LONG_TERM_MEMORY_TYPES:
        return False

    if memory.get("scope") not in LONG_TERM_MEMORY_SCOPES:
        return False

    if _coerce_confidence(memory.get("confidence")) < MIN_MEMORY_CONFIDENCE:
        return False

    lowered = content.lower()
    return not any(re.search(pattern, lowered) for pattern in TRANSIENT_MEMORY_PATTERNS)


def _filter_long_term_memories(
    memories: list[dict[str, Any]], classification: str | None
) -> list[dict[str, str]]:
    """Keep only high-confidence facts suitable for durable memory."""
    filtered = []
    for memory in memories:
        if not _is_long_term_memory_candidate(memory, classification):
            continue

        context = str(memory["context"]).strip()
        metadata = (
            f"type={memory['type']}; "
            f"scope={memory['scope']}; "
            f"confidence={_coerce_confidence(memory['confidence']):.2f}"
        )
        filtered.append(
            {
                "content": str(memory["content"]).strip(),
                "context": f"{context} ({metadata})",
            }
        )
    return filtered


async def normalize_email(state: EmailState) -> dict[str, str]:
    """Clean the raw email body before model analysis."""
    return {
        "normalized_body": _strip_html(state.body),
        "received_at": state.received_at or datetime.now().isoformat(),
        "thread_id": _derive_thread_id(state),
    }


async def retrieve_thread_context(
    state: EmailState, runtime: Runtime[Context]
) -> dict[str, list[str]]:
    """Fetch earlier emails from the same email conversation."""
    if runtime.store is None or not state.thread_id:
        return {"thread_context": []}

    history = await cast(BaseStore, runtime.store).asearch(
        _thread_namespace(runtime.context.user_id, state.thread_id),
        limit=MAX_THREAD_HISTORY_ITEMS,
    )

    items = []
    for item in history:
        if item.key == state.email_id or not isinstance(item.value, dict):
            continue
        items.append(item.value)

    items.sort(key=lambda value: str(value.get("received_at", "")))
    formatted = [
        (
            f"[{item.get('received_at', 'unknown time')}] "
            f"{item.get('sender', 'unknown sender')} | "
            f"{item.get('subject', 'no subject')}: "
            f"{item.get('summary') or item.get('body_excerpt', '')}"
        )
        for item in items[-MAX_THREAD_HISTORY_ITEMS:]
    ]
    return {"thread_context": formatted}


async def retrieve_related_memories(
    state: EmailState, runtime: Runtime[Context]
) -> dict[str, list[str]]:
    """Fetch memories that may help interpret the email."""
    if runtime.store is None:
        return {"related_memories": []}

    query = _truncate_text(
        "\n".join([state.sender, state.subject, state.normalized_body]),
        MAX_RETRIEVAL_QUERY_CHARS,
    )
    memories = await cast(BaseStore, runtime.store).asearch(
        ("memories", runtime.context.user_id),
        query=query,
        limit=8,
    )

    formatted = [
        f"[{mem.key}]: {mem.value}"
        for mem in memories
    ]
    return {"related_memories": formatted}


async def analyze_email(
    state: EmailState, runtime: Runtime[Context]
) -> dict[str, Any]:
    """Classify the email and extract durable memory candidates."""
    related = _truncate_text(
        "\n".join(state.related_memories),
        MAX_RELATED_MEMORY_CHARS,
    ) or "None"
    system_prompt = prompts.EMAIL_ANALYSIS_PROMPT.format(
        time=datetime.now().isoformat(),
        related_memories=related,
        email_preferences="None",
        thread_context=_truncate_text("\n".join(state.thread_context), MAX_THREAD_CONTEXT_CHARS)
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
            "memories_to_save": [],
            "action": "ask_user",
            "draft_reply": None,
        }

    return {
        "summary": analysis.get("summary"),
        "classification": analysis.get("classification", "unknown"),
        "key_info": analysis.get("key_info") or {},
        "memories_to_save": _filter_long_term_memories(
            _memory_items(analysis.get("memories_to_save")),
            analysis.get("classification", "unknown"),
        ),
        "action": analysis.get("action", "ask_user"),
        "draft_reply": analysis.get("draft_reply"),
    }


async def store_memory(
    state: EmailState, runtime: Runtime[Context]
) -> dict[str, list[str]]:
    """Persist extracted long-term memories."""
    if runtime.store is None or not state.memories_to_save:
        return {"stored_memories": []}

    stored: list[str] = []
    for memory in state.memories_to_save:
        result = await tools.upsert_memory.ainvoke(
            {
                "content": memory["content"],
                "context": memory["context"],
                "user_id": runtime.context.user_id,
                "store": cast(BaseStore, runtime.store),
            }
        )
        stored.append(result)

    return {"stored_memories": stored}


async def store_thread_context(
    state: EmailState, runtime: Runtime[Context]
) -> dict[str, Any]:
    """Append the current email's compact summary to its thread context."""
    if runtime.store is None or not state.thread_id:
        return {}

    await cast(BaseStore, runtime.store).aput(
        _thread_namespace(runtime.context.user_id, state.thread_id),
        key=state.email_id,
        value={
            "email_id": state.email_id,
            "sender": state.sender,
            "subject": state.subject,
            "received_at": state.received_at,
            "summary": state.summary or state.subject,
            "classification": state.classification,
            "key_info": state.key_info,
            "body_excerpt": _truncate_text(state.normalized_body, 500),
        },
    )
    return {}


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
builder.add_node(retrieve_thread_context)
builder.add_node(retrieve_related_memories)
builder.add_node(analyze_email)
builder.add_node(store_memory)
builder.add_node(store_thread_context)
builder.add_node(decide_action)
builder.add_node(draft_reply)
builder.add_node(request_reply_confirmation)

builder.add_edge("__start__", "normalize_email")
builder.add_edge("normalize_email", "retrieve_thread_context")
builder.add_edge("retrieve_thread_context", "retrieve_related_memories")
builder.add_edge("retrieve_related_memories", "analyze_email")
builder.add_edge("analyze_email", "store_memory")
builder.add_edge("store_memory", "store_thread_context")
builder.add_edge("store_thread_context", "decide_action")
builder.add_conditional_edges("decide_action", route_after_decision, ["draft_reply", END])
builder.add_edge("draft_reply", "request_reply_confirmation")
builder.add_edge("request_reply_confirmation", END)

email_graph = builder.compile()
email_graph.name = "EmailMemoryAgent"


__all__ = ["EmailState", "builder", "email_graph"]

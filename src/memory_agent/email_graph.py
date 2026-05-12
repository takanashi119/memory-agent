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

from memory_agent import tools, utils
from memory_agent.context import Context


logger = logging.getLogger(__name__)


EMAIL_ANALYSIS_PROMPT = """You process a single email for a personal memory agent.

Your job:
1. Summarize the email.
2. Classify the email.
3. Extract key information that may matter later.
4. Decide which facts are worth storing as long-term memories.
5. Recommend the next action.

Only store stable, useful facts. Do not store one-time codes, credentials,
advertisements, newsletters, or full email bodies.

Return strict JSON with this shape:
{{
  "summary": "short summary",
  "classification": "important|meeting|task|personal_info|newsletter|low_priority|unknown",
  "key_info": {{}},
  "memories_to_save": [
    {{"content": "durable memory", "context": "why this was inferred from this email"}}
  ],
  "action": "draft_reply|create_todo|archive|ignore|ask_user",
  "draft_reply": null
}}

Current time: {time}

Related existing memories:
{related_memories}
"""


@dataclass(kw_only=True)
class EmailState:
    """State for the email processing graph."""

    email_id: str
    """Stable external ID for the email."""

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

    summary: str | None = None
    classification: str | None = None
    key_info: dict[str, Any] = field(default_factory=dict)
    memories_to_save: list[dict[str, str]] = field(default_factory=list)
    stored_memories: list[str] = field(default_factory=list)
    action: str | None = None
    draft_reply: str | None = None


def _strip_html(value: str) -> str:
    """Convert simple HTML-ish email content to compact plain text."""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _coerce_analysis(value: str) -> dict[str, Any]:
    """Parse model output into the JSON object expected by the graph."""
    try:
        return cast(dict[str, Any], json.loads(value))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if not match:
            raise
        return cast(dict[str, Any], json.loads(match.group(0)))


def _memory_items(value: Any) -> list[dict[str, str]]:
    """Normalize model-produced memory items into tool arguments."""
    if not isinstance(value, list):
        return []

    memories: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            memories.append({"content": item.strip(), "context": "Extracted from email."})
            continue

        if not isinstance(item, dict):
            continue

        content = str(item.get("content", "")).strip()
        if not content:
            continue

        context = str(item.get("context", "")).strip() or "Extracted from email."
        memories.append({"content": content, "context": context})

    return memories


async def normalize_email(state: EmailState) -> dict[str, str]:
    """Clean the raw email body before model analysis."""
    return {
        "normalized_body": _strip_html(state.body),
        "received_at": state.received_at or datetime.now().isoformat(),
    }


async def retrieve_related_memories(
    state: EmailState, runtime: Runtime[Context]
) -> dict[str, list[str]]:
    """Fetch memories that may help interpret the email."""
    if runtime.store is None:
        return {"related_memories": []}

    query = "\n".join([state.sender, state.subject, state.normalized_body])
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
    related = "\n".join(state.related_memories) or "None"
    system_prompt = EMAIL_ANALYSIS_PROMPT.format(
        time=datetime.now().isoformat(),
        related_memories=related,
    )
    user_prompt = f"""Email ID: {state.email_id}
From: {state.sender}
Received at: {state.received_at}
Subject: {state.subject}

Body:
{state.normalized_body}
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
        "memories_to_save": _memory_items(analysis.get("memories_to_save")),
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
        result = await tools.upsert_memory(
            content=memory["content"],
            context=memory["context"],
            user_id=runtime.context.user_id,
            store=cast(BaseStore, runtime.store),
        )
        stored.append(result)

    return {"stored_memories": stored}


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
                    f"Email body:\n{state.normalized_body}"
                ),
            },
        ]
    )
    return {"draft_reply": str(response.content)}


def route_after_decision(state: EmailState):
    """Route to a drafting step only when a reply is needed."""
    if state.action == "draft_reply":
        return "draft_reply"
    return END


builder = StateGraph(EmailState, context_schema=Context)

builder.add_node(normalize_email)
builder.add_node(retrieve_related_memories)
builder.add_node(analyze_email)
builder.add_node(store_memory)
builder.add_node(decide_action)
builder.add_node(draft_reply)

builder.add_edge("__start__", "normalize_email")
builder.add_edge("normalize_email", "retrieve_related_memories")
builder.add_edge("retrieve_related_memories", "analyze_email")
builder.add_edge("analyze_email", "store_memory")
builder.add_edge("store_memory", "decide_action")
builder.add_conditional_edges("decide_action", route_after_decision, ["draft_reply", END])
builder.add_edge("draft_reply", END)

email_graph = builder.compile()
email_graph.name = "EmailMemoryAgent"


__all__ = ["EmailState", "builder", "email_graph"]

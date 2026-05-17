"""Email processing services shared by inbox listeners and CLIs."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from langgraph.store.memory import InMemoryStore

from memory_agent.context import Context
from memory_agent.email_graph import builder


logger = logging.getLogger(__name__)


class ProcessedMessageStore(Protocol):
    """Tracks external message IDs that have already been handled."""

    def contains(self, message_id: str) -> bool:
        """Return whether a message ID was already processed."""

    def add(self, message_id: str) -> None:
        """Mark a message ID as processed."""


class EmailNotifier(Protocol):
    """Sends processing results to a user-visible output channel."""

    async def notify(self, result: dict[str, Any]) -> None:
        """Publish a processed email result."""


class JsonProcessedMessageStore:
    """Persist processed message IDs in a local JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._processed_ids = self._load()

    def contains(self, message_id: str) -> bool:
        """Return whether a message ID was already processed."""
        return message_id in self._processed_ids

    def add(self, message_id: str) -> None:
        """Mark a message ID as processed."""
        self._processed_ids.add(message_id)
        self._save()

    def _load(self) -> set[str]:
        if not self.path.exists():
            return set()

        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid processed-message state file: %s", self.path)
            return set()

        if isinstance(value, list):
            return {str(item) for item in value}
        if isinstance(value, dict) and isinstance(value.get("processed_ids"), list):
            return {str(item) for item in value["processed_ids"]}
        return set()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().isoformat(),
            "processed_ids": sorted(self._processed_ids),
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class ConsoleEmailNotifier:
    """Write processed email summaries to stdout."""

    async def notify(self, result: dict[str, Any]) -> None:
        """Publish a processed email result."""
        sys.stdout.write(format_email_feedback(result) + "\n")
        sys.stdout.flush()


@dataclass(kw_only=True)
class EmailProcessingService:
    """Run email payloads through the email graph and publish results."""

    processed_store: ProcessedMessageStore
    notifier: EmailNotifier
    user_id: str = "default"
    model: str | None = None

    def __post_init__(self) -> None:
        """Build the graph runtime dependencies."""
        self._store = InMemoryStore()
        self._graph = builder.compile(store=self._store)
        context_kwargs: dict[str, Any] = {"user_id": self.user_id}
        if self.model:
            context_kwargs["model"] = self.model
        self._context = Context(**context_kwargs)

    def is_processed(self, email: dict[str, Any]) -> bool:
        """Return whether an email payload was already processed."""
        return self.processed_store.contains(str(email["email_id"]))

    async def process(self, email: dict[str, Any]) -> dict[str, Any]:
        """Process one email, persist its processed marker, and notify the user."""
        result = await self._graph.ainvoke(
            email,
            config={"configurable": {"thread_id": f"gmail:{email['email_id']}"}},
            context=self._context,
        )
        self.processed_store.add(str(email["email_id"]))
        await self.notifier.notify(result)
        return result


def format_email_feedback(result: dict[str, Any]) -> str:
    """Format one processed email result for human-readable output."""
    stored = result.get("stored_memories") or []
    draft_reply = result.get("draft_reply")
    lines = [
        "",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed email",
        f"  id: {result.get('email_id')}",
        f"  from: {result.get('sender')}",
        f"  subject: {result.get('subject')}",
        f"  summary: {result.get('summary')}",
        f"  class/action: {result.get('classification')} / {result.get('action')}",
        f"  memories saved: {len(stored)}",
    ]
    if draft_reply:
        lines.extend(["  draft reply:", _indent(str(draft_reply).strip(), "    ")])
    return "\n".join(lines)


def _indent(value: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in value.splitlines())

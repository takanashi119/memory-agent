"""Email processing services shared by inbox listeners and CLIs."""

from __future__ import annotations

import asyncio
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


class EmailReplyConfirmer(Protocol):
    """Asks the user whether a generated reply should be sent."""

    async def confirm(self, email: dict[str, Any], draft_reply: str) -> bool:
        """Return whether the user approved sending the draft reply."""


class EmailReplySender(Protocol):
    """Sends an approved reply through an external email provider."""

    async def send_reply(self, email: dict[str, Any], draft_reply: str) -> str:
        """Send a reply and return the provider's sent message ID."""


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


class ConsoleEmailReplyConfirmer:
    """Ask for reply approval in an interactive terminal."""

    async def confirm(self, email: dict[str, Any], draft_reply: str) -> bool:
        """Return whether the console user approved sending the draft reply."""
        if not sys.stdin.isatty():
            logger.info("Skipping reply approval because stdin is not interactive.")
            return False
        return await asyncio.to_thread(self._confirm_sync, email, draft_reply)

    def _confirm_sync(self, email: dict[str, Any], draft_reply: str) -> bool:
        sys.stdout.write("\nDraft reply ready for approval\n")
        sys.stdout.write(f"  to: {email.get('sender')}\n")
        sys.stdout.write(f"  subject: {email.get('subject')}\n")
        sys.stdout.write("  draft reply:\n")
        sys.stdout.write(_indent(draft_reply.strip(), "    ") + "\n")
        sys.stdout.write("Send this reply? [y/N]: ")
        sys.stdout.flush()
        answer = sys.stdin.readline().strip().lower()
        return answer in {"y", "yes"}


@dataclass(kw_only=True)
class EmailProcessingService:
    """Run email payloads through the email graph and publish results."""

    processed_store: ProcessedMessageStore
    notifier: EmailNotifier
    reply_confirmer: EmailReplyConfirmer | None = None
    reply_sender: EmailReplySender | None = None
    user_id: str = "default"
    model: str | None = None
    memory_path: str | Path | None = None

    def __post_init__(self) -> None:
        """Build the graph runtime dependencies."""
        self._store = InMemoryStore()
        self._memory_archive = JsonUserMemoryArchive(
            self.memory_path or user_memory_path(self.user_id)
        )
        self._memory_archive.load_into(self._store, self.user_id)
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
        await self._handle_reply_confirmation(result)
        self.processed_store.add(str(email["email_id"]))
        self._memory_archive.save_from(self._store, self.user_id)
        await self.notifier.notify(result)
        return result

    async def _handle_reply_confirmation(self, result: dict[str, Any]) -> None:
        draft_reply = result.get("draft_reply")
        if not result.get("reply_confirmation_required") or not draft_reply:
            return

        if self.reply_confirmer is None:
            result["reply_status"] = "pending_user_confirmation"
            return

        approved = await self.reply_confirmer.confirm(result, str(draft_reply))
        result["reply_send_approved"] = approved
        if not approved:
            result["reply_status"] = "declined_by_user"
            return

        if self.reply_sender is None:
            result["reply_status"] = "approved_no_sender_configured"
            return

        try:
            sent_message_id = await self.reply_sender.send_reply(result, str(draft_reply))
        except Exception as exc:
            logger.exception("Failed to send approved email reply.")
            result["reply_status"] = "send_failed"
            result["reply_send_error"] = str(exc)
            return

        result["reply_status"] = "sent"
        result["sent_reply_id"] = sent_message_id


class JsonUserMemoryArchive:
    """Persist one user's in-memory LangGraph memories to a local JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_into(self, store: InMemoryStore, user_id: str) -> None:
        """Load archived memories into the graph's in-memory store."""
        if not self.path.exists():
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid user memory file: %s", self.path)
            return

        memories = _memory_items_from_payload(payload)
        namespace = ("memories", user_id)
        for memory in memories:
            key = str(memory.get("key", "")).strip()
            value = memory.get("value")
            if key and isinstance(value, dict):
                store.put(namespace, key, value)

    def save_from(self, store: InMemoryStore, user_id: str) -> None:
        """Write the graph's current in-memory memories to disk."""
        namespace = ("memories", user_id)
        memories = [
            {
                "key": item.key,
                "value": item.value,
            }
            for item in _all_memories(store, namespace)
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().isoformat(),
            "user_id": user_id,
            "memories": memories,
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def user_memory_path(user_id: str, root: str | Path = "memory") -> Path:
    """Return the local archive path for one user's memories."""
    safe_user_id = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_" for char in user_id
    )
    return Path(root) / f"{safe_user_id}_memories"


def _memory_items_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("memories"), list):
        return [item for item in payload["memories"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _all_memories(store: InMemoryStore, namespace: tuple[str, str]) -> list[Any]:
    memories = []
    offset = 0
    limit = 100
    while True:
        batch = store.search(namespace, limit=limit, offset=offset)
        memories.extend(batch)
        if len(batch) < limit:
            return memories
        offset += limit



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
        # f"  class/action: {result.get('classification')} / {result.get('action')}",
        f"  class: {result.get('classification')}",
        f"  memories saved: {len(stored)}",
        f"  "
    ]
    if result.get("reply_status"):
        lines.append(f"  reply status: {result.get('reply_status')}")
    if result.get("sent_reply_id"):
        lines.append(f"  sent reply id: {result.get('sent_reply_id')}")
    if result.get("reply_send_error"):
        lines.append(f"  reply send error: {result.get('reply_send_error')}")
    if draft_reply:
        lines.extend(["  draft reply:", _indent(str(draft_reply).strip(), "    ")])
    return "\n".join(lines)


def _indent(value: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in value.splitlines())

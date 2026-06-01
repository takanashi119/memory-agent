"""Decoupled long-term memory backend adapters."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langgraph.store.base import BaseStore


MEMORY_NAMESPACE = "memories"


@dataclass(frozen=True, kw_only=True)
class MemoryRecord:
    """Backend-neutral memory search result."""

    key: str
    value: dict[str, Any]
    score: float | None = None


@runtime_checkable
class MemoryBackend(Protocol):
    """Interface implemented by pluggable long-term memory systems."""

    async def upsert(
        self,
        *,
        user_id: str,
        content: str,
        context: str,
        memory_id: str | None = None,
    ) -> str:
        """Store or update one memory and return its stable ID."""

    async def search(
        self,
        *,
        user_id: str,
        query: str | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        """Return memories relevant to a query."""

    async def list(self, *, user_id: str, limit: int = 100) -> list[MemoryRecord]:
        """Return memories without requiring a semantic query."""


class LangGraphMemoryBackend:
    """Adapter for LangGraph's BaseStore memory namespace."""

    def __init__(self, store: BaseStore) -> None:
        self.store = store

    async def upsert(
        self,
        *,
        user_id: str,
        content: str,
        context: str,
        memory_id: str | None = None,
    ) -> str:
        mem_id = memory_id or str(uuid.uuid4())
        await self.store.aput(
            _namespace(user_id),
            key=mem_id,
            value={"content": content, "context": context},
        )
        return mem_id

    async def search(
        self,
        *,
        user_id: str,
        query: str | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        items = await self.store.asearch(_namespace(user_id), query=query, limit=limit)
        return [_record_from_langgraph_item(item) for item in items]

    async def list(self, *, user_id: str, limit: int = 100) -> list[MemoryRecord]:
        return await self.search(user_id=user_id, limit=limit)


class Mem0MemoryBackend:
    """Adapter for mem0's Memory SDK.

    The SDK is imported lazily so the default LangGraph backend can run without
    mem0 installed in the active Python environment.
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Mem0MemoryBackend:
        """Create a mem0 backend from a mem0 config dictionary."""
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError(
                "mem0 is not importable. Install the mem0ai package before using "
                "Mem0MemoryBackend."
            ) from exc
        return cls(Memory.from_config(config))

    async def upsert(
        self,
        *,
        user_id: str,
        content: str,
        context: str,
        memory_id: str | None = None,
    ) -> str:
        metadata = {"context": context}
        if memory_id:
            metadata["memory_id"] = memory_id

        result = await asyncio.to_thread(
            self.memory.add,
            [{"role": "user", "content": f"{content}\n\nContext: {context}"}],
            user_id=user_id,
            metadata=metadata,
        )
        return _extract_mem0_id(result) or memory_id or str(uuid.uuid4())

    async def search(
        self,
        *,
        user_id: str,
        query: str | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        if query:
            raw = await asyncio.to_thread(
                self.memory.search,
                query=query,
                user_id=user_id,
                limit=limit,
            )
        else:
            raw = await asyncio.to_thread(
                self.memory.get_all,
                filters={"user_id": user_id},
            )
        return _records_from_mem0(raw)[:limit]

    async def list(self, *, user_id: str, limit: int = 100) -> list[MemoryRecord]:
        return await self.search(user_id=user_id, limit=limit)


def memory_backend_from_runtime(context: Any, store: BaseStore | None) -> MemoryBackend | None:
    """Resolve the configured memory backend for a graph runtime."""
    backend = getattr(context, "memory_backend", None)
    if backend is not None:
        return backend
    if store is None:
        return None
    return LangGraphMemoryBackend(store)


def format_memory(record: MemoryRecord, *, include_score: bool = False) -> str:
    """Render a backend-neutral memory for prompts."""
    content = record.value.get("content")
    context = record.value.get("context")
    if content and context:
        text = f"[{record.key}]: {content} ({context})"
    elif content:
        text = f"[{record.key}]: {content}"
    else:
        text = f"[{record.key}]: {record.value}"
    if include_score and record.score is not None:
        text += f" (similarity: {record.score})"
    return text


def _namespace(user_id: str) -> tuple[str, str]:
    return (MEMORY_NAMESPACE, user_id)


def _record_from_langgraph_item(item: Any) -> MemoryRecord:
    value = item.value if isinstance(item.value, dict) else {"content": str(item.value)}
    return MemoryRecord(key=str(item.key), value=value, score=getattr(item, "score", None))


def _extract_mem0_id(result: Any) -> str | None:
    if isinstance(result, dict):
        for key in ("id", "memory_id"):
            if result.get(key):
                return str(result[key])
        memories = result.get("results") or result.get("memories")
        if isinstance(memories, list) and memories:
            return _extract_mem0_id(memories[0])
    if isinstance(result, list) and result:
        return _extract_mem0_id(result[0])
    return None


def _records_from_mem0(raw: Any) -> list[MemoryRecord]:
    items = raw.get("results", raw.get("memories", raw)) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []

    records = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or item.get("memory_id") or index)
        content = str(item.get("memory") or item.get("content") or item.get("text") or "")
        context = item.get("metadata", {}).get("context") if isinstance(item.get("metadata"), dict) else None
        value = {"content": content}
        if context:
            value["context"] = str(context)
        score = item.get("score")
        records.append(
            MemoryRecord(
                key=key,
                value=value,
                score=float(score) if isinstance(score, (int, float)) else None,
            )
        )
    return records

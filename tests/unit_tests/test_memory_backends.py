import pytest
from langgraph.store.memory import InMemoryStore

from memory_agent.memory_backends import LangGraphMemoryBackend, format_memory
from memory_agent.tools import upsert_memory

pytestmark = pytest.mark.anyio


async def test_langgraph_memory_backend_upserts_and_searches():
    store = InMemoryStore()
    backend = LangGraphMemoryBackend(store)

    memory_id = await backend.upsert(
        user_id="alice",
        content="Alice prefers concise email replies.",
        context="Mentioned in email.",
    )

    memories = await backend.search(user_id="alice", query="email replies", limit=5)

    assert memory_id
    assert memories[0].key == memory_id
    assert format_memory(memories[0]) == (
        f"[{memory_id}]: Alice prefers concise email replies. (Mentioned in email.)"
    )


async def test_upsert_memory_accepts_injected_memory_backend():
    store = InMemoryStore()
    backend = LangGraphMemoryBackend(store)

    result = await upsert_memory.ainvoke(
        {
            "content": "Alice likes tea.",
            "context": "Mentioned during chat.",
            "user_id": "alice",
            "memory_backend": backend,
        }
    )

    memories = await backend.search(user_id="alice", query="tea", limit=5)

    assert result.startswith("Stored memory ")
    assert memories[0].value == {
        "content": "Alice likes tea.",
        "context": "Mentioned during chat.",
    }

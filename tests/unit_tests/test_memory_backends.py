import pytest
from langgraph.store.memory import InMemoryStore

from memory_agent.memory_backends import LangGraphMemoryBackend, Mem0MemoryBackend, format_memory
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


async def test_mem0_search_uses_user_id_filter():
    class FakeMem0:
        def __init__(self):
            self.search_kwargs = None

        def search(self, **kwargs):
            self.search_kwargs = kwargs
            if "user_id" in kwargs:
                raise ValueError("user_id must be provided through filters")
            return {
                "results": [
                    {
                        "id": "mem-1",
                        "memory": "Alice prefers concise email replies.",
                        "metadata": {"context": "Mentioned in email."},
                    }
                ]
            }

    memory = FakeMem0()
    backend = Mem0MemoryBackend(memory)

    memories = await backend.search(user_id="alice", query="email replies", limit=5)

    assert memory.search_kwargs == {
        "query": "email replies",
        "filters": {"user_id": "alice"},
        "limit": 5,
    }
    assert memories[0].key == "mem-1"


async def test_mem0_ingest_email_adds_raw_email_conversation():
    class FakeMem0:
        def __init__(self):
            self.add_args = None
            self.add_kwargs = None

        def add(self, *args, **kwargs):
            self.add_args = args
            self.add_kwargs = kwargs
            return {"id": "email-memory-1"}

    memory = FakeMem0()
    backend = Mem0MemoryBackend(memory)

    memory_id = await backend.ingest_email(
        user_id="alice",
        email={
            "email_id": "email-1",
            "thread_id": "thread-1",
            "sender": "vendor@example.com",
            "subject": "Quote",
            "received_at": "2026-06-05T12:00:00",
            "body": "Raw procurement email body with details mem0 should inspect.",
        },
        result={
            "summary": "Vendor sent a quote.",
            "classification": "task",
            "key_info": {"vendor": "Example Vendor"},
            "action": "ask_user",
            "draft_reply": None,
        },
    )

    messages = memory.add_args[0]

    assert memory_id == "email-memory-1"
    assert memory.add_kwargs["user_id"] == "alice"
    assert memory.add_kwargs["metadata"]["email_id"] == "email-1"
    assert messages[0]["role"] == "user"
    assert "Raw procurement email body" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert "Vendor sent a quote." in messages[1]["content"]
    assert '"vendor": "Example Vendor"' in messages[1]["content"]

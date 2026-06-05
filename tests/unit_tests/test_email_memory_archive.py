from types import SimpleNamespace

import pytest
from langgraph.store.memory import InMemoryStore

from memory_agent.email_graph import EmailState, retrieve_email_preferences


@pytest.mark.anyio
async def test_retrieve_email_preferences_reads_user_memories():
    store = InMemoryStore()
    store.put(
        ("memories", "alice"),
        "pref-1",
        {
            "content": "Alice prefers concise email replies.",
            "context": "type=preference; scope=user; confidence=0.95",
        },
    )
    state = EmailState(
        email_id="email-1",
        sender="bob@example.com",
        subject="Project update",
        body="Can you reply with your thoughts?",
    )
    runtime = SimpleNamespace(
        store=store,
        context=SimpleNamespace(user_id="alice"),
    )

    result = await retrieve_email_preferences(state, runtime)

    assert result == {
        "email_preferences": [
            "[pref-1]: Alice prefers concise email replies. (type=preference; scope=user; confidence=0.95)"
        ]
    }

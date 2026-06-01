from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.store.memory import InMemoryStore

from memory_agent.email_graph import EmailState, _filter_long_term_memories, retrieve_email_preferences
from memory_agent.email_service import JsonUserMemoryArchive, user_memory_path


def test_user_memory_path_uses_requested_layout():
    assert user_memory_path("default") == Path("memory/default_memories")


def test_json_user_memory_archive_round_trips_memories(tmp_path):
    path = tmp_path / "memory" / "alice_memories"
    store = InMemoryStore()
    store.put(
        ("memories", "alice"),
        "memory-1",
        {"content": "Alice likes tea", "context": "Mentioned in email."},
    )

    JsonUserMemoryArchive(path).save_from(store, "alice")

    reloaded = InMemoryStore()
    JsonUserMemoryArchive(path).load_into(reloaded, "alice")
    memory = reloaded.get(("memories", "alice"), "memory-1")

    assert memory is not None
    assert memory.value == {
        "content": "Alice likes tea",
        "context": "Mentioned in email.",
    }


def test_filter_long_term_memories_keeps_stable_high_confidence_facts():
    memories = _filter_long_term_memories(
        [
            {
                "content": "Alice prefers concise email replies.",
                "context": "Alice explicitly requested concise replies.",
                "type": "preference",
                "confidence": 0.92,
                "scope": "user",
            }
        ],
        "personal_info",
    )

    assert memories == [
        {
            "content": "Alice prefers concise email replies.",
            "context": "Alice explicitly requested concise replies. (type=preference; scope=user; confidence=0.92)",
        }
    ]


def test_filter_long_term_memories_rejects_low_confidence_facts():
    memories = _filter_long_term_memories(
        [
            {
                "content": "Alice may prefer morning meetings.",
                "context": "Weakly inferred from one scheduling email.",
                "type": "preference",
                "confidence": 0.5,
                "scope": "user",
            }
        ],
        "personal_info",
    )

    assert memories == []


def test_filter_long_term_memories_rejects_transient_meeting_facts():
    memories = _filter_long_term_memories(
        [
            {
                "content": "Alice has a meeting next week about the budget.",
                "context": "Mentioned in one email.",
                "type": "project",
                "confidence": 0.95,
                "scope": "project",
            }
        ],
        "meeting",
    )

    assert memories == []


def test_filter_long_term_memories_rejects_newsletters():
    memories = _filter_long_term_memories(
        [
            {
                "content": "Alice is interested in the product launch.",
                "context": "Inferred from a newsletter.",
                "type": "preference",
                "confidence": 0.95,
                "scope": "user",
            }
        ],
        "newsletter",
    )

    assert memories == []


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

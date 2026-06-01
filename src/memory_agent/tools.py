"""Define he agent's tools."""

import uuid
from typing import Annotated

from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

from memory_agent.memory_backends import (
    LangGraphMemoryBackend,
    MemoryBackend,
)


@tool
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: uuid.UUID | None = None,
    # Hide these arguments from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore | None, InjectedToolArg] = None,
    memory_backend: Annotated[MemoryBackend | None, InjectedToolArg] = None,
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    backend = memory_backend
    if backend is None and store is not None:
        backend = LangGraphMemoryBackend(store)
    if backend is None:
        raise ValueError("upsert_memory requires a memory backend or LangGraph store.")

    mem_id = await backend.upsert(
        user_id=user_id,
        content=content,
        context=context,
        memory_id=str(memory_id) if memory_id else None,
    )
    return f"Stored memory {mem_id}"

@tool
async def write_email(
    to: str,
    subject: str,
    body: str,
)-> str:
    """Write an email draft."""
    # In a real implementation, this would interface with an email API.
    # For this example, we'll just return the email content.
    return f"Drafted email to {to} with subject '{subject}' and body:\n{body}"

"""Command line entrypoints for memory-agent utilities."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from memory_agent.email_service import (
    ConsoleEmailReplyConfirmer,
    ConsoleEmailNotifier,
    EmailProcessingService,
    JsonProcessedMessageStore,
)
from memory_agent.gmail_client import GmailReplySender
from memory_agent.gmail_listener import GmailListenerSettings, run_gmail_listener
from memory_agent.memory_backends import EmailMemoryIngestor, MemoryBackend, Mem0MemoryBackend


DEFAULT_MEM0_CONFIG_PATH = Path("memory/mem0_config.json")


@dataclass(kw_only=True)
class GmailListenerCliSettings:
    """Settings needed to assemble the Gmail listener CLI."""

    listener: GmailListenerSettings
    state_path: Path
    user_id: str
    model: str | None
    memory_backend: str | None
    mem0_config_path: Path


def parse_gmail_listener_args() -> GmailListenerCliSettings:
    """Parse arguments for the Gmail listener command."""
    parser = argparse.ArgumentParser(
        description="Poll Gmail, process new messages with the memory agent, and print results.",
    )
    parser.add_argument("--credentials", default="credentials.json", help="Google OAuth client JSON path.")
    parser.add_argument("--token", default="token.json", help="Google OAuth token JSON path.")
    parser.add_argument(
        "--state",
        default=".gmail_listener_state.json",
        help="Local JSON file used to remember processed Gmail message IDs.",
    )
    parser.add_argument("--query", default="in:inbox", help="Gmail search query to poll.")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Seconds between Gmail checks.")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum messages to inspect per poll.")
    parser.add_argument("--user-id", default="default", help="Memory namespace user ID.")
    parser.add_argument("--gmail-user-id", default="me", help="Gmail API user ID.")
    parser.add_argument("--model", default=None, help="Override Context.model, e.g. openai/gpt-4.1-mini.")
    parser.add_argument("--once", action="store_true", help="Run one polling cycle and exit.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    parser.add_argument(
        "--memory-backend",
        choices=("langgraph", "mem0"),
        default="mem0",
        help="Optional memory backend type. Defaults to LangGraph store.",
    )
    parser.add_argument(
        "--mem0-config",
        default=str(DEFAULT_MEM0_CONFIG_PATH),
        help=f"Path to mem0 config JSON. Defaults to {DEFAULT_MEM0_CONFIG_PATH}.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    return GmailListenerCliSettings(
        listener=GmailListenerSettings(
            credentials_path=Path(args.credentials),
            token_path=Path(args.token),
            query=args.query,
            poll_seconds=args.poll_seconds,
            max_results=args.max_results,
            gmail_user_id=args.gmail_user_id,
            once=args.once,
        ),
        state_path=Path(args.state),
        user_id=args.user_id,
        model=args.model,
        memory_backend=args.memory_backend,
        mem0_config_path=Path(args.mem0_config),
    )


def build_memory_backend(
    backend_name: str | None,
    *,
    mem0_config_path: Path = DEFAULT_MEM0_CONFIG_PATH,
) -> MemoryBackend | None:
    """Build the configured long-term memory backend for the CLI."""
    if backend_name is None or backend_name == "langgraph":
        return None

    if backend_name == "mem0":
        config = _load_mem0_config(mem0_config_path)
        return Mem0MemoryBackend.from_config(config)

    raise ValueError(f"Unsupported memory backend: {backend_name}")


def _load_mem0_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"mem0 config file does not exist: {path}")

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"mem0 config file is not valid JSON: {path}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"mem0 config must be a JSON object: {path}")
    return config


async def run_gmail_listener_cli(settings: GmailListenerCliSettings) -> None:
    """Assemble and run the Gmail listener from CLI settings."""
    memory_backend = build_memory_backend(
        settings.memory_backend,
        mem0_config_path=settings.mem0_config_path,
    )
    memory_ingestor = (
        memory_backend
        if isinstance(memory_backend, EmailMemoryIngestor)
        else None
    )
    processor = EmailProcessingService(
        processed_store=JsonProcessedMessageStore(settings.state_path),
        notifier=ConsoleEmailNotifier(),
        reply_confirmer=ConsoleEmailReplyConfirmer(),
        reply_sender=GmailReplySender(
            credentials_path=settings.listener.credentials_path,
            token_path=settings.listener.token_path,
            user_id=settings.listener.gmail_user_id,
        ),
        user_id=settings.user_id,
        model=settings.model,
        memory_backend=memory_backend,
        memory_ingestor=memory_ingestor,
    )
    await run_gmail_listener(settings.listener, processor)


def gmail_listener_main() -> None:
    """Run the Gmail listener console script."""
    load_dotenv()
    asyncio.run(run_gmail_listener_cli(parse_gmail_listener_args()))


if __name__ == "__main__":
    gmail_listener_main()

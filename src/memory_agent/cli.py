"""Command line entrypoints for memory-agent utilities."""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from memory_agent.email_service import (
    ConsoleEmailReplyConfirmer,
    ConsoleEmailNotifier,
    EmailProcessingService,
    JsonProcessedMessageStore,
)
from memory_agent.gmail_client import GmailReplySender
from memory_agent.gmail_listener import GmailListenerSettings, run_gmail_listener


@dataclass(kw_only=True)
class GmailListenerCliSettings:
    """Settings needed to assemble the Gmail listener CLI."""

    listener: GmailListenerSettings
    state_path: Path
    user_id: str
    model: str | None


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
    )


async def run_gmail_listener_cli(settings: GmailListenerCliSettings) -> None:
    """Assemble and run the Gmail listener from CLI settings."""
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
    )
    await run_gmail_listener(settings.listener, processor)


def gmail_listener_main() -> None:
    """Run the Gmail listener console script."""
    load_dotenv()
    asyncio.run(run_gmail_listener_cli(parse_gmail_listener_args()))


if __name__ == "__main__":
    gmail_listener_main()

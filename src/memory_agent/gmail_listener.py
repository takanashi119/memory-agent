"""Gmail polling loop for discovering new messages."""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass
from pathlib import Path

from memory_agent.email_service import EmailProcessingService
from memory_agent.gmail_client import (
    build_gmail_service,
    fetch_gmail_messages_with_service,
)


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GmailListenerSettings:
    """Runtime settings for polling Gmail."""

    credentials_path: Path
    token_path: Path
    query: str
    poll_seconds: int
    max_results: int
    gmail_user_id: str
    once: bool


async def run_gmail_listener(
    settings: GmailListenerSettings,
    processor: EmailProcessingService,
) -> None:
    """Poll Gmail and pass new messages to an email processor."""
    service = build_gmail_service(
        credentials_path=settings.credentials_path,
        token_path=settings.token_path,
    )
    stop_event = asyncio.Event()
    _install_stop_handlers(stop_event)

    logger.info(
        "Gmail listener started. query=%r interval=%ss",
        settings.query,
        settings.poll_seconds,
    )

    while not stop_event.is_set():
        try:
            emails = fetch_gmail_messages_with_service(
                service,
                query=settings.query,
                max_results=settings.max_results,
                user_id=settings.gmail_user_id,
            )
            new_emails = [
                email for email in reversed(emails) if not processor.is_processed(email)
            ]

            if not new_emails:
                logger.info("No new matching emails.")

            for email in new_emails:
                ### Key point: process each email sequentially to avoid parallel API calls and rate limits.
                await processor.process(email)
        except Exception:
            logger.exception("Gmail listener cycle failed.")

        if settings.once:
            break

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.poll_seconds)
        except TimeoutError:
            pass

    logger.info("Gmail listener stopped.")


def _install_stop_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass

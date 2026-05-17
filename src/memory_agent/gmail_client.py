"""Utilities for reading real Gmail messages into the email graph."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def build_gmail_service(
    *,
    credentials_path: str | Path = "credentials.json",
    token_path: str | Path = "token.json",
    scopes: list[str] | None = None,
) -> Any:
    """Create an authenticated Gmail API service.

    Download ``credentials.json`` from Google Cloud Console after enabling the
    Gmail API for an OAuth desktop app. The first run opens a browser consent
    flow and writes ``token.json`` for later runs.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Gmail dependencies are not installed. Run `uv sync` after the "
            "new google-api dependencies have been added."
        ) from exc

    credentials_file = Path(credentials_path)
    token_file = Path(token_path)
    auth_scopes = scopes or [GMAIL_READONLY_SCOPE]

    credentials = None
    if token_file.exists():
        credentials = Credentials.from_authorized_user_file(str(token_file), auth_scopes)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not credentials_file.exists():
                raise FileNotFoundError(
                    f"Missing Gmail OAuth credentials file: {credentials_file}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_file),
                auth_scopes,
            )
            credentials = flow.run_local_server(port=0)

        token_file.write_text(credentials.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=credentials)


def fetch_gmail_messages(
    *,
    query: str = "in:inbox newer_than:7d",
    max_results: int = 10,
    user_id: str = "me",
    credentials_path: str | Path = "credentials.json",
    token_path: str | Path = "token.json",
) -> list[dict[str, Any]]:
    """Fetch Gmail messages and convert them to ``EmailState`` input dicts."""
    service = build_gmail_service(
        credentials_path=credentials_path,
        token_path=token_path,
    )
    return fetch_gmail_messages_with_service(
        service,
        query=query,
        max_results=max_results,
        user_id=user_id,
    )


def fetch_gmail_messages_with_service(
    service: Any,
    *,
    query: str = "in:inbox newer_than:7d",
    max_results: int = 10,
    user_id: str = "me",
) -> list[dict[str, Any]]:
    """Fetch Gmail messages with an existing Gmail API service."""
    messages = _list_messages(service, user_id=user_id, query=query, max_results=max_results)
    return [
        gmail_message_to_email_input(
            _get_message(service, user_id=user_id, message_id=message["id"])
        )
        for message in messages
    ]


async def process_gmail_messages(
    email_graph: Any,
    *,
    context: Any,
    config: dict[str, Any] | None = None,
    query: str = "in:inbox newer_than:7d",
    max_results: int = 10,
    user_id: str = "me",
    credentials_path: str | Path = "credentials.json",
    token_path: str | Path = "token.json",
) -> list[dict[str, Any]]:
    """Fetch Gmail messages and run each one through the compiled email graph."""
    results = []
    for email in fetch_gmail_messages(
        query=query,
        max_results=max_results,
        user_id=user_id,
        credentials_path=credentials_path,
        token_path=token_path,
    ):
        result = await email_graph.ainvoke(email, config=config, context=context)
        results.append(result)
    return results


def gmail_message_to_email_input(message: dict[str, Any]) -> dict[str, Any]:
    """Convert a Gmail API Message resource into ``EmailState`` input."""
    headers = _headers_by_name(message.get("payload", {}).get("headers", []))
    body = _message_body(message.get("payload", {})) or message.get("snippet", "")
    return {
        "email_id": message["id"],
        "thread_id": message.get("threadId"),
        "sender": headers.get("from", ""),
        "subject": headers.get("subject", ""),
        "body": body,
        "received_at": headers.get("date"),
    }


def _list_messages(
    service: Any,
    *,
    user_id: str,
    query: str,
    max_results: int,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    request = (
        service.users()
        .messages()
        .list(userId=user_id, q=query, maxResults=min(max_results, 500))
    )

    while request is not None and len(messages) < max_results:
        response = request.execute()
        messages.extend(response.get("messages", []))
        if len(messages) >= max_results:
            break
        request = service.users().messages().list_next(request, response)

    return messages[:max_results]


def _get_message(service: Any, *, user_id: str, message_id: str) -> dict[str, Any]:
    return (
        service.users()
        .messages()
        .get(
            userId=user_id,
            id=message_id,
            format="full",
        )
        .execute()
    )


def _headers_by_name(headers: list[dict[str, str]]) -> dict[str, str]:
    return {
        header.get("name", "").lower(): header.get("value", "")
        for header in headers
    }


def _message_body(payload: dict[str, Any]) -> str:
    plain = _find_part_data(payload, "text/plain")
    if plain:
        return plain
    return _find_part_data(payload, "text/html")


def _find_part_data(part: dict[str, Any], mime_type: str) -> str:
    if part.get("mimeType") == mime_type:
        data = part.get("body", {}).get("data")
        if data:
            return _decode_base64url(data)

    for child in part.get("parts", []):
        body = _find_part_data(child, mime_type)
        if body:
            return body

    return ""


def _decode_base64url(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding).decode(
        "utf-8",
        errors="replace",
    )

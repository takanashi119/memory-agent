"""Utilities for reading real Gmail messages into the email graph."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from email.message import EmailMessage
from pathlib import Path
from typing import Any


GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"
GMAIL_READ_SEND_SCOPES = [GMAIL_READONLY_SCOPE, GMAIL_SEND_SCOPE]


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
    if token_file.exists() and _token_file_has_scopes(token_file, auth_scopes):
        credentials = Credentials.from_authorized_user_file(str(token_file), auth_scopes)

    if credentials and not _credentials_have_scopes(credentials, auth_scopes):
        credentials = None

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


def _credentials_have_scopes(credentials: Any, scopes: list[str]) -> bool:
    has_scopes = getattr(credentials, "has_scopes", None)
    if has_scopes is None:
        return True
    return bool(has_scopes(scopes))


def _token_file_has_scopes(token_file: Path, scopes: list[str]) -> bool:
    try:
        payload = json.loads(token_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    granted_scopes = payload.get("scopes")
    if not isinstance(granted_scopes, list):
        return False
    return set(scopes).issubset({str(scope) for scope in granted_scopes})


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


class GmailReplySender:
    """Send approved replies through Gmail."""

    def __init__(
        self,
        *,
        credentials_path: str | Path = "credentials.json",
        token_path: str | Path = "token.json",
        user_id: str = "me",
    ) -> None:
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.user_id = user_id
        self._service: Any | None = None

    async def send_reply(self, email: dict[str, Any], draft_reply: str) -> str:
        """Send a Gmail reply and return the sent message ID."""
        return await asyncio.to_thread(self._send_reply_sync, email, draft_reply)

    def _send_reply_sync(self, email: dict[str, Any], draft_reply: str) -> str:
        if self._service is None:
            self._service = build_gmail_service(
                credentials_path=self.credentials_path,
                token_path=self.token_path,
                scopes=GMAIL_READ_SEND_SCOPES,
            )
        return send_gmail_reply_with_service(
            self._service,
            email=email,
            draft_reply=draft_reply,
            user_id=self.user_id,
        )


def send_gmail_reply_with_service(
    service: Any,
    *,
    email: dict[str, Any],
    draft_reply: str,
    user_id: str = "me",
) -> str:
    """Send a Gmail reply for a processed email and return the sent message ID."""
    message = _build_reply_message(
        from_address=_gmail_profile_email(service, user_id),
        to_address=str(email.get("sender", "")).strip(),
        subject=_reply_subject(str(email.get("subject", ""))),
        body=draft_reply,
    )
    payload: dict[str, Any] = {
        "raw": base64.urlsafe_b64encode(message.as_bytes()).decode("ascii"),
    }
    thread_id = email.get("thread_id")
    if thread_id:
        payload["threadId"] = str(thread_id)

    sent = service.users().messages().send(userId=user_id, body=payload).execute()
    return str(sent["id"])


def _gmail_profile_email(service: Any, user_id: str) -> str:
    profile = service.users().getProfile(userId=user_id).execute()
    return str(profile["emailAddress"])


def _build_reply_message(
    *,
    from_address: str,
    to_address: str,
    subject: str,
    body: str,
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = from_address
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(body)
    return message


def _reply_subject(subject: str) -> str:
    subject = subject.strip() or "(no subject)"
    if re.match(r"(?i)^\s*re\s*:", subject):
        return subject
    return f"Re: {subject}"


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

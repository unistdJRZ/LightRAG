"""HTTP client helpers for OpenCode agent search integration."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from lightrag.utils import logger

from .config import global_args

StatusCallback = Callable[["OpencodeStatusEvent"], Awaitable[None] | None]


@dataclass(slots=True)
class OpencodeStatusEvent:
    agent_search_id: str
    phase: str
    message: str
    event_type: str | None = None
    session_id: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class OpencodeAgentRunResult:
    agent_search_id: str
    workspace: str
    retrieval_target: str
    session_id: str | None = None
    final_output: str | None = None
    error: str | None = None
    status_events: list[OpencodeStatusEvent] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class _EventStreamState:
    idle: bool = False
    error: str | None = None


def is_opencode_enabled() -> bool:
    return bool(
        getattr(global_args, "opencode_server_url", None)
        and getattr(global_args, "opencode_rag_agent", None)
    )


def _normalize_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().rstrip("/")
    return normalized or None


def _build_message(
    agent_search_id: str,
    workspace: str,
    retrieval_target: str,
    prior_rag_context: str | None = None,
) -> str:
    lines = [
        "You are the LightRAG RAG search agent.",
        f"workspace: {workspace}",
        f"agent_search_id: {agent_search_id}",
        f"agent_submit_id: {agent_search_id}",
        "",
        "Goal: infer the user's real retrieval intent and submit precise entity_ids and chunk_ids for the final RAG query.",
        "",
        "Input format:",
        "- retrieval_target may contain [history] and [latest_query]. Messages can include a reference list after the message content.",
        "- [latest_query] is the highest-priority user message. Read its text and any following reference list before searching.",
        "- prior_rag_context is an initial LightRAG pass. It may contain [entities], [relationships], [chunks], and [doc_qa]. Treat it as retrieval hints, not as the final answer.",
        "- [doc_qa] can include preset questions and reference information derived from documents. Use it to recover context that is implicit in the latest message.",
        "",
        "Required workflow:",
        "1. Read [latest_query] first, including its reference list if present. Extract all chunk_id values from the latest message references.",
        "2. Read [history] to infer omitted subjects, pronouns, follow-up intent, comparison targets, and whether older references still matter.",
        "3. Prefer chunk_id values from the latest message references. Then inspect [doc_qa] rows whose belong_chunk matches those chunk_id values to understand the referenced document context.",
        "4. If the latest message has no usable chunk_id, use history references, [chunks], [entities], and [relationships] as fallback clues.",
        "5. Convert the inferred intent into retrieval rules: core topic, required chunks/entities, expansion keywords, constraints, and older references to ignore if they conflict with the latest query.",
        "6. Use the LightRAG agent APIs to verify, expand, and refine evidence before submitting. Do not rely only on prior_rag_context.",
        "",
        "When you finish, call /api/agent/submit with the exact workspace and agent_submit_id above.",
    ]
    if isinstance(prior_rag_context, str) and prior_rag_context.strip():
        lines.extend(
            [
                "",
                "prior_rag_context:",
                prior_rag_context.strip(),
            ]
        )
    lines.extend(
        [
            "",
            "retrieval_target:",
            retrieval_target,
        ]
    )
    return "\n".join(lines)


def _extract_text_parts(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    parts = payload.get("parts")
    if not isinstance(parts, list):
        return ""

    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key in ("text", "content", "delta"):
            value = part.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value)
                break
    return "".join(chunks).strip()


def _extract_message_text_from_messages(payload: Any) -> str:
    if not isinstance(payload, list):
        return ""

    assistant_messages: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        info = item.get("info")
        if isinstance(info, dict) and info.get("role") == "assistant":
            assistant_messages.append(item)

    if not assistant_messages:
        return ""

    for item in reversed(assistant_messages):
        parts = item.get("parts")
        if not isinstance(parts, list):
            continue
        texts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
        if texts:
            return "".join(texts).strip()
    return ""


def _extract_session_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("sessionID", "session_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in payload.values():
            nested = _extract_session_id(value)
            if nested:
                return nested
    elif isinstance(payload, list):
        for value in payload:
            nested = _extract_session_id(value)
            if nested:
                return nested
    return None


def _extract_status_message(event_type: str | None, payload: Any) -> str | None:
    if isinstance(payload, dict):
        nested = payload.get("properties")
        if isinstance(nested, dict):
            nested_message = _extract_status_message(event_type, nested)
            if nested_message:
                return nested_message

        for key in ("message", "text", "content", "delta", "title"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        parts_text = _extract_text_parts(payload)
        if parts_text:
            return parts_text

        error_value = payload.get("error")
        if isinstance(error_value, str) and error_value.strip():
            return error_value.strip()

    if event_type:
        return event_type
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return None


async def _emit_status(
    callback: StatusCallback | None,
    events: list[OpencodeStatusEvent],
    status: OpencodeStatusEvent,
) -> None:
    events.append(status)
    if callback is None:
        return
    result = callback(status)
    if asyncio.iscoroutine(result):
        await result


async def _validate_agent_exists(client: httpx.AsyncClient, agent_name: str) -> None:
    response = await client.get("/agent")
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list):
        return

    available: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        for key in ("id", "name", "key"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                available.add(value.strip())

    if available and agent_name not in available:
        raise RuntimeError(
            f"Configured OpenCode agent '{agent_name}' was not found. Available agents: {', '.join(sorted(available))}"
        )


async def _stream_events(
    client: httpx.AsyncClient,
    session_id: str,
    agent_search_id: str,
    callback: StatusCallback | None,
    events: list[OpencodeStatusEvent],
    state: _EventStreamState,
) -> None:
    try:
        async with client.stream("GET", "/event") as response:
            response.raise_for_status()
            event_type: str | None = None
            data_lines: list[str] = []

            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event_type = line[6:].strip() or None
                    continue

                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue

                if line != "":
                    continue

                raw_data = "\n".join(data_lines).strip()
                data_lines = []

                if not raw_data:
                    event_type = None
                    continue

                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError:
                    payload = {"raw": raw_data}

                if (event_type or "").strip() == "server.connected":
                    event_type = None
                    continue

                payload_session_id = _extract_session_id(payload)
                if payload_session_id and payload_session_id != session_id:
                    event_type = None
                    continue

                normalized_event_type = event_type
                if (
                    normalized_event_type is None
                    and isinstance(payload, dict)
                    and isinstance(payload.get("type"), str)
                ):
                    normalized_event_type = payload["type"]

                if normalized_event_type == "session.idle":
                    state.idle = True

                if normalized_event_type == "session.error":
                    state.error = _extract_status_message(normalized_event_type, payload) or "OpenCode session error"

                message = _extract_status_message(normalized_event_type, payload)
                if message:
                    await _emit_status(
                        callback,
                        events,
                        OpencodeStatusEvent(
                            agent_search_id=agent_search_id,
                            phase="event",
                            message=message,
                            event_type=normalized_event_type,
                            session_id=session_id,
                            raw=payload if isinstance(payload, dict) else None,
                        ),
                    )

                event_type = None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("OpenCode event stream closed early for session %s: %s", session_id, exc)


async def _wait_until_session_complete(
    client: httpx.AsyncClient,
    session_id: str,
    state: _EventStreamState,
    timeout_seconds: float,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    while True:
        if state.error:
            return state.error
        if state.idle:
            return None
        if asyncio.get_running_loop().time() >= deadline:
            return "OpenCode session did not reach idle state before timeout"

        response = await client.get("/session/status")
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            session_status = payload.get(session_id)
            if isinstance(session_status, dict):
                status_type = session_status.get("type")
                if status_type == "idle":
                    state.idle = True
                    return None

        await asyncio.sleep(0.5)


async def _delete_session(client: httpx.AsyncClient, session_id: str) -> None:
    try:
        response = await client.delete(f"/session/{session_id}")
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to delete OpenCode session %s: %s", session_id, exc)


async def run_agent_search(
    agent_search_id: str,
    workspace: str,
    retrieval_target: str,
    prior_rag_context: str | None = None,
    callback: StatusCallback | None = None,
) -> OpencodeAgentRunResult:
    base_url = _normalize_base_url(getattr(global_args, "opencode_server_url", None))
    agent_name = str(getattr(global_args, "opencode_rag_agent", "") or "").strip()
    username = str(getattr(global_args, "opencode_server_username", "opencode") or "opencode")
    password = getattr(global_args, "opencode_server_password", None)
    timeout_seconds = float(getattr(global_args, "opencode_timeout", 120.0) or 120.0)

    result = OpencodeAgentRunResult(
        agent_search_id=agent_search_id,
        workspace=workspace,
        retrieval_target=retrieval_target,
    )

    if not base_url:
        result.error = "OpenCode server URL is not configured"
        return result
    if not agent_name:
        result.error = "OpenCode RAG agent name is not configured"
        return result

    auth = httpx.BasicAuth(username, str(password)) if password else None
    timeout = httpx.Timeout(timeout_seconds)
    message = _build_message(
        agent_search_id,
        workspace,
        retrieval_target,
        prior_rag_context=prior_rag_context,
    )

    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            auth=auth,
            timeout=timeout,
            headers={"Accept": "application/json"},
        ) as client:
            health_response = await client.get("/global/health")
            health_response.raise_for_status()
            await _validate_agent_exists(client, agent_name)

            await _emit_status(
                callback,
                result.status_events,
                OpencodeStatusEvent(
                    agent_search_id=agent_search_id,
                    phase="starting",
                    message=f"Connecting to OpenCode agent '{agent_name}'",
                ),
            )

            session_response = await client.post(
                "/session",
                json={"title": f"LightRAG agent search {agent_search_id}"},
            )
            session_response.raise_for_status()
            session_payload = session_response.json()
            session_id = (
                session_payload.get("id")
                if isinstance(session_payload, dict)
                else None
            )
            if not isinstance(session_id, str) or not session_id.strip():
                raise RuntimeError("OpenCode session creation did not return a valid session id")

            result.session_id = session_id
            try:
                await _emit_status(
                    callback,
                    result.status_events,
                    OpencodeStatusEvent(
                        agent_search_id=agent_search_id,
                        phase="session_created",
                        message=f"Created OpenCode session {session_id}",
                        session_id=session_id,
                    ),
                )

                stream_state = _EventStreamState()
                event_task = asyncio.create_task(
                    _stream_events(
                        client=client,
                        session_id=session_id,
                        agent_search_id=agent_search_id,
                        callback=callback,
                        events=result.status_events,
                        state=stream_state,
                    )
                )

                try:
                    prompt_response = await client.post(
                        f"/session/{session_id}/prompt_async",
                        json={
                            "agent": agent_name,
                            "parts": [{"type": "text", "text": message}],
                        },
                    )
                    prompt_response.raise_for_status()

                    error = await _wait_until_session_complete(
                        client=client,
                        session_id=session_id,
                        state=stream_state,
                        timeout_seconds=timeout_seconds,
                    )
                    if error:
                        raise RuntimeError(error)

                    messages_response = await client.get(f"/session/{session_id}/message")
                    messages_response.raise_for_status()
                    result.final_output = _extract_message_text_from_messages(
                        messages_response.json()
                    )
                finally:
                    event_task.cancel()
                    try:
                        await event_task
                    except asyncio.CancelledError:
                        pass

                await _emit_status(
                    callback,
                    result.status_events,
                    OpencodeStatusEvent(
                        agent_search_id=agent_search_id,
                        phase="completed",
                        message=result.final_output or "OpenCode agent search completed",
                        session_id=session_id,
                    ),
                )
                return result
            finally:
                await _delete_session(client, session_id)
    except Exception as exc:
        logger.error("OpenCode agent search failed: %s", exc)
        result.error = str(exc)
        await _emit_status(
            callback,
            result.status_events,
            OpencodeStatusEvent(
                agent_search_id=agent_search_id,
                phase="failed",
                message=str(exc),
                session_id=result.session_id,
            ),
        )
        return result

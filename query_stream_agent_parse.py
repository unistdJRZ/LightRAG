from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, Mapping


@dataclass(slots=True)
class QueryStreamParseResult:
    """Structured result for /query/stream NDJSON responses."""

    response_chunks: List[str] = field(default_factory=list)
    references: List[dict[str, Any]] | None = None
    errors: List[str] = field(default_factory=list)
    agent_statuses: List[dict[str, Any]] = field(default_factory=list)
    agent_search_result: dict[str, Any] | None = None
    raw_events: List[dict[str, Any]] = field(default_factory=list)

    @property
    def response_text(self) -> str:
        return "".join(self.response_chunks)

    @property
    def has_response(self) -> bool:
        return bool(self.response_chunks)


def _normalize_line(line: str | bytes) -> str:
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if line.startswith("data: "):
        line = line[6:].strip()
    return line


def _iter_json_events(lines: Iterable[str | bytes]) -> Iterator[dict[str, Any]]:
    for raw_line in lines:
        line = _normalize_line(raw_line)
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def parse_query_stream_lines(lines: Iterable[str | bytes]) -> QueryStreamParseResult:
    """
    Parse LightRAG /query/stream responses when stream=True.

    The endpoint returns NDJSON and may emit:
    - {"agent_status": {...}}
    - {"references": [...], "agent_search_result": {...}}
    - {"response": "..."}  # one or more chunks
    - {"response": "...", "references": [...], "agent_search_result": {...}}
    - {"error": "..."}

    Important: when agent_search succeeds and submits context, the first
    business payload may contain only `references` and/or `agent_search_result`
    without a `response` field. Callers must keep reading until the stream ends.
    """

    result = QueryStreamParseResult()

    for event in _iter_json_events(lines):
        result.raw_events.append(event)

        agent_status = event.get("agent_status")
        if isinstance(agent_status, dict):
            result.agent_statuses.append(agent_status)

        references = event.get("references")
        if isinstance(references, list):
            result.references = references

        agent_search_result = event.get("agent_search_result")
        if isinstance(agent_search_result, dict):
            result.agent_search_result = agent_search_result

        response_chunk = event.get("response")
        if isinstance(response_chunk, str):
            result.response_chunks.append(response_chunk)

        error_message = event.get("error")
        if isinstance(error_message, str) and error_message:
            result.errors.append(error_message)

    return result


def parse_query_stream_text(response_text: str) -> QueryStreamParseResult:
    """Parse an already-buffered NDJSON response body."""

    return parse_query_stream_lines(response_text.splitlines())


def parse_query_stream_response(response: Any) -> QueryStreamParseResult:
    """
    Parse a streaming HTTP response object.

    Supports common clients such as `requests.Response` via `iter_lines()`
    and `httpx.Response` via `iter_lines()`.
    """

    iter_lines = getattr(response, "iter_lines", None)
    if callable(iter_lines):
        return parse_query_stream_lines(iter_lines())

    raise TypeError("response must provide an iter_lines() method")


def extract_query_stream_summary(
    parsed: QueryStreamParseResult,
) -> Mapping[str, Any]:
    """Return a compact summary convenient for upstream services."""

    return {
        "response": parsed.response_text,
        "references": parsed.references,
        "agent_search_result": parsed.agent_search_result,
        "agent_statuses": parsed.agent_statuses,
        "errors": parsed.errors,
        "raw_event_count": len(parsed.raw_events),
    }


if __name__ == "__main__":
    sample_lines = [
        '{"agent_status":{"phase":"starting","event_type":"local.started"}}',
        '{"agent_status":{"phase":"rag_started","event_type":"local.rag_started"}}',
        '{"references":[{"chunk_id":"chunk-1"}],"agent_search_result":{"status":"completed","submitted":true}}',
        '{"response":"Light"}',
        '{"response":"RAG"}',
    ]

    parsed_result = parse_query_stream_lines(sample_lines)
    print(json.dumps(extract_query_stream_summary(parsed_result), ensure_ascii=False, indent=2))

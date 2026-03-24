import asyncio
import json
import sys
import types

import pytest

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

ascii_colors_stub = types.ModuleType("ascii_colors")
ascii_colors_stub.ASCIIColors = object
sys.modules.setdefault("ascii_colors", ascii_colors_stub)

from lightrag.api.routers import document_routes as routes  # noqa: E402

sys.argv = _ORIGINAL_ARGV


class _FakeResponse:
    def __init__(self, status: int, body: str):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self) -> str:
        return self._body


class _FakeClientSession:
    def __init__(self, responses, timeout=None):
        self._responses = iter(responses)
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params=None):
        return next(self._responses)


def test_ocr_timeout_defaults():
    assert routes.global_args.ocr_poll_timeout_seconds == 300
    assert routes.global_args.ocr_request_timeout_seconds == 300


def test_fetch_text_from_ocr_server_poll_pending_then_finished(monkeypatch):
    responses = [
        _FakeResponse(200, json.dumps({"status": "PENDING"})),
        _FakeResponse(
            200,
            json.dumps(
                {
                    "status": "FINISHED",
                    "content": {
                        "results": {
                            "demo.pdf": {"md_content": "markdown text", "images": {}}
                        }
                    },
                }
            ),
        ),
    ]
    sleep_calls = []

    async def _fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(routes, "OCR_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(routes.global_args, "ocr_server_url", "http://ocr-server")
    monkeypatch.setattr(routes.global_args, "ocr_poll_timeout_seconds", 300)
    monkeypatch.setattr(routes.global_args, "ocr_request_timeout_seconds", 123)
    captured = {}

    def _fake_client_session(timeout=None):
        captured["timeout"] = timeout
        return _FakeClientSession(responses, timeout=timeout)

    monkeypatch.setattr(
        routes.aiohttp,
        "ClientSession",
        _fake_client_session,
    )
    monkeypatch.setattr(routes.asyncio, "sleep", _fake_sleep)

    text = asyncio.run(routes._fetch_text_from_ocr_server("ocr-123"))

    assert text == "markdown text"
    assert sleep_calls == [0]
    assert captured["timeout"].total == 123


def test_fetch_text_from_ocr_server_fail_status(monkeypatch):
    responses = [
        _FakeResponse(
            200,
            json.dumps({"status": "FAIL", "detail": "OCR engine crashed"}),
        )
    ]

    monkeypatch.setattr(routes.global_args, "ocr_server_url", "http://ocr-server")
    monkeypatch.setattr(routes.global_args, "ocr_poll_timeout_seconds", 300)
    monkeypatch.setattr(routes.global_args, "ocr_request_timeout_seconds", 300)
    monkeypatch.setattr(
        routes.aiohttp,
        "ClientSession",
        lambda timeout=None: _FakeClientSession(responses, timeout=timeout),
    )

    with pytest.raises(RuntimeError, match="OCR task failed"):
        asyncio.run(routes._fetch_text_from_ocr_server("ocr-123"))


def test_extract_structured_segments_from_ocr_chunks():
    ocr_chunks = [
        {
            "chunk_id": 270,
            "page_idx": 57,
            "page_size": [612, 792],
            "bbox": [88, 154, 417, 207],
            "markdown": "Hello\n\nWorld",
            "content_type": "text",
        },
        {
            "chunk_id": 271,
            "page_idx": 58,
            "page_size": [612, 792],
            "bbox": [10, 20, 30, 40],
            "markdown": "QUJDREVGRw==",
            "content_type": "image",
        },
    ]

    segments = routes._extract_structured_segments_from_ocr_chunks(ocr_chunks)

    assert segments == [
        {
            "content": "hello\n\nworld",
            "content_type": "text",
            "page_id": 57,
            "bbox": [88.0, 154.0, 417.0, 207.0],
            "page_size": [612.0, 792.0],
            "ocr_chunk_id": 270,
        },
        {
            "content": "QUJDREVGRw==",
            "content_type": "image",
            "page_id": 58,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "page_size": [612.0, 792.0],
            "ocr_chunk_id": 271,
        },
    ]


def test_fetch_text_from_ocr_server_prefers_chunk_list(monkeypatch):
    responses = [
        _FakeResponse(
            200,
            json.dumps(
                {
                    "status": "FINISHED",
                    "content": {
                        "results": {
                            "demo.pdf": {
                                "md_content": "markdown text",
                                "chunk": [
                                    {
                                        "chunk_id": 10,
                                        "page_idx": 1,
                                        "page_size": [612, 792],
                                        "bbox": [0, 1, 2, 3],
                                        "content_type": "text",
                                        "markdown": "Segment",
                                    },
                                    {
                                        "chunk_id": 11,
                                        "page_idx": 1,
                                        "page_size": [612, 792],
                                        "bbox": [4, 5, 6, 7],
                                        "content_type": "image",
                                        "markdown": "YmFzZTY0LWltYWdl",
                                    },
                                ],
                            }
                        }
                    },
                }
            ),
        )
    ]

    monkeypatch.setattr(routes.global_args, "ocr_server_url", "http://ocr-server")
    monkeypatch.setattr(routes.global_args, "ocr_poll_timeout_seconds", 300)
    monkeypatch.setattr(routes.global_args, "ocr_request_timeout_seconds", 300)
    monkeypatch.setattr(
        routes.aiohttp,
        "ClientSession",
        lambda timeout=None: _FakeClientSession(responses, timeout=timeout),
    )

    content = asyncio.run(routes._fetch_text_from_ocr_server("ocr-456"))

    assert content == {
        "content": "markdown text",
        "content_segments": [
            {
                "content": "segment",
                "content_type": "text",
                "page_id": 1,
                "bbox": [0.0, 1.0, 2.0, 3.0],
                "page_size": [612.0, 792.0],
                "ocr_chunk_id": 10,
            },
            {
                "content": "YmFzZTY0LWltYWdl",
                "content_type": "image",
                "page_id": 1,
                "bbox": [4.0, 5.0, 6.0, 7.0],
                "page_size": [612.0, 792.0],
                "ocr_chunk_id": 11,
            },
        ],
    }


def test_extract_enqueue_input_logs_chunk_parse_context(monkeypatch):
    payload = {
        "status": "FINISHED",
        "content": {
            "results": {
                "demo.pdf": {
                    "md_content": "fallback markdown",
                    "chunk": {"unexpected": "object"},
                }
            }
        },
    }

    warning_messages = []

    def _capture_warning(message, *args, **kwargs):
        if args:
            message = message % args
        warning_messages.append(message)

    monkeypatch.setattr(routes.logger, "warning", _capture_warning)

    content = routes._extract_enqueue_input_from_ocr_payload(payload)

    assert content == "fallback markdown"
    assert len(warning_messages) == 1
    warning_text = warning_messages[0]
    assert "demo.pdf" in warning_text
    assert "chunk list" in warning_text
    assert "field 'chunk' must be a list" in warning_text

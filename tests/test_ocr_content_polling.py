import asyncio
import json
import sys
import types
from pathlib import Path
import uuid

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


class _CapturingClientSession(_FakeClientSession):
    def __init__(self, responses, captured, timeout=None):
        super().__init__(responses, timeout=timeout)
        self._captured = captured

    def get(self, url, params=None):
        self._captured["url"] = url
        self._captured["params"] = params
        return super().get(url, params=params)


def test_ocr_timeout_defaults():
    assert routes.global_args.ocr_poll_timeout_seconds == 1800
    assert routes.global_args.ocr_request_timeout_seconds == 300


def test_get_ocr_poll_delay_seconds_uses_exponential_backoff_with_cap(monkeypatch):
    monkeypatch.setattr(routes, "OCR_POLL_INTERVAL_SECONDS", 2)
    monkeypatch.setattr(routes, "OCR_POLL_MAX_INTERVAL_SECONDS", 30)

    delays = [routes._get_ocr_poll_delay_seconds(attempt) for attempt in range(6)]

    assert delays == [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_resolve_ocr_content_url_switches_to_content_router():
    assert (
        routes._resolve_ocr_content_url("http://ocr-server", ocr_router=True)
        == "http://ocr-server/content_router"
    )
    assert (
        routes._resolve_ocr_content_url("http://ocr-server/content", ocr_router=True)
        == "http://ocr-server/content_router"
    )
    assert (
        routes._resolve_ocr_content_url(
            "http://ocr-server/content_router", ocr_router=False
        )
        == "http://ocr-server/content"
    )


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


def test_fetch_text_from_ocr_server_uses_backoff_sequence(monkeypatch):
    responses = [
        _FakeResponse(200, json.dumps({"status": "PENDING"})),
        _FakeResponse(200, json.dumps({"status": "RUNNING"})),
        _FakeResponse(200, json.dumps({"status": "PENDING"})),
        _FakeResponse(200, json.dumps({"status": "RUNNING"})),
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

    monkeypatch.setattr(routes, "OCR_POLL_INTERVAL_SECONDS", 2)
    monkeypatch.setattr(routes, "OCR_POLL_MAX_INTERVAL_SECONDS", 30)
    monkeypatch.setattr(routes.global_args, "ocr_server_url", "http://ocr-server")
    monkeypatch.setattr(routes.global_args, "ocr_poll_timeout_seconds", 300)
    monkeypatch.setattr(routes.global_args, "ocr_request_timeout_seconds", 123)
    monkeypatch.setattr(
        routes.aiohttp,
        "ClientSession",
        lambda timeout=None: _FakeClientSession(responses, timeout=timeout),
    )
    monkeypatch.setattr(routes.asyncio, "sleep", _fake_sleep)

    text = asyncio.run(routes._fetch_text_from_ocr_server("ocr-123"))

    assert text == "markdown text"
    assert sleep_calls == [2.0, 4.0, 8.0, 16.0, 30.0]


def test_fetch_text_from_ocr_server_poll_running_then_finished(monkeypatch):
    responses = [
        _FakeResponse(200, json.dumps({"status": "RUNNING"})),
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
    monkeypatch.setattr(
        routes.aiohttp,
        "ClientSession",
        lambda timeout=None: _FakeClientSession(responses, timeout=timeout),
    )
    monkeypatch.setattr(routes.asyncio, "sleep", _fake_sleep)

    text = asyncio.run(routes._fetch_text_from_ocr_server("ocr-123"))

    assert text == "markdown text"
    assert sleep_calls == [0]


def test_fetch_text_from_ocr_server_uses_content_router_endpoint(monkeypatch):
    responses = [
        _FakeResponse(
            200,
            json.dumps(
                {
                    "status": "FINISHED",
                    "content": {
                        "results": {
                            "demo.pdf": {
                                "md_content": "router markdown",
                                "images": {},
                            }
                        }
                    },
                }
            ),
        )
    ]
    captured = {}

    monkeypatch.setattr(routes.global_args, "ocr_server_url", "http://ocr-server")
    monkeypatch.setattr(routes.global_args, "ocr_poll_timeout_seconds", 300)
    monkeypatch.setattr(routes.global_args, "ocr_request_timeout_seconds", 123)
    monkeypatch.setattr(
        routes.aiohttp,
        "ClientSession",
        lambda timeout=None: _CapturingClientSession(
            responses, captured, timeout=timeout
        ),
    )

    text = asyncio.run(
        routes._fetch_text_from_ocr_server("ocr-123", ocr_router=True)
    )

    assert text == "router markdown"
    assert captured["url"] == "http://ocr-server/content_router"
    assert captured["params"] == {"ocr_id": "ocr-123"}


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


class _FakeRAGForEnqueueError:
    def __init__(self):
        self.captured_error_files = None
        self.captured_track_id = None

    async def apipeline_enqueue_error_documents(self, error_files, track_id):
        self.captured_error_files = error_files
        self.captured_track_id = track_id


def test_pipeline_enqueue_file_preserves_register_ocr_retry_metadata(monkeypatch):
    rag = _FakeRAGForEnqueueError()

    async def _fail_fetch(_ocr_id, ocr_router=False):
        raise TimeoutError("timeout while polling")

    monkeypatch.setattr(routes, "_fetch_text_from_ocr_server", _fail_fetch)
    temp_dir = Path("G:/lightRAG/LightRAG/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / f"sample-{uuid.uuid4().hex}.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    try:
        success, track_id = asyncio.run(
            routes.pipeline_enqueue_file(
                rag,
                file_path,
                track_id="track-register-1",
                file_source=str(file_path),
                ocr_id="ocr-xyz",
                move_to_enqueued=False,
                meta_info={"file_id": "file-123"},
            )
        )
    finally:
        if file_path.exists():
            try:
                file_path.unlink()
            except PermissionError:
                pass

    assert success is False
    assert track_id == "track-register-1"
    assert rag.captured_track_id == "track-register-1"
    assert rag.captured_error_files == [
        {
            "file_path": str(file_path),
            "error_description": "[File Extraction]PDF processing error",
            "original_error": "Failed to extract text from PDF: timeout while polling",
            "file_size": len(b"%PDF-1.4\n"),
            "metadata": {
                "ocr_id": "ocr-xyz",
                "ocr_router": False,
                "meta_info": {"file_id": "file-123"},
                "retry_source": "register_external_file",
                "retry_stage": "ocr_content_fetch",
            },
        }
    ]


class _FakeDocStatusForRetry:
    def __init__(self, failed_docs):
        self.failed_docs = failed_docs
        self.deleted = []

    async def get_docs_by_status(self, status):
        assert status == routes.DocStatus.FAILED
        return self.failed_docs

    async def delete(self, doc_ids):
        self.deleted.extend(doc_ids)


class _FakeFullDocsForRetry:
    async def get_by_id(self, _doc_id):
        return None


class _FakeRetryStatusDoc:
    def __init__(self, *, file_path, track_id, metadata):
        self.file_path = file_path
        self.track_id = track_id
        self.metadata = metadata


class _FakeRAGForRetry:
    def __init__(self, failed_docs):
        self.doc_status = _FakeDocStatusForRetry(failed_docs)
        self.full_docs = _FakeFullDocsForRetry()


def test_retry_failed_registered_ocr_documents_reenqueues_missing_content(monkeypatch):
    file_path = str(Path("G:/lightRAG/LightRAG/inputs/demo.pdf"))
    rag = _FakeRAGForRetry(
        {
            "error-doc-1": _FakeRetryStatusDoc(
                file_path=file_path,
                track_id="track-register-2",
                metadata={
                    "retry_source": "register_external_file",
                    "retry_stage": "ocr_content_fetch",
                    "ocr_id": "ocr-456",
                    "ocr_router": True,
                    "meta_info": {"file_id": "file-456"},
                },
            )
        }
    )
    captured = {}

    async def _fake_pipeline_enqueue_file(
        rag_obj,
        resolved_file_path,
        track_id=None,
        file_source=None,
        ocr_id=None,
        ocr_router=False,
        move_to_enqueued=True,
        meta_info=None,
    ):
        captured["rag"] = rag_obj
        captured["resolved_file_path"] = resolved_file_path
        captured["track_id"] = track_id
        captured["file_source"] = file_source
        captured["ocr_id"] = ocr_id
        captured["ocr_router"] = ocr_router
        captured["move_to_enqueued"] = move_to_enqueued
        captured["meta_info"] = meta_info
        return True, track_id

    monkeypatch.setattr(routes, "pipeline_enqueue_file", _fake_pipeline_enqueue_file)

    retried_count = asyncio.run(routes.retry_failed_registered_ocr_documents(rag))

    assert retried_count == 1
    assert captured == {
        "rag": rag,
        "resolved_file_path": Path(file_path).expanduser().resolve(strict=False),
        "track_id": "track-register-2",
        "file_source": file_path,
        "ocr_id": "ocr-456",
        "ocr_router": True,
        "move_to_enqueued": False,
        "meta_info": {"file_id": "file-456"},
    }
    assert rag.doc_status.deleted == ["error-doc-1"]


class _FakeDocManagerForScan:
    def scan_directory_for_new_files(self):
        return []


class _FakeRAGForScan:
    def __init__(self):
        self.process_calls = []

    async def apipeline_process_enqueue_documents(self, extract_kg=True):
        self.process_calls.append(extract_kg)


def test_run_scanning_process_retries_registered_ocr_failures_when_no_new_files(
    monkeypatch,
):
    rag = _FakeRAGForScan()
    doc_manager = _FakeDocManagerForScan()

    async def _fake_retry_failed_registered_ocr_documents(_rag):
        return 2

    monkeypatch.setattr(
        routes,
        "retry_failed_registered_ocr_documents",
        _fake_retry_failed_registered_ocr_documents,
    )

    asyncio.run(routes.run_scanning_process(rag, doc_manager, track_id="scan-1"))

    assert rag.process_calls == [False]


def test_reprocess_failed_documents_with_ocr_retry_disables_kg_by_default(
    monkeypatch,
):
    rag = _FakeRAGForScan()

    async def _fake_retry_failed_registered_ocr_documents(_rag):
        return 0

    monkeypatch.setattr(
        routes,
        "retry_failed_registered_ocr_documents",
        _fake_retry_failed_registered_ocr_documents,
    )

    asyncio.run(routes.reprocess_failed_documents_with_ocr_retry(rag))

    assert rag.process_calls == [False]


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
            "image_text": "Hello\n\nWorld",
            "page_id": 57,
            "bbox": [88.0, 154.0, 417.0, 207.0],
            "page_size": [612.0, 792.0],
            "ocr_chunk_id": 270,
        },
        {
            "content": "QUJDREVGRw==",
            "content_type": "image",
            "image_text": "QUJDREVGRw==",
            "page_id": 58,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "page_size": [612.0, 792.0],
            "ocr_chunk_id": 271,
        },
    ]


def test_extract_structured_segments_merges_short_chunks_repeatedly(monkeypatch):
    monkeypatch.setattr(routes.global_args, "ocr_chunk_merge_thr", 20)

    ocr_chunks = [
        {
            "chunk_id": 1,
            "page_idx": 1,
            "bbox": [0, 0, 100, 10],
            "markdown": "alpha",
            "content_type": "text",
        },
        {
            "chunk_id": 2,
            "page_idx": 1,
            "bbox": [0, 20, 100, 30],
            "markdown": "beta",
            "content_type": "text",
        },
        {
            "chunk_id": 3,
            "page_idx": 1,
            "bbox": [0, 40, 100, 60],
            "markdown": "this is a long enough paragraph for merge target",
            "content_type": "text",
        },
    ]

    segments = routes._extract_structured_segments_from_ocr_chunks(ocr_chunks)

    assert segments == [
        {
            "content": "alpha\nbeta\nthis is a long enough paragraph for merge target",
            "content_type": "text",
            "image_text": "alpha\nbeta\nthis is a long enough paragraph for merge target",
            "page_id": 1,
            "bbox": [0.0, 0.0, 100.0, 60.0],
            "ocr_chunk_id": 3,
        }
    ]


def test_extract_structured_segments_does_not_merge_across_pages(monkeypatch):
    monkeypatch.setattr(routes.global_args, "ocr_chunk_merge_thr", 20)

    ocr_chunks = [
        {
            "chunk_id": 1,
            "page_idx": 1,
            "bbox": [0, 0, 100, 10],
            "markdown": "alpha",
            "content_type": "text",
        },
        {
            "chunk_id": 2,
            "page_idx": 2,
            "bbox": [0, 0, 100, 20],
            "markdown": "this is a long enough paragraph for page two",
            "content_type": "text",
        },
    ]

    segments = routes._extract_structured_segments_from_ocr_chunks(ocr_chunks)

    assert segments == [
        {
            "content": "alpha",
            "content_type": "text",
            "image_text": "alpha",
            "page_id": 1,
            "bbox": [0.0, 0.0, 100.0, 10.0],
            "ocr_chunk_id": 1,
        },
        {
            "content": "this is a long enough paragraph for page two",
            "content_type": "text",
            "image_text": "this is a long enough paragraph for page two",
            "page_id": 2,
            "bbox": [0.0, 0.0, 100.0, 20.0],
            "ocr_chunk_id": 2,
        },
    ]


def test_extract_structured_segments_merges_title_chunks(monkeypatch):
    monkeypatch.setattr(routes.global_args, "ocr_chunk_merge_thr", 20)

    ocr_chunks = [
        {
            "chunk_id": 1,
            "page_idx": 1,
            "bbox": [0, 0, 100, 10],
            "markdown": "chapter 1",
            "content_type": "title",
        },
        {
            "chunk_id": 2,
            "page_idx": 1,
            "bbox": [0, 20, 100, 40],
            "markdown": "this is a long enough paragraph for merge target",
            "content_type": "text",
        },
    ]

    segments = routes._extract_structured_segments_from_ocr_chunks(ocr_chunks)

    assert segments == [
        {
            "content": "chapter 1\nthis is a long enough paragraph for merge target",
            "content_type": "text",
            "image_text": "chapter 1\nthis is a long enough paragraph for merge target",
            "page_id": 1,
            "bbox": [0.0, 0.0, 100.0, 40.0],
            "ocr_chunk_id": 2,
        }
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
                    "image_text": "Segment",
                    "page_id": 1,
                    "bbox": [0.0, 1.0, 2.0, 3.0],
                    "page_size": [612.0, 792.0],
                    "ocr_chunk_id": 10,
                },
                {
                    "content": "YmFzZTY0LWltYWdl",
                    "content_type": "image",
                    "image_text": "YmFzZTY0LWltYWdl",
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

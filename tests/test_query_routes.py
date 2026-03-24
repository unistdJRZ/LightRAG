import json
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

ascii_colors_stub = types.ModuleType("ascii_colors")


class _ASCIIColors:
    @staticmethod
    def yellow(message: str) -> None:
        return None

    @staticmethod
    def red(message: str) -> None:
        return None


ascii_colors_stub.ASCIIColors = _ASCIIColors
sys.modules.setdefault("ascii_colors", ascii_colors_stub)


def _load_create_query_routes():
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        from lightrag.api.routers.query_routes import create_query_routes
    finally:
        sys.argv = argv
    return create_query_routes


class _FakeRAG:
    def __init__(self, result: dict):
        self._result = result

    async def aquery_llm(self, query: str, param=None):
        return self._result


def _build_test_client(result: dict) -> TestClient:
    create_query_routes = _load_create_query_routes()
    app = FastAPI()
    app.include_router(
        create_query_routes(
            rag_by_workspace={"default": _FakeRAG(result)},
            workspace="default",
        )
    )
    return TestClient(app)


def test_query_endpoint_references_include_page_id_and_bbox():
    client = _build_test_client(
        {
            "llm_response": {"content": "answer", "response_iterator": None, "is_streaming": False},
            "data": {
                "chunks": [
                    {
                        "reference_id": "1",
                        "chunk_id": "chunk-1",
                        "file_path": "/tmp/doc.pdf",
                        "content": "chunk preview",
                        "page_id": 7,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                    }
                ],
                "references": [
                    {
                        "reference_id": "1",
                        "chunk_id": "chunk-1",
                        "file_path": "/tmp/doc.pdf",
                    }
                ],
            },
        }
    )

    response = client.post(
        "/query",
        json={"query": "test query", "mode": "mix", "include_references": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"] == "answer"
    assert payload["references"] == [
        {
            "reference_id": "1",
            "chunk_id": "chunk-1",
            "file_path": "/tmp/doc.pdf",
            "file_id": None,
            "page_id": 7,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "content": "chunk preview",
        }
    ]


def test_query_stream_endpoint_references_include_page_id_and_bbox():
    client = _build_test_client(
        {
            "llm_response": {"content": "answer", "response_iterator": None, "is_streaming": False},
            "data": {
                "chunks": [
                    {
                        "reference_id": "1",
                        "chunk_id": "chunk-1",
                        "file_path": "/tmp/doc.pdf",
                        "content": "chunk preview",
                        "page_id": 7,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                    }
                ],
                "references": [
                    {
                        "reference_id": "1",
                        "chunk_id": "chunk-1",
                        "file_path": "/tmp/doc.pdf",
                    }
                ],
            },
        }
    )

    response = client.post(
        "/query/stream",
        json={
            "query": "test query",
            "mode": "mix",
            "include_references": True,
            "stream": False,
        },
    )

    assert response.status_code == 200
    line = response.text.strip()
    payload = json.loads(line)
    assert payload["response"] == "answer"
    assert len(payload["references"]) == 1
    assert payload["references"][0]["reference_id"] == "1"
    assert payload["references"][0]["chunk_id"] == "chunk-1"
    assert payload["references"][0]["file_path"] == "/tmp/doc.pdf"
    assert payload["references"][0]["page_id"] == 7
    assert payload["references"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert payload["references"][0]["content"] == "chunk preview"

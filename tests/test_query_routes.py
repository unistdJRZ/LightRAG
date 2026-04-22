import asyncio
import json
import queue
import sys
import threading
import types
from copy import deepcopy

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


def _load_query_routes_module():
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        import lightrag.api.routers.query_routes as query_routes
    finally:
        sys.argv = argv
    return query_routes


def _load_create_query_routes():
    return _load_query_routes_module().create_query_routes


class _FakeRAG:
    def __init__(self, result: dict):
        self._result = result
        self.last_query = None
        self.last_param = None
        self.last_agent_context = None
        self.data_calls = []
        self.llm_calls = []

    def _build_result(self, agent_context=None) -> dict:
        result = deepcopy(self._result)
        data = result.setdefault("data", {})
        chunks = list(data.get("chunks", []))
        references = list(data.get("references", []))
        entities = list(data.get("entities", []))

        if agent_context:
            next_reference_id = len(references) + 1
            for entity in agent_context.get("entities", []):
                entities.append(deepcopy(entity))
            for chunk in agent_context.get("chunks", []):
                chunk_copy = deepcopy(chunk)
                chunk_copy.setdefault("reference_id", str(next_reference_id))
                references.append(
                    {
                        "reference_id": chunk_copy["reference_id"],
                        "chunk_id": chunk_copy["chunk_id"],
                        "file_path": chunk_copy.get("file_path", "unknown_source"),
                    }
                )
                chunks.append(chunk_copy)
                next_reference_id += 1

        data["entities"] = entities
        data["chunks"] = chunks
        data["references"] = references
        return result

    async def aquery_llm(self, query: str, param=None, agent_context=None):
        self.last_query = query
        self.last_param = param
        self.last_agent_context = agent_context
        self.llm_calls.append(
            {
                "query": query,
                "param": param,
                "agent_context": agent_context,
            }
        )
        return self._build_result(agent_context)

    async def aquery_data(self, query: str, param=None, agent_context=None):
        self.last_query = query
        self.last_param = param
        self.last_agent_context = agent_context
        self.data_calls.append(
            {
                "query": query,
                "param": param,
                "agent_context": agent_context,
            }
        )
        result = self._build_result(agent_context)
        return {
            "status": "success",
            "message": "Query executed successfully",
            "data": result.get("data", {}),
            "metadata": {},
        }


def _build_test_client(result: dict) -> tuple[TestClient, _FakeRAG]:
    create_query_routes = _load_create_query_routes()
    app = FastAPI()
    fake_rag = _FakeRAG(result)
    app.include_router(
        create_query_routes(
            rag_by_workspace={"default": fake_rag},
            workspace="default",
        )
    )
    return TestClient(app), fake_rag


def _post_with_timeout(
    client: TestClient,
    path: str,
    payload: dict,
    timeout_seconds: float = 5.0,
):
    result_queue: queue.Queue = queue.Queue()
    error_queue: queue.Queue = queue.Queue()

    def _target() -> None:
        try:
            result_queue.put(client.post(path, json=payload))
        except BaseException as exc:  # noqa: BLE001
            error_queue.put(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise AssertionError(f"Request to {path} exceeded {timeout_seconds} seconds")

    if not error_queue.empty():
        raise error_queue.get()

    return result_queue.get_nowait()


def test_query_endpoint_references_include_page_id_and_bbox():
    client, _ = _build_test_client(
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
                        "content_type": "image",
                        "image_text": "page image text",
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

    response = _post_with_timeout(
        client,
        "/query",
        {"query": "test query", "mode": "mix", "include_references": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"] == "answer"
    assert payload["references"][0]["reference_id"] == "1"
    assert payload["references"][0]["chunk_id"] == "chunk-1"
    assert payload["references"][0]["file_path"] == "/tmp/doc.pdf"
    assert payload["references"][0]["file_id"] is None
    assert payload["references"][0]["page_id"] == 7
    assert payload["references"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert payload["references"][0]["content_type"] == "image"
    assert payload["references"][0]["image_text"] == "page image text"
    assert payload["references"][0]["content"] == "chunk preview"
    assert "retrieval_result" not in payload


def test_query_stream_endpoint_references_include_page_id_and_bbox():
    client, _ = _build_test_client(
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
                        "content_type": "image",
                        "image_text": "page image text",
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

    response = _post_with_timeout(
        client,
        "/query/stream",
        {
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
    assert payload["references"][0]["content_type"] == "image"
    assert payload["references"][0]["image_text"] == "page image text"
    assert payload["references"][0]["content"] == "chunk preview"
    assert "retrieval_result" not in payload


def test_query_endpoint_accepts_structured_query_payload():
    client, fake_rag = _build_test_client(
        {
            "llm_response": {"content": "answer", "response_iterator": None, "is_streaming": False},
            "data": {"chunks": [], "references": []},
        }
    )

    response = _post_with_timeout(
        client,
        "/query",
        {
            "query": {
                "history": [
                    {"role": "user", "content": "What is LightRAG?"},
                    {"role": "assistant", "content": "LightRAG is a graph-based RAG system."},
                ],
                "latest_query": "How does keyword extraction work?",
            },
            "mode": "mix",
            "include_references": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["response"] == "answer"
    assert fake_rag.last_query.model_dump() == {
        "history": [
            {"role": "user", "content": "What is LightRAG?"},
            {"role": "assistant", "content": "LightRAG is a graph-based RAG system."},
        ],
        "latest_query": "How does keyword extraction work?",
    }
    assert fake_rag.last_param is not None
    assert fake_rag.last_param.conversation_history == [
        {"role": "user", "content": "What is LightRAG?"},
        {"role": "assistant", "content": "LightRAG is a graph-based RAG system."},
    ]


def test_query_endpoint_includes_agent_search_result(monkeypatch):
    query_routes = _load_query_routes_module()
    client, fake_rag = _build_test_client(
        {
            "llm_response": {"content": "answer", "response_iterator": None, "is_streaming": False},
            "data": {"chunks": [], "references": []},
        }
    )

    async def _fake_agent_search_pipeline(
        rag, request, workspace, param=None, status_queue=None
    ):
        return query_routes.AgentSearchMergeBundle(
            public_result=query_routes.AgentSearchResultPayload(
                agent_search_id="agent-search-001",
                workspace=workspace,
                retrieval_target="[latest_query]\ntest query",
                status="completed",
                submitted=True,
                entity_count=1,
                chunk_count=1,
            ),
            search_entities=[
                {
                    "entity_name": "entity-1",
                    "description": "agent entity",
                    "source_id": "chunk-9",
                    "entity_type": "concept",
                    "file_path": "unknown_source",
                }
            ],
            search_chunks=[
                {
                    "chunk_id": "chunk-9",
                    "full_doc_id": "doc-9",
                    "file_path": "/tmp/agent.txt",
                    "content": "agent chunk",
                    "content_type": "text",
                }
            ],
            cache_signature='{"chunk_ids":["chunk-9"],"entity_ids":["entity-1"]}',
        )

    monkeypatch.setattr(
        query_routes, "_run_agent_search_pipeline", _fake_agent_search_pipeline
    )

    response = _post_with_timeout(
        client,
        "/query",
        {
            "query": "test query",
            "mode": "mix",
            "include_references": True,
            "agent_search": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_search_result"]["agent_search_id"] == "agent-search-001"
    assert payload["agent_search_result"]["chunk_count"] == 1
    assert "chunk_ids" not in payload["agent_search_result"]
    assert payload["references"][-1]["chunk_id"] == "chunk-9"
    assert payload["references"][-1]["content"] == "agent chunk"
    assert fake_rag.last_agent_context["chunks"][0]["chunk_id"] == "chunk-9"


def test_query_stream_endpoint_emits_agent_status(monkeypatch):
    query_routes = _load_query_routes_module()
    client, _ = _build_test_client(
        {
            "llm_response": {"content": "answer", "response_iterator": None, "is_streaming": False},
            "data": {"chunks": [], "references": []},
        }
    )

    async def _fake_agent_search_pipeline(
        rag, request, workspace, param=None, status_queue=None
    ):
        if status_queue is not None:
            await status_queue.put(
                {
                    "agent_search_id": "agent-search-002",
                    "phase": "event",
                    "message": "agent is searching",
                    "event_type": "message.part.updated",
                    "opencode_session_id": "ses-demo",
                }
            )
        return query_routes.AgentSearchMergeBundle(
            public_result=query_routes.AgentSearchResultPayload(
                agent_search_id="agent-search-002",
                workspace=workspace,
                retrieval_target="[latest_query]\ntest query",
                status="completed",
                submitted=True,
            ),
            search_entities=[],
            search_chunks=[],
        )

    monkeypatch.setattr(
        query_routes, "_run_agent_search_pipeline", _fake_agent_search_pipeline
    )

    response = _post_with_timeout(
        client,
        "/query/stream",
        {
            "query": "test query",
            "mode": "mix",
            "stream": True,
            "agent_search": True,
        },
    )

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.strip().splitlines()]
    assert lines[0]["agent_status"]["event_type"] == "local.started"
    assert lines[1]["agent_status"]["message"] == "agent is searching"
    assert lines[-1]["response"] == "answer"


def test_query_stream_endpoint_emits_initial_agent_status_when_agent_is_silent(
    monkeypatch,
):
    query_routes = _load_query_routes_module()
    client, _ = _build_test_client(
        {
            "llm_response": {
                "content": "answer",
                "response_iterator": None,
                "is_streaming": False,
            },
            "data": {"chunks": [], "references": []},
        }
    )

    async def _fake_agent_search_pipeline_guarded(
        rag, request, workspace, param=None, status_queue=None
    ):
        await asyncio.sleep(0.3)
        return query_routes.AgentSearchMergeBundle(
            public_result=query_routes.AgentSearchResultPayload(
                agent_search_id="agent-search-004",
                workspace=workspace,
                retrieval_target="[latest_query]\ntest query",
                status="completed",
                submitted=True,
            ),
            search_entities=[],
            search_chunks=[],
        )

    monkeypatch.setattr(
        query_routes,
        "_run_agent_search_pipeline_guarded",
        _fake_agent_search_pipeline_guarded,
    )

    response = _post_with_timeout(
        client,
        "/query/stream",
        {
            "query": "test query",
            "mode": "mix",
            "stream": True,
            "agent_search": True,
        },
    )

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.strip().splitlines()]
    assert lines[0]["agent_status"]["phase"] == "starting"
    assert lines[0]["agent_status"]["event_type"] == "local.started"
    assert lines[1]["agent_status"]["phase"] == "rag_started"
    assert lines[1]["agent_status"]["event_type"] == "local.rag_started"
    assert lines[-1]["response"] == "answer"


def test_query_data_endpoint_merges_agent_search_chunks(monkeypatch):
    query_routes = _load_query_routes_module()
    client, fake_rag = _build_test_client(
        {
            "llm_response": {"content": "answer", "response_iterator": None, "is_streaming": False},
            "data": {"entities": [], "chunks": [], "references": []},
        }
    )

    async def _fake_agent_search_pipeline(
        rag, request, workspace, param=None, status_queue=None
    ):
        return query_routes.AgentSearchMergeBundle(
            public_result=query_routes.AgentSearchResultPayload(
                agent_search_id="agent-search-003",
                workspace=workspace,
                retrieval_target="[latest_query]\ntest query",
                status="completed",
                submitted=True,
                chunk_count=1,
            ),
            search_entities=[],
            search_chunks=[
                {
                    "chunk_id": "chunk-7",
                    "full_doc_id": "doc-7",
                    "file_path": "/tmp/merged.txt",
                    "content": "merged chunk",
                    "content_type": "text",
                }
            ],
            cache_signature='{"chunk_ids":["chunk-7"],"entity_ids":[]}',
        )

    monkeypatch.setattr(
        query_routes, "_run_agent_search_pipeline", _fake_agent_search_pipeline
    )

    response = _post_with_timeout(
        client,
        "/query/data",
        {
            "query": "test query",
            "mode": "mix",
            "agent_search": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_search_result"]["agent_search_id"] == "agent-search-003"
    assert payload["data"]["chunks"][0]["chunk_id"] == "chunk-7"
    assert payload["data"]["references"][0]["chunk_id"] == "chunk-7"
    assert fake_rag.last_agent_context["chunks"][0]["chunk_id"] == "chunk-7"


def test_query_endpoint_runs_prior_rag_before_agent_search(monkeypatch):
    query_routes = _load_query_routes_module()
    client, fake_rag = _build_test_client(
        {
            "llm_response": {
                "content": "answer",
                "response_iterator": None,
                "is_streaming": False,
            },
            "data": {
                "entities": [
                    {
                        "entity_name": "LightRAG",
                        "entity_type": "project",
                        "description": "graph rag framework",
                        "source_id": "chunk-1",
                        "file_path": "/tmp/doc.txt",
                    }
                ],
                "relationships": [
                    {
                        "src_id": "LightRAG",
                        "tgt_id": "RAG",
                        "description": "implements retrieval augmentation",
                        "keywords": "retrieval, augmentation",
                    }
                ],
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "file_path": "/tmp/doc.txt",
                        "content": "LightRAG is a graph-based retrieval system.",
                    }
                ],
                "references": [],
            },
        }
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(query_routes, "is_opencode_enabled", lambda: True)

    async def _fake_run_agent_search(
        agent_search_id,
        workspace,
        retrieval_target,
        prior_rag_context=None,
        callback=None,
    ):
        captured["agent_search_id"] = agent_search_id
        captured["workspace"] = workspace
        captured["retrieval_target"] = retrieval_target
        captured["prior_rag_context"] = prior_rag_context
        return types.SimpleNamespace(
            ok=True,
            session_id="session-1",
            final_output="agent done",
            error=None,
        )

    async def _fake_wait_for_submit(rag, agent_search_id, timeout_seconds=5.0, interval_seconds=0.5):
        return {"entity_ids": [], "chunk_ids": []}, "default"

    monkeypatch.setattr(query_routes, "run_agent_search", _fake_run_agent_search)
    monkeypatch.setattr(query_routes, "_wait_for_agent_submit_payload", _fake_wait_for_submit)

    response = _post_with_timeout(
        client,
        "/query",
        {
            "query": "test query",
            "mode": "mix",
            "include_references": False,
            "agent_search": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_search_result"]["prior_rag_used"] is True
    assert payload["agent_search_result"]["prior_rag_entity_count"] == 1
    assert payload["agent_search_result"]["prior_rag_relation_count"] == 1
    assert payload["agent_search_result"]["prior_rag_chunk_count"] == 1
    assert len(fake_rag.data_calls) == 1
    assert fake_rag.data_calls[0]["agent_context"] is None
    assert len(fake_rag.llm_calls) == 1
    assert "[entities]" in captured["prior_rag_context"]
    assert "[relationships]" in captured["prior_rag_context"]
    assert "[chunks]" in captured["prior_rag_context"]


def test_query_endpoint_continues_when_prior_rag_fails(monkeypatch):
    query_routes = _load_query_routes_module()
    client, fake_rag = _build_test_client(
        {
            "llm_response": {
                "content": "answer",
                "response_iterator": None,
                "is_streaming": False,
            },
            "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
        }
    )

    original_aquery_data = fake_rag.aquery_data

    async def _failing_aquery_data(query, param=None, agent_context=None):
        if agent_context is None:
            raise RuntimeError("prior rag failed")
        return await original_aquery_data(query, param=param, agent_context=agent_context)

    fake_rag.aquery_data = _failing_aquery_data

    captured: dict[str, object] = {}

    monkeypatch.setattr(query_routes, "is_opencode_enabled", lambda: True)

    async def _fake_run_agent_search(
        agent_search_id,
        workspace,
        retrieval_target,
        prior_rag_context=None,
        callback=None,
    ):
        captured["prior_rag_context"] = prior_rag_context
        return types.SimpleNamespace(
            ok=True,
            session_id="session-2",
            final_output="agent done",
            error=None,
        )

    async def _fake_wait_for_submit(rag, agent_search_id, timeout_seconds=5.0, interval_seconds=0.5):
        return {"entity_ids": [], "chunk_ids": []}, "default"

    monkeypatch.setattr(query_routes, "run_agent_search", _fake_run_agent_search)
    monkeypatch.setattr(query_routes, "_wait_for_agent_submit_payload", _fake_wait_for_submit)

    response = _post_with_timeout(
        client,
        "/query",
        {
            "query": "test query",
            "mode": "mix",
            "include_references": False,
            "agent_search": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_search_result"]["prior_rag_used"] is False
    assert payload["agent_search_result"]["prior_rag_entity_count"] == 0
    assert payload["agent_search_result"]["prior_rag_relation_count"] == 0
    assert payload["agent_search_result"]["prior_rag_chunk_count"] == 0
    assert captured["prior_rag_context"] is None

import sys
import types
from datetime import datetime, timezone

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


def _load_create_agent_routes():
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        from lightrag.api.agent import create_agent_routes
    finally:
        sys.argv = argv
    return create_agent_routes


class _FakeDB:
    def __init__(self, rows=None, query_results=None):
        self.rows = [] if rows is None else rows
        self.query_results = list(query_results or [])
        self.last_sql = None
        self.last_params = None
        self.last_multirows = None
        self.execute_calls = []

    async def query(self, sql, params=None, multirows=False):
        self.last_sql = sql
        self.last_params = params
        self.last_multirows = multirows
        if self.query_results:
            return self.query_results.pop(0)
        return self.rows

    async def execute(self, sql, data=None, **kwargs):
        self.execute_calls.append(
            {
                "sql": sql,
                "data": data,
                "kwargs": kwargs,
            }
        )
        return "DELETE 0"


class _FakeTextChunks:
    namespace = "text_chunks"

    def __init__(self, db, workspace="default"):
        self.db = db
        self.workspace = workspace


class _FakeRAG:
    def __init__(self, text_chunks=None, graph_storage=None, llm_response_cache=None):
        self.text_chunks = text_chunks
        self.chunk_entity_relation_graph = graph_storage
        self.llm_response_cache = llm_response_cache


class _FakePGGraphStorage:
    namespace = "chunk_entity_relation"

    def __init__(self, rows, workspace="default", graph_name="graph_default"):
        self.workspace = workspace
        self.graph_name = graph_name
        self.rows = rows
        self.last_query = None
        self.last_params = None

    async def _query(self, query, params=None):
        self.last_query = query
        self.last_params = params
        return self.rows


class _FakeNeo4jResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def __aiter__(self):
        self._iter = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def consume(self):
        return None


class _FakeNeo4jSession:
    def __init__(self, rows, storage):
        self._rows = rows
        self._storage = storage

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        self._storage.last_query = query
        self._storage.last_params = params
        return _FakeNeo4jResult(self._rows)


class _FakeNeo4jDriver:
    def __init__(self, rows, storage):
        self._rows = rows
        self._storage = storage

    def session(self, database=None, default_access_mode=None):
        self._storage.last_database = database
        self._storage.last_access_mode = default_access_mode
        return _FakeNeo4jSession(self._rows, self._storage)


class _FakeNeo4jGraphStorage:
    namespace = "chunk_entity_relation"

    def __init__(self, rows, workspace="default"):
        self.workspace = workspace
        self._DATABASE = "neo4j"
        self.last_query = None
        self.last_params = None
        self.last_database = None
        self.last_access_mode = None
        self._driver = _FakeNeo4jDriver(rows, self)

    def _get_workspace_label(self):
        return self.workspace


def _build_test_client(rag) -> tuple[TestClient, _FakeDB | None]:
    create_agent_routes = _load_create_agent_routes()
    app = FastAPI()
    app.include_router(
        create_agent_routes(
            rag_by_workspace={"default": rag},
            workspace="default",
        )
    )
    db = getattr(getattr(rag, "text_chunks", None), "db", None)
    return TestClient(app), db


def test_chunk_search_endpoint_returns_rows_and_builds_sql():
    client, db = _build_test_client(
        _FakeRAG(
            _FakeTextChunks(
                _FakeDB(
                    [
                        {
                            "chunk_id": "chunk-1",
                            "doc_id": "doc-1",
                            "content": "Exact content",
                            "content_type": "text",
                        }
                    ]
                )
            )
        )
    )

    response = client.post(
        "/agent/chunk_search",
        json={
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "content": "Exact content",
            "content_like": "a%b_c",
            "content_type": ["Text", " image ", ""],
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "content": "Exact content",
                "content_type": "text",
            }
        ],
        "count": 1,
    }
    assert "FROM LIGHTRAG_DOC_CHUNKS" in db.last_sql
    assert "WHERE workspace = $1" in db.last_sql
    assert "AND id = $2" in db.last_sql
    assert "AND full_doc_id = $3" in db.last_sql
    assert "AND COALESCE(content, '') = $4" in db.last_sql
    assert "AND COALESCE(content, '') ILIKE $5 ESCAPE '\\'" in db.last_sql
    assert "AND LOWER(COALESCE(content_type, '')) = ANY($6)" in db.last_sql
    assert "LIMIT $7" in db.last_sql
    assert db.last_params == [
        "default",
        "chunk-1",
        "doc-1",
        "Exact content",
        "%a\\%b\\_c%",
        ["text", "image"],
        5,
    ]
    assert db.last_multirows is True


def test_chunk_search_accepts_full_doc_id_alias_and_string_content_type():
    client, db = _build_test_client(_FakeRAG(_FakeTextChunks(_FakeDB([]))))

    response = client.post(
        "/agent/chunk_search",
        json={
            "full_doc_id": "doc-42",
            "content_type": "table",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"chunks": [], "count": 0}
    assert "AND full_doc_id = $2" in db.last_sql
    assert "AND LOWER(COALESCE(content_type, '')) = ANY($3)" in db.last_sql
    assert "LIMIT $4" in db.last_sql
    assert db.last_params == ["default", "doc-42", ["table"], 20]


def test_chunk_search_returns_501_for_non_postgres_backing_store():
    class _NonPgTextChunks:
        namespace = "text_chunks"
        workspace = "default"

    client, _ = _build_test_client(_FakeRAG(_NonPgTextChunks()))

    response = client.post("/agent/chunk_search", json={})

    assert response.status_code == 501
    assert "requires PostgreSQL-backed text chunk storage" in response.json()["detail"]


def test_chunk_search_uses_workspace_from_body():
    create_agent_routes = _load_create_agent_routes()
    default_db = _FakeDB([])
    team_db = _FakeDB(
        [
            {
                "chunk_id": "chunk-team",
                "doc_id": "doc-team",
                "content": "team content",
                "content_type": "text",
            }
        ]
    )
    app = FastAPI()
    app.include_router(
        create_agent_routes(
            rag_by_workspace={
                "default": _FakeRAG(_FakeTextChunks(default_db, workspace="default")),
                "team-a": _FakeRAG(_FakeTextChunks(team_db, workspace="team-a")),
            },
            workspace="default",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/agent/chunk_search",
        json={
            "workspace": "team-a",
            "doc_id": "doc-team",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "chunks": [
            {
                "chunk_id": "chunk-team",
                "doc_id": "doc-team",
                "content": "team content",
                "content_type": "text",
            }
        ],
        "count": 1,
    }
    assert default_db.last_sql is None
    assert team_db.last_params == ["team-a", "doc-team", 20]


def test_chunk_full_context_returns_neighbor_rows_and_builds_sql():
    client, db = _build_test_client(
        _FakeRAG(
            _FakeTextChunks(
                _FakeDB(
                    [
                        {"chunk_id": "chunk-4", "content": "chunk 4"},
                        {"chunk_id": "chunk-5", "content": "chunk 5"},
                        {"chunk_id": "chunk-6", "content": "chunk 6"},
                    ]
                )
            )
        )
    )

    response = client.post(
        "/agent/chunk_full_context",
        json={
            "chunk_id": "chunk-5",
            "neighb": 1,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "chunks": [
            {"chunk_id": "chunk-4", "content": "chunk 4"},
            {"chunk_id": "chunk-5", "content": "chunk 5"},
            {"chunk_id": "chunk-6", "content": "chunk 6"},
        ],
        "count": 3,
    }
    assert "WITH target_chunk AS (" in db.last_sql
    assert "FROM LIGHTRAG_DOC_CHUNKS" in db.last_sql
    assert "AND id = $2" in db.last_sql
    assert "c.full_doc_id = t.full_doc_id" in db.last_sql
    assert "c.segment_order_index BETWEEN t.segment_order_index - $3 AND t.segment_order_index + $3" in db.last_sql
    assert "ORDER BY c.segment_order_index ASC, c.id ASC" in db.last_sql
    assert db.last_params == ["default", "chunk-5", 1]
    assert db.last_multirows is True


def test_chunk_full_context_returns_404_when_chunk_not_found():
    client, db = _build_test_client(_FakeRAG(_FakeTextChunks(_FakeDB([]))))

    response = client.post(
        "/agent/chunk_full_context",
        json={
            "chunk_id": "chunk-missing",
        },
    )

    assert response.status_code == 404
    assert "Chunk 'chunk-missing' not found" in response.json()["detail"]
    assert db.last_params == ["default", "chunk-missing", 3]


def test_agent_rag_chunk_endpoint_returns_top_five_chunks_and_fixed_query_params():
    class _FakeRagWithData(_FakeRAG):
        def __init__(self):
            super().__init__()
            self.last_query = None
            self.last_param = None
            self.last_agent_context = "unset"

        async def aquery_data(self, query, param=None, agent_context=None):
            self.last_query = query
            self.last_param = param
            self.last_agent_context = agent_context
            return {
                "status": "success",
                "message": "ok",
                "data": {
                    "chunks": [
                        {
                            "chunk_id": f"chunk-{index}",
                            "content": f"content {index}",
                            "content_type": "text" if index % 2 else "image",
                        }
                        for index in range(1, 8)
                    ]
                },
            }

    create_agent_routes = _load_create_agent_routes()
    fake_rag = _FakeRagWithData()
    app = FastAPI()
    app.include_router(
        create_agent_routes(
            rag_by_workspace={"default": fake_rag},
            workspace="default",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/agent/rag_chunk",
        json={
            "query": "test query",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "chunks": [
            {"id": "chunk-1", "content": "content 1", "content_type": "text"},
            {"id": "chunk-2", "content": "content 2", "content_type": "image"},
            {"id": "chunk-3", "content": "content 3", "content_type": "text"},
            {"id": "chunk-4", "content": "content 4", "content_type": "image"},
            {"id": "chunk-5", "content": "content 5", "content_type": "text"},
        ],
        "count": 5,
    }
    assert fake_rag.last_query == "test query"
    assert fake_rag.last_param is not None
    assert fake_rag.last_param.mode == "naive"
    assert fake_rag.last_param.chunk_top_k == 5
    assert fake_rag.last_param.max_total_tokens == 10000
    assert fake_rag.last_param.enable_rerank is True
    assert fake_rag.last_agent_context is None


def test_agent_rag_chunk_endpoint_returns_501_without_aquery_data_support():
    client, _ = _build_test_client(_FakeRAG())

    response = client.post(
        "/agent/rag_chunk",
        json={
            "query": "test query",
        },
    )

    assert response.status_code == 501
    assert "requires aquery_data support" in response.json()["detail"]


def test_entity_serach_postgres_returns_rows_and_splits_source_ids():
    create_agent_routes = _load_create_agent_routes()
    graph_storage = _FakePGGraphStorage(
        [
            {
                "description": "retrieval framework",
                "entity_name": "LightRAG",
                "source_id": "chunk-1<SEP>chunk-2",
                "entity_type": "project",
            }
        ],
        workspace="default",
        graph_name="graph_default",
    )
    app = FastAPI()
    app.include_router(
        create_agent_routes(
            rag_by_workspace={
                "default": _FakeRAG(graph_storage=graph_storage),
            },
            workspace="default",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/agent/entity_serach",
        json={
            "description": "retrieval",
            "entity_name": "LightRAG",
            "entity_name_like": "light",
            "entity_type": ["project", "framework"],
            "entity_type_like": ["proj"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "entities": [
            {
                "description": "retrieval framework",
                "entity_id": "LightRAG",
                "entity_name": "LightRAG",
                "source_id": ["chunk-1", "chunk-2"],
                "entity_type": "project",
            }
        ],
        "count": 1,
    }
    assert "FROM graph_default._ag_label_vertex" in graph_storage.last_query
    assert "AND entity_name = $2" in graph_storage.last_query
    assert "AND LOWER(entity_name) ILIKE $3 ESCAPE '\\'" in graph_storage.last_query
    assert "AND LOWER(COALESCE(entity_type, '')) = ANY($4)" in graph_storage.last_query
    assert "LOWER(COALESCE(entity_type, '')) ILIKE $5 ESCAPE '\\'" in graph_storage.last_query
    assert graph_storage.last_params == {
        1: "%retrieval%",
        2: "LightRAG",
        3: "%light%",
        4: ["project", "framework"],
        5: "%proj%",
    }


def test_entity_serach_neo4j_uses_workspace_from_body():
    create_agent_routes = _load_create_agent_routes()
    default_graph = _FakeNeo4jGraphStorage([], workspace="default")
    team_graph = _FakeNeo4jGraphStorage(
        [
            {
                "description": "team graph entity",
                "entity_name": "TeamEntity",
                "source_id": "chunk-a<SEP>chunk-b",
                "entity_type": "organization",
            }
        ],
        workspace="team-a",
    )
    app = FastAPI()
    app.include_router(
        create_agent_routes(
            rag_by_workspace={
                "default": _FakeRAG(graph_storage=default_graph),
                "team-a": _FakeRAG(graph_storage=team_graph),
            },
            workspace="default",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/agent/entity_serach",
        json={
            "workspace": "team-a",
            "entity_name_like": "team",
            "entity_type": "organization",
            "entity_type_like": ["org"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "entities": [
            {
                "description": "team graph entity",
                "entity_id": "TeamEntity",
                "entity_name": "TeamEntity",
                "source_id": ["chunk-a", "chunk-b"],
                "entity_type": "organization",
            }
        ],
        "count": 1,
    }
    assert default_graph.last_query is None
    assert "MATCH (n:`team-a`)" in team_graph.last_query
    assert team_graph.last_params == {
        "entity_name_like": "team",
        "entity_type_values": ["organization"],
        "entity_type_like_values": ["org"],
    }
    assert team_graph.last_database == "neo4j"
    assert team_graph.last_access_mode == "READ"


def test_submit_persists_payload_and_runs_cleanup():
    client, db = _build_test_client(
        _FakeRAG(
            text_chunks=_FakeTextChunks(
                _FakeDB(
                    query_results=[
                        {
                            "expires_at": datetime(
                                2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc
                            )
                        }
                    ]
                ),
                workspace="default",
            )
        )
    )

    response = client.post(
        "/agent/submit",
        json={
            "agent_submit_id": "submit-001",
            "entity_ids": ["entity-1", "entity-2"],
            "chunk_ids": ["chunk-1", "chunk-2"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "submitted",
        "workspace": "default",
        "agent_submit_id": "submit-001",
        "entity_ids": ["entity-1", "entity-2"],
        "chunk_ids": ["chunk-1", "chunk-2"],
        "expires_at": "2026-04-15T12:00:00Z",
    }
    assert len(db.execute_calls) == 1
    assert (
        db.execute_calls[0]["sql"]
        == "DELETE FROM LIGHTRAG_AGENT_SUBMIT_CACHE WHERE expires_at <= CURRENT_TIMESTAMP"
    )
    assert "INSERT INTO LIGHTRAG_AGENT_SUBMIT_CACHE" in db.last_sql
    assert db.last_params == [
        "default",
        "submit-001",
        '["entity-1", "entity-2"]',
        '["chunk-1", "chunk-2"]',
    ]


def test_submit_uses_workspace_from_body_and_single_id_aliases():
    create_agent_routes = _load_create_agent_routes()
    default_db = _FakeDB(query_results=[{"expires_at": datetime(2026, 4, 15, 0, 0, 0)}])
    team_db = _FakeDB(query_results=[{"expires_at": datetime(2026, 4, 15, 8, 30, 0)}])
    app = FastAPI()
    app.include_router(
        create_agent_routes(
            rag_by_workspace={
                "default": _FakeRAG(
                    text_chunks=_FakeTextChunks(default_db, workspace="default")
                ),
                "team-a": _FakeRAG(
                    text_chunks=_FakeTextChunks(team_db, workspace="team-a")
                ),
            },
            workspace="default",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/agent/submit",
        json={
            "workspace": "team-a",
            "agent_submit_id": "submit-team",
            "entity_id": "entity-x",
            "chunk_id": "chunk-9",
        },
    )

    assert response.status_code == 200
    assert response.json()["workspace"] == "team-a"
    assert response.json()["entity_ids"] == ["entity-x"]
    assert response.json()["chunk_ids"] == ["chunk-9"]
    assert default_db.last_sql is None
    assert team_db.last_params == [
        "team-a",
        "submit-team",
        '["entity-x"]',
        '["chunk-9"]',
    ]

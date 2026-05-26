"""Structured agent APIs for database retrieval."""

import asyncio
import json
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from lightrag.base import QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.namespace import NameSpace, is_namespace
from lightrag.utils import logger

from ..utils_api import (
    WorkspaceObjectProxy,
    create_workspace_scope_dependency,
    get_combined_auth_dependency,
)

_AGENT_SUBMIT_CACHE_TTL_HOURS = 24
_AGENT_SUBMIT_CACHE_CLEANUP_INTERVAL_SECONDS = 600
_agent_submit_cleanup_last_run: dict[int, float] = {}
_agent_submit_cleanup_locks: dict[int, asyncio.Lock] = {}


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _escape_like_pattern(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


class ChunkSearchRequest(BaseModel):
    workspace: Optional[str] = Field(
        default=None,
        description="Target workspace for this request. If omitted, falls back to query/header/default routing.",
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Exact match against LIGHTRAG_DOC_CHUNKS.id.",
    )
    doc_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("doc_id", "full_doc_id"),
        description="Exact match against LIGHTRAG_DOC_CHUNKS.full_doc_id.",
    )
    content: Optional[str] = Field(
        default=None,
        description="Exact match against the stored chunk content.",
    )
    content_like: Optional[str] = Field(
        default=None,
        description="Case-insensitive substring match against the stored chunk content.",
    )
    content_type: list[str] = Field(
        default_factory=list,
        description="Optional content_type filter list. Matches are case-insensitive and combined with OR.",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of matching chunks to return.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "workspace": "default",
                "doc_id": "doc-123",
                "content_like": "LightRAG",
                "content_type": ["text"],
                "top_k": 10,
            }
        }
    }

    @field_validator("chunk_id", "doc_id", "content", "content_like", mode="before")
    @classmethod
    def _strip_optional_text_fields(cls, value: Any) -> str | None:
        return _normalize_optional_text(value)

    @field_validator("content_type", mode="before")
    @classmethod
    def _normalize_content_types(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raise TypeError("content_type must be a string or a list of strings")

        normalized_values: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            normalized = _normalize_optional_text(item)
            if normalized is None:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized_values.append(lowered)
        return normalized_values


class ChunkSearchResult(BaseModel):
    chunk_id: str = Field(description="Chunk identifier")
    doc_id: str = Field(description="Document identifier from full_doc_id")
    content: str = Field(description="Stored chunk content")
    content_type: str | None = Field(
        default=None,
        description="Chunk content type when available",
    )


class ChunkSearchResponse(BaseModel):
    chunks: list[ChunkSearchResult] = Field(
        description="Matched chunk records after applying all filters."
    )
    count: int = Field(description="Number of returned chunk records.")


class ChunkFullContextRequest(BaseModel):
    workspace: Optional[str] = Field(
        default=None,
        description="Target workspace for this request. If omitted, falls back to query/header/default routing.",
    )
    chunk_id: str = Field(
        min_length=1,
        description="Target chunk id. The API first resolves this chunk's full_doc_id and segment_order_index, then fetches neighbor chunks from the same document.",
    )
    neighb: int = Field(
        default=3,
        ge=0,
        le=100,
        description="How many neighboring chunks to return on each side of the target chunk. The query window is [target.segment_order_index - neighb, target.segment_order_index + neighb], so neighb=3 returns up to 7 chunks including the target chunk when the document boundaries allow it.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "workspace": "default",
                "chunk_id": "chunk-123",
                "neighb": 3,
            }
        }
    }

    @field_validator("chunk_id", mode="before")
    @classmethod
    def _strip_chunk_id(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("chunk_id must be a non-empty string")
        return normalized


class ChunkFullContextResult(BaseModel):
    chunk_id: str = Field(description="Chunk identifier")
    content: str = Field(
        description="Stored chunk content. Callers should treat this as the canonical text payload for context stitching."
    )


class ChunkFullContextResponse(BaseModel):
    chunks: list[ChunkFullContextResult] = Field(
        description="Neighbor chunks from the same full_doc_id, ordered by segment_order_index ASC and chunk_id ASC. The target chunk is included when found. Returned size is not guaranteed to be 2 * neighb + 1 because the range is naturally clipped by document boundaries."
    )
    count: int = Field(
        description="Number of returned chunk records. Equal to chunks.length."
    )


class AgentRagChunkRequest(BaseModel):
    workspace: Optional[str] = Field(
        default=None,
        description="Target workspace for this request. If omitted, falls back to query/header/default routing.",
    )
    query: str = Field(
        min_length=3,
        description="Query text used for chunk-only retrieval.",
    )

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query_text(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None or len(normalized) < 3:
            raise ValueError("query must be at least 3 characters long")
        return normalized


class AgentRagChunkResult(BaseModel):
    id: str = Field(description="Chunk identifier")
    content: str = Field(description="Chunk content")
    content_type: str | None = Field(
        default=None,
        description="Chunk content type when available",
    )


class AgentRagChunkResponse(BaseModel):
    chunks: list[AgentRagChunkResult] = Field(
        description="Top five chunks returned by naive chunk-only retrieval."
    )
    count: int = Field(description="Number of returned chunks.")


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raise TypeError("Value must be a string or a list of strings")

    normalized_values: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        normalized = _normalize_optional_text(item)
        if normalized is None:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized_values.append(lowered)
    return normalized_values


def _split_source_id_list(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [item for item in value.split(GRAPH_FIELD_SEP) if item]


def _format_datetime_utc(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, datetime):
        return str(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class EntitySearchRequest(BaseModel):
    workspace: Optional[str] = Field(
        default=None,
        description="Target workspace for this request. If omitted, falls back to query/header/default routing.",
    )
    description: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("description", "description_like"),
        description="Case-insensitive substring match on entity description.",
    )
    entity_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("entity_name", "entity_id"),
        description="Exact match on graph node entity_id.",
    )
    entity_name_like: Optional[str] = Field(
        default=None,
        description="Case-insensitive substring match on graph node entity_id.",
    )
    entity_type: list[str] = Field(
        default_factory=list,
        description="Exact entity_type filter list. Matches are case-insensitive.",
    )
    entity_type_like: list[str] = Field(
        default_factory=list,
        description="Case-insensitive substring match list for entity_type.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "workspace": "default",
                "description": "retrieval",
                "entity_name_like": "LightRAG",
                "entity_type": ["organization", "project"],
                "entity_type_like": ["org"],
            }
        }
    }

    @field_validator("description", "entity_name", "entity_name_like", mode="before")
    @classmethod
    def _strip_optional_entity_text_fields(cls, value: Any) -> str | None:
        return _normalize_optional_text(value)

    @field_validator("entity_type", "entity_type_like", mode="before")
    @classmethod
    def _normalize_entity_type_fields(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)


class EntitySearchResult(BaseModel):
    description: str | None = Field(
        default=None,
        description="Entity description when available",
    )
    entity_id: str = Field(
        description="Canonical entity id for downstream agent submit payloads"
    )
    entity_name: str = Field(description="Entity identifier from graph storage")
    source_id: list[str] = Field(
        default_factory=list,
        description="Chunk ids parsed from the source_id field",
    )
    entity_type: str | None = Field(
        default=None,
        description="Entity type when available",
    )


class EntitySearchResponse(BaseModel):
    entities: list[EntitySearchResult] = Field(
        description="Matched entity records after applying all filters."
    )
    count: int = Field(description="Number of returned entity records.")


class AgentSubmitRequest(BaseModel):
    workspace: Optional[str] = Field(
        default=None,
        description="Target workspace for this request. If omitted, falls back to query/header/default routing.",
    )
    agent_submit_id: str = Field(
        min_length=1,
        description="Caller-provided id used as the cache key for this submit payload.",
    )
    entity_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("entity_ids", "entity_id"),
        description="Entity ids to submit. Accepts either a string or a list of strings.",
    )
    chunk_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("chunk_ids", "chunk_id"),
        description="Chunk ids to submit. Accepts either a string or a list of strings.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "workspace": "default",
                "agent_submit_id": "submit-001",
                "entity_ids": ["entity-1", "entity-2"],
                "chunk_ids": ["chunk-1", "chunk-2"],
            }
        }
    }

    @field_validator("agent_submit_id", mode="before")
    @classmethod
    def _strip_agent_submit_id(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("agent_submit_id must be a non-empty string")
        return normalized

    @field_validator("entity_ids", "chunk_ids", mode="before")
    @classmethod
    def _normalize_submit_ids(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raise TypeError("submit ids must be a string or a list of strings")

        normalized_values: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            normalized = _normalize_optional_text(item)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            normalized_values.append(normalized)
        return normalized_values

    @field_validator("chunk_ids", mode="after")
    @classmethod
    def _validate_chunk_ids(cls, value: list[str]) -> list[str]:
        for chunk_id in value:
            if not chunk_id.startswith("chunk-"):
                raise ValueError("chunk_ids must contain ids starting with 'chunk-'")
        return value

    @model_validator(mode="after")
    def _ensure_payload_not_empty(self):
        if not self.entity_ids and not self.chunk_ids:
            raise ValueError("At least one of entity_ids or chunk_ids must be provided")
        return self


class AgentSubmitResponse(BaseModel):
    status: str = Field(description="Submission result status")
    workspace: str = Field(description="Resolved workspace used for storage")
    agent_submit_id: str = Field(description="Cache key for this submit payload")
    entity_ids: list[str] = Field(description="Stored entity ids")
    chunk_ids: list[str] = Field(description="Stored chunk ids")
    expires_at: str = Field(description="UTC expiration time for the cache entry")


class WorkspaceInfoRequest(BaseModel):
    workspace: Optional[str] = Field(
        default=None,
        description="Target workspace for this request. If omitted, falls back to query/header/default routing.",
    )


class WorkspaceInfoResponse(BaseModel):
    workspace: str = Field(description="Resolved workspace id")
    alias: str = Field(description="Workspace display alias")
    description: str = Field(description="Stored workspace description")
    has_description: bool = Field(description="Whether a non-empty description exists")


def _resolve_chunk_storage(rag: Any) -> Any:
    text_chunks = getattr(rag, "text_chunks", None)
    namespace = str(getattr(text_chunks, "namespace", "") or "")
    if text_chunks is None or not is_namespace(namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
        raise HTTPException(
            status_code=500,
            detail="Current workspace is not configured with a text_chunks storage namespace.",
        )

    db = getattr(text_chunks, "db", None)
    if not callable(getattr(db, "query", None)):
        raise HTTPException(
            status_code=501,
            detail="The /api/agent/chunk_search endpoint currently requires PostgreSQL-backed text chunk storage.",
        )

    return text_chunks


def _resolve_postgres_storage(
    rag: Any,
    endpoint: str = "/api/agent/submit",
) -> Any:
    for attr_name in ("text_chunks", "llm_response_cache", "doc_status"):
        storage = getattr(rag, attr_name, None)
        db = getattr(storage, "db", None)
        if callable(getattr(db, "query", None)) and callable(getattr(db, "execute", None)):
            return storage

    raise HTTPException(
        status_code=501,
        detail=f"The {endpoint} endpoint currently requires PostgreSQL-backed storage.",
    )


async def _get_workspace_description(rag: Any) -> tuple[str, str]:
    storage = _resolve_postgres_storage(rag, endpoint="/api/agent/workspace_info")
    records = await storage.db.query(
        """
        SELECT COALESCE(description, '') AS description
        FROM LIGHTRAG_WORKSPACE_INFO
        WHERE workspace = $1
        LIMIT 1
        """,
        [storage.workspace],
        multirows=True,
    )
    if not records:
        return storage.workspace, ""
    return storage.workspace, str(records[0].get("description") or "")


async def _maybe_cleanup_agent_submit_cache(db: Any) -> bool:
    db_key = id(db)
    now = time.monotonic()
    last_run = _agent_submit_cleanup_last_run.get(db_key)
    if (
        last_run is not None
        and now - last_run < _AGENT_SUBMIT_CACHE_CLEANUP_INTERVAL_SECONDS
    ):
        return False

    lock = _agent_submit_cleanup_locks.setdefault(db_key, asyncio.Lock())
    if lock.locked():
        return False

    async with lock:
        now = time.monotonic()
        last_run = _agent_submit_cleanup_last_run.get(db_key)
        if (
            last_run is not None
            and now - last_run < _AGENT_SUBMIT_CACHE_CLEANUP_INTERVAL_SECONDS
        ):
            return False

        await db.execute(
            "DELETE FROM LIGHTRAG_AGENT_SUBMIT_CACHE WHERE expires_at <= CURRENT_TIMESTAMP"
        )
        _agent_submit_cleanup_last_run[db_key] = time.monotonic()
        return True


async def _submit_agent_payload(
    rag: Any, payload: AgentSubmitRequest
) -> tuple[dict[str, Any], str]:
    storage = _resolve_postgres_storage(rag)
    db = storage.db

    await _maybe_cleanup_agent_submit_cache(db)

    sql = f"""
    INSERT INTO LIGHTRAG_AGENT_SUBMIT_CACHE (
        workspace,
        id,
        entity_ids,
        chunk_ids,
        expires_at
    )
    VALUES (
        $1,
        $2,
        $3::jsonb,
        $4::jsonb,
        CURRENT_TIMESTAMP + INTERVAL '{_AGENT_SUBMIT_CACHE_TTL_HOURS} hours'
    )
    ON CONFLICT (workspace, id) DO UPDATE
    SET
        entity_ids = EXCLUDED.entity_ids,
        chunk_ids = EXCLUDED.chunk_ids,
        updated_at = CURRENT_TIMESTAMP,
        expires_at = CURRENT_TIMESTAMP + INTERVAL '{_AGENT_SUBMIT_CACHE_TTL_HOURS} hours'
    RETURNING expires_at
    """
    result = await db.query(
        sql,
        [
            storage.workspace,
            payload.agent_submit_id,
            json.dumps(payload.entity_ids),
            json.dumps(payload.chunk_ids),
        ],
    )
    if not result:
        raise RuntimeError("Failed to persist agent submit payload")

    return result, storage.workspace


async def _get_agent_submit_payload(
    rag: Any, agent_submit_id: str
) -> tuple[dict[str, Any] | None, str]:
    storage = _resolve_postgres_storage(rag)
    db = storage.db

    sql = """
    SELECT
        entity_ids,
        chunk_ids,
        expires_at
    FROM LIGHTRAG_AGENT_SUBMIT_CACHE
    WHERE workspace = $1
      AND id = $2
      AND expires_at > CURRENT_TIMESTAMP
    LIMIT 1
    """
    records = await db.query(sql, [storage.workspace, agent_submit_id], multirows=True)
    if not records:
        return None, storage.workspace

    record = records[0]
    entity_ids = record.get("entity_ids") or []
    chunk_ids = record.get("chunk_ids") or []

    if isinstance(entity_ids, str):
        try:
            entity_ids = json.loads(entity_ids)
        except json.JSONDecodeError:
            entity_ids = [entity_ids]
    if isinstance(chunk_ids, str):
        try:
            chunk_ids = json.loads(chunk_ids)
        except json.JSONDecodeError:
            chunk_ids = [chunk_ids]

    return {
        "entity_ids": [
            str(item).strip()
            for item in entity_ids
            if str(item).strip()
        ]
        if isinstance(entity_ids, list)
        else [],
        "chunk_ids": [
            str(item).strip()
            for item in chunk_ids
            if str(item).strip()
        ]
        if isinstance(chunk_ids, list)
        else [],
        "expires_at": record.get("expires_at"),
    }, storage.workspace


async def _search_doc_chunks(rag: Any, payload: ChunkSearchRequest) -> list[dict[str, Any]]:
    text_chunks = _resolve_chunk_storage(rag)
    sql_lines = [
        "SELECT",
        "    id AS chunk_id,",
        "    full_doc_id AS doc_id,",
        "    COALESCE(content, '') AS content,",
        "    content_type",
        "FROM LIGHTRAG_DOC_CHUNKS",
        "WHERE workspace = $1",
    ]
    params: list[Any] = [text_chunks.workspace]

    if payload.chunk_id is not None:
        params.append(payload.chunk_id)
        sql_lines.append(f"  AND id = ${len(params)}")

    if payload.doc_id is not None:
        params.append(payload.doc_id)
        sql_lines.append(f"  AND full_doc_id = ${len(params)}")

    if payload.content is not None:
        params.append(payload.content)
        sql_lines.append(f"  AND COALESCE(content, '') = ${len(params)}")

    if payload.content_like is not None:
        params.append(f"%{_escape_like_pattern(payload.content_like)}%")
        sql_lines.append(f"  AND COALESCE(content, '') ILIKE ${len(params)} ESCAPE '\\'")

    if payload.content_type:
        params.append(payload.content_type)
        sql_lines.append(
            f"  AND LOWER(COALESCE(content_type, '')) = ANY(${len(params)})"
        )

    sql_lines.extend(
        [
            "ORDER BY update_time DESC NULLS LAST, full_doc_id ASC, chunk_order_index ASC NULLS LAST, id ASC",
        ]
    )
    params.append(payload.top_k)
    sql_lines.append(f"LIMIT ${len(params)}")

    records = await text_chunks.db.query("\n".join(sql_lines), params, multirows=True)
    return records or []


async def _get_chunk_full_context(
    rag: Any, payload: ChunkFullContextRequest
) -> list[dict[str, Any]]:
    text_chunks = _resolve_chunk_storage(rag)
    sql = """
    WITH target_chunk AS (
        SELECT
            id,
            full_doc_id,
            segment_order_index
        FROM LIGHTRAG_DOC_CHUNKS
        WHERE workspace = $1
          AND id = $2
        LIMIT 1
    )
    SELECT
        c.id AS chunk_id,
        COALESCE(c.content, '') AS content
    FROM LIGHTRAG_DOC_CHUNKS c
    JOIN target_chunk t
      ON c.workspace = $1
     AND c.full_doc_id = t.full_doc_id
    WHERE t.segment_order_index IS NOT NULL
      AND c.segment_order_index IS NOT NULL
      AND c.segment_order_index BETWEEN t.segment_order_index - $3 AND t.segment_order_index + $3
    ORDER BY c.segment_order_index ASC, c.id ASC
    """
    records = await text_chunks.db.query(
        sql,
        [text_chunks.workspace, payload.chunk_id, payload.neighb],
        multirows=True,
    )
    return records or []


async def _run_agent_rag_chunk_query(
    rag: Any, payload: AgentRagChunkRequest
) -> list[dict[str, Any]]:
    if not callable(getattr(rag, "aquery_data", None)):
        raise HTTPException(
            status_code=501,
            detail="The /api/agent/rag_chunk endpoint requires aquery_data support in the current workspace.",
        )

    result = await rag.aquery_data(
        payload.query,
        param=QueryParam(
            mode="naive",
            chunk_top_k=5,
            max_total_tokens=10000,
            enable_rerank=True,
        ),
        agent_context=None,
    )
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, dict):
        return []

    chunks = data.get("chunks") or []
    if not isinstance(chunks, list):
        return []

    normalized_chunks: list[dict[str, Any]] = []
    for chunk in chunks[:5]:
        if not isinstance(chunk, dict):
            continue
        chunk_id = _normalize_optional_text(chunk.get("chunk_id") or chunk.get("id"))
        if chunk_id is None:
            continue
        normalized_chunks.append(
            {
                "id": chunk_id,
                "content": str(chunk.get("content", "") or ""),
                "content_type": _normalize_optional_text(chunk.get("content_type")),
            }
        )
    return normalized_chunks


def _resolve_graph_storage(rag: Any) -> Any:
    graph_storage = getattr(rag, "chunk_entity_relation_graph", None)
    namespace = str(getattr(graph_storage, "namespace", "") or "")
    if graph_storage is None or not is_namespace(
        namespace, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
    ):
        raise HTTPException(
            status_code=500,
            detail="Current workspace is not configured with a chunk_entity_relation_graph storage namespace.",
        )
    return graph_storage


async def _search_graph_entities_postgres(
    graph_storage: Any, payload: EntitySearchRequest
) -> list[dict[str, Any]]:
    sql_lines = [
        "WITH entity_rows AS (",
        "    SELECT",
        "        (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"entity_id\"'::agtype]))::text AS entity_name,",
        "        (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"description\"'::agtype]))::text AS description,",
        "        (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"source_id\"'::agtype]))::text AS source_id,",
        "        (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"entity_type\"'::agtype]))::text AS entity_type",
        f"    FROM {graph_storage.graph_name}._ag_label_vertex",
        ")",
        "SELECT description, entity_name, source_id, entity_type",
        "FROM entity_rows",
        "WHERE entity_name IS NOT NULL",
    ]
    params: list[Any] = []

    if payload.description is not None:
        params.append(f"%{_escape_like_pattern(payload.description.lower())}%")
        sql_lines.append(
            f"  AND LOWER(COALESCE(description, '')) ILIKE ${len(params)} ESCAPE '\\'"
        )

    if payload.entity_name is not None:
        params.append(payload.entity_name)
        sql_lines.append(f"  AND entity_name = ${len(params)}")

    if payload.entity_name_like is not None:
        params.append(f"%{_escape_like_pattern(payload.entity_name_like.lower())}%")
        sql_lines.append(
            f"  AND LOWER(entity_name) ILIKE ${len(params)} ESCAPE '\\'"
        )

    if payload.entity_type:
        params.append(payload.entity_type)
        sql_lines.append(
            f"  AND LOWER(COALESCE(entity_type, '')) = ANY(${len(params)})"
        )

    if payload.entity_type_like:
        like_clauses: list[str] = []
        for value in payload.entity_type_like:
            params.append(f"%{_escape_like_pattern(value)}%")
            like_clauses.append(
                f"LOWER(COALESCE(entity_type, '')) ILIKE ${len(params)} ESCAPE '\\'"
            )
        sql_lines.append(f"  AND ({' OR '.join(like_clauses)})")

    sql_lines.append("ORDER BY entity_name ASC")
    records = await graph_storage._query("\n".join(sql_lines), params=dict(enumerate(params, 1)))
    return records or []


async def _search_graph_entities_neo4j(
    graph_storage: Any, payload: EntitySearchRequest
) -> list[dict[str, Any]]:
    workspace_label = graph_storage._get_workspace_label()
    conditions = ["n.entity_id IS NOT NULL"]
    params: dict[str, Any] = {}

    if payload.description is not None:
        params["description_like"] = payload.description.lower()
        conditions.append(
            "toLower(coalesce(n.description, '')) CONTAINS $description_like"
        )

    if payload.entity_name is not None:
        params["entity_name"] = payload.entity_name
        conditions.append("n.entity_id = $entity_name")

    if payload.entity_name_like is not None:
        params["entity_name_like"] = payload.entity_name_like.lower()
        conditions.append("toLower(n.entity_id) CONTAINS $entity_name_like")

    if payload.entity_type:
        params["entity_type_values"] = payload.entity_type
        conditions.append(
            "toLower(coalesce(n.entity_type, '')) IN $entity_type_values"
        )

    if payload.entity_type_like:
        params["entity_type_like_values"] = payload.entity_type_like
        conditions.append(
            "ANY(term IN $entity_type_like_values WHERE toLower(coalesce(n.entity_type, '')) CONTAINS term)"
        )

    cypher_query = f"""
    MATCH (n:`{workspace_label}`)
    WHERE {' AND '.join(conditions)}
    RETURN
        n.description AS description,
        n.entity_id AS entity_name,
        n.source_id AS source_id,
        n.entity_type AS entity_type
    ORDER BY entity_name ASC
    """

    async with graph_storage._driver.session(
        database=graph_storage._DATABASE, default_access_mode="READ"
    ) as session:
        result = await session.run(cypher_query, **params)
        records = [dict(record) async for record in result]
        await result.consume()
    return records


async def _search_graph_entities(
    rag: Any, payload: EntitySearchRequest
) -> list[dict[str, Any]]:
    graph_storage = _resolve_graph_storage(rag)

    if callable(getattr(graph_storage, "_query", None)) and hasattr(
        graph_storage, "graph_name"
    ):
        return await _search_graph_entities_postgres(graph_storage, payload)

    if getattr(graph_storage, "_driver", None) is not None and hasattr(
        graph_storage, "_DATABASE"
    ):
        return await _search_graph_entities_neo4j(graph_storage, payload)

    raise HTTPException(
        status_code=501,
        detail="The /api/agent/entity_serach endpoint currently supports PostgreSQL AGE and Neo4j graph storage only.",
    )


def _format_entity_search_results(records: list[dict[str, Any]]) -> list[EntitySearchResult]:
    return [
        EntitySearchResult(
            description=_normalize_optional_text(record.get("description")),
            entity_id=str(record.get("entity_name") or ""),
            entity_name=str(record.get("entity_name") or ""),
            source_id=_split_source_id_list(record.get("source_id")),
            entity_type=_normalize_optional_text(record.get("entity_type")),
        )
        for record in records
        if _normalize_optional_text(record.get("entity_name")) is not None
    ]


def create_agent_routes(
    rag_by_workspace: dict[str, Any],
    api_key: Optional[str] = None,
    workspace: str = "",
    workspace_aliases: Mapping[str, str] | None = None,
    workspace_display_names: Mapping[str, str] | None = None,
):
    if not rag_by_workspace:
        raise ValueError("rag_by_workspace cannot be empty")

    default_workspace = workspace.strip() or next(iter(rag_by_workspace.keys()))
    if default_workspace not in rag_by_workspace:
        raise ValueError(
            f"Default workspace '{default_workspace}' not found in rag_by_workspace"
        )

    workspace_context: ContextVar[str] = ContextVar(
        "agent_workspace", default=default_workspace
    )
    rag = WorkspaceObjectProxy(
        rag_by_workspace,
        default_workspace=default_workspace,
        workspace_context=workspace_context,
    )
    workspace_scope = create_workspace_scope_dependency(
        rag_by_workspace,
        workspace_context=workspace_context,
        default_workspace=default_workspace,
        workspace_aliases=workspace_aliases,
    )
    combined_auth = get_combined_auth_dependency(api_key)
    alias_by_workspace = dict(workspace_display_names or {})
    if workspace_aliases:
        for alias, workspace_id in workspace_aliases.items():
            alias_by_workspace.setdefault(workspace_id, alias)
    router = APIRouter(
        prefix="/agent",
        tags=["agent"],
        dependencies=[Depends(workspace_scope)],
    )

    def _workspace_alias(workspace_id: str) -> str:
        return alias_by_workspace.get(workspace_id) or workspace_id or "default"

    async def _workspace_info_response(current_rag: Any) -> WorkspaceInfoResponse:
        try:
            resolved_workspace, description = await _get_workspace_description(current_rag)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to fetch workspace info: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch workspace info: {exc}",
            ) from exc

        description = description.strip()
        return WorkspaceInfoResponse(
            workspace=resolved_workspace,
            alias=_workspace_alias(resolved_workspace),
            description=description,
            has_description=bool(description),
        )

    @router.get(
        "/workspace_info",
        dependencies=[Depends(combined_auth)],
        response_model=WorkspaceInfoResponse,
        summary="Get workspace description for agents",
        description="Returns the configured description of the resolved workspace so agents can decide whether to query it.",
    )
    async def workspace_info_get():
        return await _workspace_info_response(rag.resolve_current())

    @router.post(
        "/workspace_info",
        dependencies=[Depends(combined_auth)],
        response_model=WorkspaceInfoResponse,
        summary="Get workspace description for agents",
        description="Returns the configured description of the resolved workspace so agents can decide whether to query it.",
    )
    async def workspace_info_post(payload: WorkspaceInfoRequest):
        return await _workspace_info_response(rag.resolve_current())

    @router.post(
        "/chunk_search",
        dependencies=[Depends(combined_auth)],
        response_model=ChunkSearchResponse,
        summary="Search doc chunks with structured filters",
        description="Structured retrieval API for LIGHTRAG_DOC_CHUNKS intended for external LLM agents.",
    )
    async def chunk_search(payload: ChunkSearchRequest):
        current_rag = rag.resolve_current()
        current_workspace = str(
            getattr(getattr(current_rag, "text_chunks", None), "workspace", default_workspace)
            or default_workspace
        )
        try:
            chunks = await _search_doc_chunks(current_rag, payload)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to search doc chunks in workspace '%s': %s",
                current_workspace,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to search doc chunks: {exc}",
            ) from exc

        return ChunkSearchResponse(
            chunks=[ChunkSearchResult(**chunk) for chunk in chunks],
            count=len(chunks),
        )

    @router.post(
        "/chunk_full_context",
        dependencies=[Depends(combined_auth)],
        response_model=ChunkFullContextResponse,
        summary="Get neighboring chunks from the same source document",
        description=(
            "Given a chunk_id, this endpoint first resolves the target chunk's full_doc_id and "
            "segment_order_index, then returns chunks from the same full_doc_id whose "
            "segment_order_index falls within [target.segment_order_index - neighb, "
            "target.segment_order_index + neighb]. "
            "The target chunk itself is included. "
            "Results are ordered by segment_order_index ASC and chunk_id ASC, so callers can "
            "directly use the returned order as the natural document order for context assembly. "
            "The returned size is not fixed: when the target chunk is near the beginning or end "
            "of the document, out-of-range neighbors are naturally clipped instead of padded. "
            "For example, if the target chunk belongs to doc A and has segment_order_index=2 with "
            "neighb=3, the effective range is [-1, 5], and the actual returned chunk orders are "
            "[0, 1, 2, 3, 4, 5] because negative indexes do not exist. "
            "Successful responses return a minimal payload shape: chunks[].chunk_id, "
            "chunks[].content, and count. Callers should not assume a fixed number of chunks."
        ),
    )
    async def chunk_full_context(payload: ChunkFullContextRequest):
        current_rag = rag.resolve_current()
        current_workspace = str(
            getattr(getattr(current_rag, "text_chunks", None), "workspace", default_workspace)
            or default_workspace
        )
        try:
            chunks = await _get_chunk_full_context(current_rag, payload)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to fetch chunk full context in workspace '%s' for chunk '%s': %s",
                current_workspace,
                payload.chunk_id,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch chunk full context: {exc}",
            ) from exc

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Chunk '{payload.chunk_id}' not found in workspace '{current_workspace}' "
                    "or it does not have segment_order_index."
                ),
            )

        return ChunkFullContextResponse(
            chunks=[ChunkFullContextResult(**chunk) for chunk in chunks],
            count=len(chunks),
        )

    @router.post(
        "/rag_chunk",
        dependencies=[Depends(combined_auth)],
        response_model=AgentRagChunkResponse,
        summary="Run chunk-only RAG retrieval for agents",
        description=(
            "A simplified agent-facing retrieval endpoint. It internally runs aquery_data "
            "with fixed parameters: mode='naive', agent_search disabled by design, "
            "chunk_top_k=5, max_total_tokens=10000, and enable_rerank=True. "
            "The response is reduced to a structured list of up to five chunks, each "
            "containing id, content, and content_type."
        ),
    )
    async def rag_chunk(payload: AgentRagChunkRequest):
        current_rag = rag.resolve_current()
        current_workspace = str(
            getattr(getattr(current_rag, "text_chunks", None), "workspace", default_workspace)
            or default_workspace
        )
        try:
            chunks = await _run_agent_rag_chunk_query(current_rag, payload)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to run agent rag chunk query in workspace '%s': %s",
                current_workspace,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to run agent rag chunk query: {exc}",
            ) from exc

        return AgentRagChunkResponse(
            chunks=[AgentRagChunkResult(**chunk) for chunk in chunks],
            count=len(chunks),
        )

    @router.post(
        "/entity_serach",
        dependencies=[Depends(combined_auth)],
        response_model=EntitySearchResponse,
        summary="Search graph entities with structured filters",
        description="Structured retrieval API for graph entities intended for external LLM agents.",
    )
    async def entity_serach(payload: EntitySearchRequest):
        current_rag = rag.resolve_current()
        current_workspace = str(
            getattr(
                getattr(current_rag, "chunk_entity_relation_graph", None),
                "workspace",
                default_workspace,
            )
            or default_workspace
        )
        try:
            entities = _format_entity_search_results(
                await _search_graph_entities(current_rag, payload)
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to search graph entities in workspace '%s': %s",
                current_workspace,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to search graph entities: {exc}",
            ) from exc

        return EntitySearchResponse(
            entities=entities,
            count=len(entities),
        )

    @router.post(
        "/submit",
        dependencies=[Depends(combined_auth)],
        response_model=AgentSubmitResponse,
        summary="Persist agent submit ids into cache storage",
        description="Stores agent-submitted entity ids and chunk ids in PostgreSQL-backed cache storage with a 24-hour TTL.",
    )
    async def submit(payload: AgentSubmitRequest):
        current_rag = rag.resolve_current()
        try:
            result, resolved_workspace = await _submit_agent_payload(current_rag, payload)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to persist agent submit payload: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to persist agent submit payload: {exc}",
            ) from exc

        return AgentSubmitResponse(
            status="submitted",
            workspace=resolved_workspace,
            agent_submit_id=payload.agent_submit_id,
            entity_ids=payload.entity_ids,
            chunk_ids=payload.chunk_ids,
            expires_at=_format_datetime_utc(result.get("expires_at")),
        )

    return router

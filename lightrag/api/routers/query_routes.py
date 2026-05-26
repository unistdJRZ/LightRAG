"""
This module contains all query-related routes for the LightRAG API.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from contextvars import ContextVar
from typing import Any, Dict, List, Literal, Mapping, Optional
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException
from lightrag.base import QueryParam
from lightrag.api.agent.routes import (
    _get_agent_submit_payload,
    _resolve_graph_storage,
)
from lightrag.api.config import global_args
from lightrag.api.opencode_client import (
    OpencodeStatusEvent,
    is_opencode_enabled,
    run_agent_search,
)
from lightrag.api.utils_api import (
    get_combined_auth_dependency,
    WorkspaceObjectProxy,
    create_workspace_scope_dependency,
)
from lightrag.utils import (
    get_content_summary,
    get_chunk_image_fields,
    is_image_content_type,
    logger,
)
from lightrag.operate import _stringify_history_reference_items
from pydantic import BaseModel, Field, field_validator

router = APIRouter(tags=["query"])
_agent_search_run_state: ContextVar[dict[str, Any] | None] = ContextVar(
    "agent_search_run_state", default=None
)


def _validate_history_references(
    references: Any, field_name: str = "references"
) -> None:
    if references is None:
        return
    if not isinstance(references, list):
        raise ValueError(f"'{field_name}' must be a list when provided.")

    for ref in references:
        if not isinstance(ref, dict):
            raise ValueError(f"Each item in '{field_name}' must be an object.")


def _log_query_request(endpoint: str, request: "QueryRequest") -> None:
    """Emit request parameters when DEBUG logging is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    try:
        payload = request.model_dump(exclude_none=False)
    except Exception as exc:
        logger.debug("Failed to serialize %s request payload: %s", endpoint, exc)
        return

    logger.debug(
        "Received %s request params: %s",
        endpoint,
        json.dumps(payload, ensure_ascii=False, default=str),
    )


def _log_query_response(endpoint: str, response_content: Any) -> None:
    """Emit only the response text when DEBUG logging is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "Returning %s response: %s",
        endpoint,
        json.dumps(
            {"response": str(response_content or "")},
            ensure_ascii=False,
            default=str,
        ),
    )


class QueryInputPayload(BaseModel):
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation history for keyword extraction and, by default, LLM response context.",
    )
    latest_query: str = Field(
        description="The latest user message to execute against the knowledge base.",
    )

    @field_validator("latest_query", mode="after")
    @classmethod
    def latest_query_strip_after(cls, latest_query: str) -> str:
        latest_query = latest_query.strip()
        if len(latest_query) < 3:
            raise ValueError("latest_query must be at least 3 characters long")
        return latest_query

    @field_validator("history", mode="after")
    @classmethod
    def history_role_check(
        cls, history: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]]:
        if history is None:
            return []
        for msg in history:
            if "role" not in msg:
                raise ValueError("Each message must have a 'role' key.")
            if not isinstance(msg["role"], str) or not msg["role"].strip():
                raise ValueError("Each message 'role' must be a non-empty string.")
            _validate_history_references(msg.get("references"))
        return history


class QueryRequest(BaseModel):
    workspace: Optional[str] = Field(
        default=None,
        description="Target workspace for this request. If omitted, falls back to query/header/default routing.",
    )

    query: str | QueryInputPayload = Field(
        description="The query text, or an object with `history` and `latest_query` for history-aware keyword extraction.",
    )

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="mix",
        description="Query mode",
    )

    only_need_context: Optional[bool] = Field(
        default=None,
        description="If True, only returns the retrieved context without generating a response.",
    )

    only_need_prompt: Optional[bool] = Field(
        default=None,
        description="If True, only returns the generated prompt without producing a response.",
    )

    response_type: Optional[str] = Field(
        min_length=1,
        default=None,
        description="Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'.",
    )

    top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode.",
    )

    chunk_top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of text chunks to retrieve initially from vector search and keep after reranking.",
    )

    max_entity_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for entity context in unified token control system.",
        ge=1,
    )

    max_relation_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for relationship context in unified token control system.",
        ge=1,
    )

    max_total_tokens: Optional[int] = Field(
        default=None,
        description="Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt).",
        ge=1,
    )

    hl_keywords: list[str] = Field(
        default_factory=list,
        description="List of high-level keywords to prioritize in retrieval. Leave empty to use the LLM to generate the keywords.",
    )

    ll_keywords: list[str] = Field(
        default_factory=list,
        description="List of low-level keywords to refine retrieval focus. Leave empty to use the LLM to generate the keywords.",
    )

    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="History messages are only sent to LLM for context, not used for retrieval. Format: [{'role': 'user/assistant', 'content': 'message', 'references': [{'reference_id': '1', 'chunk_id': 'chunk-1', 'workspace': 'default'}]}].",
    )

    user_prompt: Optional[str] = Field(
        default=None,
        description="User-provided prompt for the query. If provided, this will be used instead of the default value from prompt template.",
    )

    enable_rerank: Optional[bool] = Field(
        default=None,
        description="Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued. Default is True.",
    )

    include_references: Optional[bool] = Field(
        default=True,
        description="If True, includes reference list in responses. Affects /query and /query/stream endpoints. /query/data returns agent-focused structured data and ignores this flag.",
    )

    query_ref: list[str] = Field(
        default_factory=list,
        description="Chunk IDs referenced by this query. When provided, their documents' QA pairs are searched and used as supplemental context.",
    )

    stream: Optional[bool] = Field(
        default=True,
        description="If True, enables streaming output for real-time responses. Only affects /query/stream endpoint.",
    )
    agent_search: bool = Field(
        default=False,
        description="If True, dispatches the combined history and query to the configured OpenCode RAG search agent and merges its submitted results into the final output.",
    )
    stream_agent_status: bool = Field(
        default=False,
        description="If True, emits intermediate agent_search status events on /query/stream. Defaults to False so only final result packets are streamed.",
    )

    @field_validator("query", mode="after")
    @classmethod
    def query_strip_after(
        cls, query: str | QueryInputPayload
    ) -> str | QueryInputPayload:
        if isinstance(query, str):
            query = query.strip()
            if len(query) < 3:
                raise ValueError("query must be at least 3 characters long")
        return query

    @field_validator("conversation_history", mode="after")
    @classmethod
    def conversation_history_role_check(
        cls, conversation_history: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        if conversation_history is None:
            return None
        for msg in conversation_history:
            if "role" not in msg:
                raise ValueError("Each message must have a 'role' key.")
            if not isinstance(msg["role"], str) or not msg["role"].strip():
                raise ValueError("Each message 'role' must be a non-empty string.")
            _validate_history_references(msg.get("references"))
        return conversation_history

    @field_validator("query_ref", mode="before")
    @classmethod
    def query_ref_normalize(cls, query_ref: Any) -> list[str]:
        if query_ref is None:
            return []
        values = query_ref if isinstance(query_ref, list) else [query_ref]
        normalized: list[str] = []
        for item in values:
            if isinstance(item, dict):
                item = item.get("chunk_id")
            chunk_id = str(item or "").strip()
            if chunk_id and chunk_id not in normalized:
                normalized.append(chunk_id)
        return normalized

    def to_query_params(self, is_stream: bool) -> "QueryParam":
        """Converts a QueryRequest instance into a QueryParam instance."""
        # Use Pydantic's `.model_dump(exclude_none=True)` to remove None values automatically
        # Exclude API-level parameters that don't belong in QueryParam
        request_data = self.model_dump(
            exclude_none=True,
            exclude={"query", "workspace", "agent_search", "stream_agent_status"},
        )

        if self.conversation_history is None:
            request_data["conversation_history"] = self.get_effective_history()

        # Ensure `mode` and `stream` are set explicitly
        param = QueryParam(**request_data)
        param.stream = is_stream
        return param

    def get_query_text(self) -> str:
        if isinstance(self.query, QueryInputPayload):
            return self.query.latest_query
        return self.query

    def get_effective_history(self) -> List[Dict[str, Any]]:
        if self.conversation_history is not None:
            return self.conversation_history
        if isinstance(self.query, QueryInputPayload):
            return self.query.history
        return []


class ReferenceItem(BaseModel):
    """A single reference item in query responses."""

    reference_id: str = Field(description="Unique reference identifier")
    chunk_id: str = Field(description="Referenced chunk identifier")
    file_path: str = Field(description="Path to the source file")
    file_id: Optional[str] = Field(
        default=None, description="File ID from document metadata when available"
    )
    page_id: Optional[int] = Field(
        default=None, description="Page index of the referenced chunk when available"
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Bounding box of the referenced chunk when available",
    )
    content_type: Optional[str] = Field(
        default=None,
        description="Referenced chunk content type when available",
    )
    content: str = Field(
        description="Referenced chunk content preview for text, or full content for image chunks"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Full image payload for image chunks when available",
    )
    image_text: Optional[str] = Field(
        default=None,
        description="Accompanying OCR text for image chunks when available",
    )


class QueryResponse(BaseModel):
    response: str = Field(
        description="The generated response",
    )
    references: Optional[List[ReferenceItem]] = Field(
        default=None,
        description="Reference list (Disabled when include_references=False, /query/data always includes references.)",
    )
    agent_search_result: Optional["AgentSearchResultPayload"] = Field(
        default=None,
        description="Final agent_search execution status and summary when enabled.",
    )


class QueryDataResponse(BaseModel):
    status: str = Field(description="Query execution status")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(
        description="Agent-focused retrieval data containing entities, relationships, and chunks"
    )
    metadata: Dict[str, Any] = Field(
        description="Query metadata including mode, keywords, and processing information"
    )
    agent_search_result: Optional["AgentSearchResultPayload"] = Field(
        default=None,
        description="Final agent_search execution status and summary when enabled.",
    )


class StreamChunkResponse(BaseModel):
    """Response model for streaming chunks in NDJSON format"""

    references: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Reference list (only in first chunk when include_references=True)",
    )
    response: Optional[str] = Field(
        default=None, description="Response content chunk or complete response"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing fails"
    )
    agent_status: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Runtime status emitted by OpenCode agent_search while the query is in progress.",
    )
    agent_search_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Final agent_search execution status and summary when enabled.",
    )


class AgentSearchResultPayload(BaseModel):
    agent_search_id: str = Field(description="Generated id used to correlate the OpenCode search and LightRAG submit cache.")
    workspace: str = Field(description="Resolved workspace used for the agent search.")
    retrieval_target: str = Field(description="Combined history and latest query sent to the OpenCode agent.")
    status: Literal["completed", "missing_submit", "failed"] = Field(
        description="Final status of the OpenCode agent_search pipeline."
    )
    submitted: bool = Field(description="Whether a matching /api/agent/submit payload was found.")
    entity_count: int = Field(default=0, description="Number of submitted entities merged into the RAG retrieval context.")
    chunk_count: int = Field(default=0, description="Number of submitted chunks merged into the RAG retrieval context.")
    prior_rag_used: bool = Field(
        default=False,
        description="Whether a server-side prior RAG data pass was summarized and injected into the agent prompt.",
    )
    prior_rag_entity_count: int = Field(
        default=0,
        description="Number of entity items included in the prior RAG summary passed to the agent.",
    )
    prior_rag_relation_count: int = Field(
        default=0,
        description="Number of relationship items included in the prior RAG summary passed to the agent.",
    )
    prior_rag_chunk_count: int = Field(
        default=0,
        description="Number of chunk items included in the prior RAG summary passed to the agent.",
    )
    prior_rag_doc_qa_count: int = Field(
        default=0,
        description="Number of document QA items included in the prior RAG summary passed to the agent.",
    )
    opencode_session_id: str | None = Field(
        default=None,
        description="OpenCode session id used for the search run.",
    )
    opencode_output: str | None = Field(
        default=None,
        description="Final text output returned by OpenCode for this search run.",
    )
    error: str | None = Field(
        default=None,
        description="Failure reason when agent_search did not complete successfully.",
    )


@dataclass(slots=True)
class AgentSearchMergeBundle:
    public_result: AgentSearchResultPayload
    search_entities: list[dict[str, Any]]
    search_chunks: list[dict[str, Any]]
    cache_signature: str = ""


QueryResponse.model_rebuild()
QueryDataResponse.model_rebuild()


def _enrich_references_with_chunk_preview(
    references: List[Dict[str, Any]], chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Attach chunk content to references using reference_id mapping."""
    if not references:
        return references

    ref_id_to_content: Dict[str, str] = {}
    ref_id_to_content_type: Dict[str, str] = {}
    ref_id_to_image_base64: Dict[str, str] = {}
    ref_id_to_image_text: Dict[str, str] = {}
    for chunk in chunks:
        ref_id = str(chunk.get("reference_id", "")).strip()
        content = chunk.get("content", "")
        if ref_id and ref_id not in ref_id_to_content:
            content_type = str(chunk.get("content_type", "")).strip()
            resolved_image_base64, resolved_image_text = get_chunk_image_fields(chunk)
            ref_id_to_content[ref_id] = (
                str(resolved_image_base64 or content)
                if is_image_content_type(content_type)
                else get_content_summary(str(content))
            )
            if content_type:
                ref_id_to_content_type[ref_id] = content_type
            if resolved_image_base64 is not None:
                ref_id_to_image_base64[ref_id] = resolved_image_base64
            if resolved_image_text is not None:
                ref_id_to_image_text[ref_id] = resolved_image_text

    enriched_references: List[Dict[str, Any]] = []
    for ref in references:
        ref_copy = ref.copy()
        ref_id = str(ref.get("reference_id", "")).strip()
        ref_copy["content"] = ref_id_to_content.get(ref_id, "")
        if ref_id in ref_id_to_content_type:
            ref_copy["content_type"] = ref_id_to_content_type[ref_id]
        if ref_id in ref_id_to_image_base64:
            ref_copy["image_base64"] = ref_id_to_image_base64[ref_id]
        if ref_id in ref_id_to_image_text:
            ref_copy["image_text"] = ref_id_to_image_text[ref_id]
        enriched_references.append(ref_copy)

    return enriched_references


def _normalize_chunk_level_references(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize references to chunk-level shape for API responses."""
    references = data.get("references", []) or []
    chunks = data.get("chunks", []) or []

    chunk_metadata_by_chunk_id: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        chunk_metadata_by_chunk_id[chunk_id] = {
            "content_type": chunk.get("content_type"),
            "page_id": chunk.get("page_id"),
            "bbox": chunk.get("bbox"),
            "image_base64": chunk.get("image_base64"),
            "image_text": chunk.get("image_text"),
        }

    if references and all(
        isinstance(ref, dict) and str(ref.get("chunk_id", "")).strip()
        for ref in references
    ):
        normalized_references: List[Dict[str, Any]] = []
        for ref in references:
            ref_copy = ref.copy()
            chunk_meta = chunk_metadata_by_chunk_id.get(
                str(ref_copy.get("chunk_id", "")).strip()
            )
            if chunk_meta:
                if (
                    ref_copy.get("content_type") is None
                    and chunk_meta.get("content_type") is not None
                ):
                    ref_copy["content_type"] = chunk_meta["content_type"]
                if ref_copy.get("page_id") is None and chunk_meta.get("page_id") is not None:
                    ref_copy["page_id"] = chunk_meta["page_id"]
                if ref_copy.get("bbox") is None and chunk_meta.get("bbox") is not None:
                    ref_copy["bbox"] = chunk_meta["bbox"]
                if (
                    ref_copy.get("image_base64") is None
                    and chunk_meta.get("image_base64") is not None
                ):
                    ref_copy["image_base64"] = chunk_meta["image_base64"]
                if (
                    ref_copy.get("image_text") is None
                    and chunk_meta.get("image_text") is not None
                ):
                    ref_copy["image_text"] = chunk_meta["image_text"]
            normalized_references.append(ref_copy)
        return normalized_references

    file_id_by_reference_id: Dict[str, Any] = {}
    for ref in references:
        if isinstance(ref, dict):
            ref_id = str(ref.get("reference_id", "")).strip()
            if ref_id:
                file_id_by_reference_id[ref_id] = ref.get("file_id")

    normalized: List[Dict[str, Any]] = []
    seen_reference_ids: set[str] = set()
    for chunk in chunks:
        ref_id = str(chunk.get("reference_id", "")).strip()
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not ref_id or not chunk_id or ref_id in seen_reference_ids:
            continue

        normalized.append(
            {
                "reference_id": ref_id,
                "chunk_id": chunk_id,
                "file_path": chunk.get("file_path", "unknown_source"),
                "file_id": file_id_by_reference_id.get(ref_id),
                "content_type": chunk.get("content_type"),
                "page_id": chunk.get("page_id"),
                "bbox": chunk.get("bbox"),
                "image_base64": chunk.get("image_base64"),
                "image_text": chunk.get("image_text"),
            }
        )
        seen_reference_ids.add(ref_id)

    if normalized:
        return normalized

    # Final fallback for old cached structures with no chunk_id.
    fallback: List[Dict[str, Any]] = []
    for ref in references:
        if not isinstance(ref, dict):
            continue
        ref_copy = ref.copy()
        ref_copy.setdefault("chunk_id", "")
        fallback.append(ref_copy)
    return fallback


def _format_query_data_for_agent(response: Dict[str, Any]) -> Dict[str, Any]:
    """Return the /query/data payload in the compact schema expected by agents."""
    formatted = response.copy()
    data = response.get("data", {})
    if not isinstance(data, dict):
        formatted["data"] = {"entities": [], "relationships": [], "chunks": []}
        return formatted

    entities: list[dict[str, Any]] = []
    for item in data.get("entities", []) or []:
        if not isinstance(item, dict):
            continue
        entity_id = str(
            item.get("id")
            or item.get("entity_name")
            or item.get("entity")
            or item.get("name")
            or ""
        ).strip()
        if not entity_id:
            continue
        entities.append(
            {
                "id": entity_id,
                "description": item.get("description", ""),
            }
        )

    relationships: list[dict[str, Any]] = []
    for item in data.get("relationships", data.get("relations", [])) or []:
        if not isinstance(item, dict):
            continue
        relation_id = str(item.get("id") or item.get("name") or "").strip()
        if not relation_id:
            src_id = str(item.get("src_id") or item.get("entity1") or "").strip()
            tgt_id = str(item.get("tgt_id") or item.get("entity2") or "").strip()
            relation_id = (
                f"{src_id} -> {tgt_id}" if src_id or tgt_id else ""
            ).strip()
        if not relation_id:
            continue
        relationships.append(
            {
                "id": relation_id,
                "description": item.get("description", ""),
            }
        )

    chunks: list[dict[str, Any]] = []
    for item in data.get("chunks", []) or []:
        if not isinstance(item, dict):
            continue
        resolved_image_base64, resolved_image_text = get_chunk_image_fields(item)
        chunks.append(
            {
                "chunk_id": item.get("chunk_id", ""),
                "full_doc_id": item.get("full_doc_id"),
                "image_text": resolved_image_text,
                "content_type": item.get("content_type"),
            }
        )

    formatted["data"] = {
        "entities": entities,
        "relationships": relationships,
        "chunks": chunks,
    }
    return formatted


def _build_agent_search_id() -> str:
    return f"agent-search-{uuid4().hex}"


def _get_agent_search_timeout_seconds() -> float:
    configured = float(getattr(global_args, "opencode_timeout", 120.0) or 120.0)
    return max(configured + 5.0, 10.0)


def _get_agent_submit_wait_timeout_seconds() -> float:
    configured = float(getattr(global_args, "opencode_timeout", 120.0) or 120.0)
    return min(max(configured / 6.0, 1.0), 10.0)


def _update_agent_search_run_state(**values: Any) -> None:
    state = _agent_search_run_state.get()
    if state is not None:
        state.update(values)


def _build_retrieval_target(request: QueryRequest) -> str:
    sections: list[str] = []
    history = request.get_effective_history()
    if history:
        history_entries: list[dict[str, Any]] = []
        for message in history:
            role = str(message.get("role", "user") or "user").strip() or "user"
            content = str(message.get("content", "") or "").strip()
            if not content:
                continue
            history_entry: dict[str, Any] = {"role": role, "content": content}
            references = _stringify_history_reference_items(message.get("references"))
            if references:
                history_entry["references"] = references
            history_entries.append(history_entry)
        if history_entries:
            sections.append("[history]")
            sections.append(
                json.dumps(history_entries, ensure_ascii=False, indent=2)
            )

    query_text = request.get_query_text().strip()
    sections.extend(["[latest_query]", query_text])
    return "\n".join(sections)


def _build_latest_query_retrieval_target(request: QueryRequest) -> str:
    return "\n".join(["[latest_query]", request.get_query_text().strip()])


def _summarize_prior_rag_data(
    result: dict[str, Any] | None,
    *,
    max_entities: int = 5,
    max_relations: int = 5,
    max_chunks: int = 5,
) -> tuple[str | None, dict[str, int]]:
    empty_counts = {
        "entities": 0,
        "relationships": 0,
        "chunks": 0,
        "doc_qa": 0,
    }
    if not isinstance(result, dict):
        return None, empty_counts

    data = result.get("data")
    if not isinstance(data, dict):
        return None, empty_counts

    entities = [
        item for item in (data.get("entities") or []) if isinstance(item, dict)
    ][:max_entities]
    relationships = [
        item for item in (data.get("relationships") or []) if isinstance(item, dict)
    ][:max_relations]
    chunks = [item for item in (data.get("chunks") or []) if isinstance(item, dict)][
        :max_chunks
    ]
    doc_qa = [item for item in (data.get("doc_qa") or []) if isinstance(item, dict)][
        :5
    ]

    if not entities and not relationships and not chunks and not doc_qa:
        return None, empty_counts

    lines: list[str] = []
    if entities:
        lines.append("[entities]")
        for item in entities:
            entity_name = str(item.get("entity_name", "") or "").strip()
            entity_type = str(item.get("entity_type", "") or "").strip()
            description = get_content_summary(
                str(item.get("description", "") or "").strip()
            )
            source_id = str(item.get("source_id", "") or "").strip()
            parts = [entity_name]
            if entity_type:
                parts.append(f"type={entity_type}")
            if description:
                parts.append(f"description={description}")
            if source_id:
                parts.append(f"source_id={source_id}")
            lines.append(" - " + " | ".join(part for part in parts if part))

    if relationships:
        lines.append("[relationships]")
        for item in relationships:
            src_id = str(item.get("src_id", "") or "").strip()
            tgt_id = str(item.get("tgt_id", "") or "").strip()
            description = get_content_summary(
                str(item.get("description", "") or "").strip()
            )
            keywords = str(item.get("keywords", "") or "").strip()
            parts = [f"{src_id} -> {tgt_id}".strip()]
            if description:
                parts.append(f"description={description}")
            if keywords:
                parts.append(f"keywords={keywords}")
            lines.append(" - " + " | ".join(part for part in parts if part))

    if chunks:
        lines.append("[chunks]")
        for item in chunks:
            chunk_id = str(item.get("chunk_id", "") or "").strip()
            file_path = str(item.get("file_path", "") or "").strip()
            content = get_content_summary(str(item.get("content", "") or "").strip())
            parts = [chunk_id]
            if file_path:
                parts.append(f"file_path={file_path}")
            if content:
                parts.append(f"content={content}")
            lines.append(" - " + " | ".join(part for part in parts if part))

    if doc_qa:
        lines.append("[doc_qa]")
        for item in doc_qa:
            belong_chunk = str(item.get("belong_chunk", "") or "").strip()
            question = str(item.get("question", "") or "").strip()
            answer = str(item.get("answer", "") or "").strip()
            parts = []
            if belong_chunk:
                parts.append(f"belong_chunk={belong_chunk}")
            if question:
                parts.append(f"question={question}")
            if answer:
                parts.append(f"answer={answer}")
            if parts:
                lines.append(" - " + " | ".join(parts))

    summary = "\n".join(lines).strip()
    if not summary:
        return None, empty_counts

    return summary, {
        "entities": len(entities),
        "relationships": len(relationships),
        "chunks": len(chunks),
        "doc_qa": len(doc_qa),
    }


async def _build_prior_rag_context(
    rag: Any,
    request: QueryRequest,
    param: QueryParam,
    *,
    ignore_history: bool = False,
) -> tuple[str | None, dict[str, int]]:
    prior_param = QueryParam(**param.__dict__)
    prior_param.stream = False
    if ignore_history:
        prior_param.conversation_history = []
    query_payload: str | QueryInputPayload = (
        request.get_query_text() if ignore_history else request.query
    )
    try:
        result = await rag.aquery_data(
            query_payload,
            param=prior_param,
            agent_context=None,
        )
    except Exception as exc:
        logger.warning("Prior RAG context generation failed before agent_search: %s", exc)
        return None, {"entities": 0, "relationships": 0, "chunks": 0, "doc_qa": 0}

    return _summarize_prior_rag_data(result)


async def _emit_agent_status(
    queue: asyncio.Queue[dict[str, Any]] | None,
    status: OpencodeStatusEvent,
) -> None:
    if queue is None:
        return
    await queue.put(
        {
            "agent_search_id": status.agent_search_id,
            "phase": status.phase,
            "message": status.message,
            "event_type": status.event_type,
            "opencode_session_id": status.session_id,
        }
    )


def _dedupe_records(
    items: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        key = ""
        for field in key_fields:
            value = str(item.get(field, "") or "").strip()
            if value:
                key = f"{field}:{value}"
                break
        if not key:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _split_source_id(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [item for item in value.split("<SEP>") if item]


async def _wait_for_agent_submit_payload(
    rag: Any,
    agent_search_id: str,
    timeout_seconds: float = 5.0,
    interval_seconds: float = 0.5,
) -> tuple[dict[str, Any] | None, str]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_workspace = ""

    while True:
        payload, workspace = await _get_agent_submit_payload(rag, agent_search_id)
        last_workspace = workspace
        if payload is not None:
            return payload, workspace
        if asyncio.get_running_loop().time() >= deadline:
            return None, last_workspace
        await asyncio.sleep(interval_seconds)


def _build_agent_cache_signature(
    entity_ids: list[str],
    chunk_ids: list[str],
) -> str:
    return json.dumps(
        {
            "entity_ids": entity_ids,
            "chunk_ids": chunk_ids,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


async def _load_agent_entities(entity_ids: list[str], rag: Any) -> list[dict[str, Any]]:
    if not entity_ids:
        return []

    graph_storage = _resolve_graph_storage(rag)
    nodes = await graph_storage.get_nodes_batch(entity_ids)
    records: list[dict[str, Any]] = []
    for entity_id in entity_ids:
        node = nodes.get(entity_id)
        if not isinstance(node, dict):
            continue
        records.append(
            {
                "entity_name": node.get("entity_id") or entity_id,
                "file_path": node.get("file_path", "unknown_source"),
                "description": node.get("description"),
                "source_id": node.get("source_id"),
                "entity_type": node.get("entity_type"),
                "created_at": node.get("created_at"),
            }
        )
    return records


async def _load_agent_chunks(chunk_ids: list[str], rag: Any) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []

    records = await rag.text_chunks.get_by_ids(chunk_ids)
    if not isinstance(records, list):
        return []

    record_map: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        chunk_id = str(record.get("id") or record.get("chunk_id") or "").strip()
        if chunk_id:
            record_map[chunk_id] = record

    chunk_payloads: list[dict[str, Any]] = []
    for chunk_id in chunk_ids:
        record = record_map.get(chunk_id)
        if record is None:
            continue
        resolved_image_base64, resolved_image_text = get_chunk_image_fields(record)
        chunk_content = record.get("content", "")
        if is_image_content_type(record.get("content_type")) and resolved_image_base64:
            chunk_content = resolved_image_base64
        chunk_payload = {
            "chunk_id": chunk_id,
            "full_doc_id": record.get("full_doc_id"),
            "content": str(chunk_content or ""),
            "file_path": record.get("file_path", "unknown_source"),
            "content_type": record.get("content_type"),
            "page_id": record.get("page_id"),
            "bbox": record.get("bbox"),
        }
        if resolved_image_base64 is not None:
            chunk_payload["image_base64"] = resolved_image_base64
        if resolved_image_text is not None:
            chunk_payload["image_text"] = resolved_image_text
        chunk_payloads.append(chunk_payload)
    return chunk_payloads


def _build_agent_context(
    bundle: AgentSearchMergeBundle | None,
) -> dict[str, Any] | None:
    if bundle is None or not bundle.public_result.submitted:
        return None

    if not bundle.search_entities and not bundle.search_chunks:
        return None

    return {
        "signature": bundle.cache_signature,
        "entities": bundle.search_entities,
        "chunks": bundle.search_chunks,
    }


def _normalize_agent_submit_ids(submit_payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    entity_ids = [
        str(item).strip()
        for item in submit_payload.get("entity_ids", [])
        if str(item).strip()
    ]
    chunk_ids = [
        str(item).strip()
        for item in submit_payload.get("chunk_ids", [])
        if str(item).strip()
    ]
    return entity_ids, chunk_ids


async def _build_agent_bundle_from_submit(
    rag: Any,
    *,
    agent_search_id: str,
    workspace: str,
    retrieval_target: str,
    submit_payload: dict[str, Any],
    resolved_workspace: str,
    status: Literal["completed", "missing_submit", "failed"],
    prior_rag_context: str | None,
    prior_rag_counts: dict[str, int],
    opencode_session_id: str | None = None,
    opencode_output: str | None = None,
    error: str | None = None,
) -> AgentSearchMergeBundle:
    entity_ids, chunk_ids = _normalize_agent_submit_ids(submit_payload)
    search_entities = await _load_agent_entities(entity_ids, rag)
    search_chunks = await _load_agent_chunks(chunk_ids, rag)
    cache_signature = _build_agent_cache_signature(entity_ids, chunk_ids)

    return AgentSearchMergeBundle(
        public_result=AgentSearchResultPayload(
            agent_search_id=agent_search_id,
            workspace=resolved_workspace or workspace,
            retrieval_target=retrieval_target,
            status=status,
            submitted=True,
            entity_count=len(search_entities),
            chunk_count=len(search_chunks),
            prior_rag_used=bool(prior_rag_context),
            prior_rag_entity_count=prior_rag_counts["entities"],
            prior_rag_relation_count=prior_rag_counts["relationships"],
            prior_rag_chunk_count=prior_rag_counts["chunks"],
            prior_rag_doc_qa_count=prior_rag_counts["doc_qa"],
            opencode_session_id=opencode_session_id,
            opencode_output=opencode_output,
            error=error,
        ),
        search_entities=search_entities,
        search_chunks=search_chunks,
        cache_signature=cache_signature,
    )


def _build_agent_bundle_without_submit(
    *,
    agent_search_id: str,
    workspace: str,
    retrieval_target: str,
    resolved_workspace: str = "",
    status: Literal["completed", "missing_submit", "failed"],
    prior_rag_context: str | None,
    prior_rag_counts: dict[str, int],
    opencode_session_id: str | None = None,
    opencode_output: str | None = None,
    error: str | None = None,
) -> AgentSearchMergeBundle:
    return AgentSearchMergeBundle(
        public_result=AgentSearchResultPayload(
            agent_search_id=agent_search_id,
            workspace=resolved_workspace or workspace,
            retrieval_target=retrieval_target,
            status=status,
            submitted=False,
            prior_rag_used=bool(prior_rag_context),
            prior_rag_entity_count=prior_rag_counts["entities"],
            prior_rag_relation_count=prior_rag_counts["relationships"],
            prior_rag_chunk_count=prior_rag_counts["chunks"],
            prior_rag_doc_qa_count=prior_rag_counts["doc_qa"],
            opencode_session_id=opencode_session_id,
            opencode_output=opencode_output,
            error=error,
        ),
        search_entities=[],
        search_chunks=[],
    )


async def _run_agent_search_pipeline(
    rag: Any,
    request: QueryRequest,
    workspace: str,
    param: QueryParam,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
    ignore_history: bool = False,
) -> AgentSearchMergeBundle | None:
    if not request.agent_search:
        return None

    agent_search_id = _build_agent_search_id()
    retrieval_target = (
        _build_latest_query_retrieval_target(request)
        if ignore_history
        else _build_retrieval_target(request)
    )
    prior_rag_context: str | None = None
    prior_rag_counts = {"entities": 0, "relationships": 0, "chunks": 0, "doc_qa": 0}
    _update_agent_search_run_state(
        agent_search_id=agent_search_id,
        retrieval_target=retrieval_target,
        prior_rag_context=prior_rag_context,
        prior_rag_counts=prior_rag_counts,
    )

    if status_queue is not None:
        await status_queue.put(
            {
                "agent_search_id": agent_search_id,
                "phase": "pre_rag",
                "message": "building prior rag context for agent",
                "event_type": "local.pre_rag_started",
                "opencode_session_id": None,
            }
        )

    prior_rag_context, prior_rag_counts = await _build_prior_rag_context(
        rag,
        request,
        param,
        ignore_history=ignore_history,
    )
    _update_agent_search_run_state(
        prior_rag_context=prior_rag_context,
        prior_rag_counts=prior_rag_counts,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Sending agent_search payload: %s",
            json.dumps(
                {
                    "agent_search_id": agent_search_id,
                    "workspace": workspace,
                    "retrieval_target": retrieval_target,
                    "prior_rag_context": prior_rag_context,
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    if not is_opencode_enabled():
        return AgentSearchMergeBundle(
            public_result=AgentSearchResultPayload(
                agent_search_id=agent_search_id,
                workspace=workspace,
                retrieval_target=retrieval_target,
                status="failed",
                submitted=False,
                prior_rag_used=bool(prior_rag_context),
                prior_rag_entity_count=prior_rag_counts["entities"],
                prior_rag_relation_count=prior_rag_counts["relationships"],
                prior_rag_chunk_count=prior_rag_counts["chunks"],
                prior_rag_doc_qa_count=prior_rag_counts["doc_qa"],
                error="OpenCode integration is not configured",
            ),
            search_entities=[],
            search_chunks=[],
        )

    opencode_result = None
    opencode_error: str | None = None
    try:
        opencode_result = await run_agent_search(
            agent_search_id=agent_search_id,
            workspace=workspace,
            retrieval_target=retrieval_target,
            prior_rag_context=prior_rag_context,
            callback=(
                None
                if status_queue is None
                else lambda status: _emit_agent_status(status_queue, status)
            ),
        )
    except Exception as exc:
        logger.error("OpenCode agent search raised unexpectedly: %s", exc)
        opencode_error = str(exc)

    submit_payload, resolved_workspace = await _wait_for_agent_submit_payload(
        rag,
        agent_search_id,
        timeout_seconds=_get_agent_submit_wait_timeout_seconds(),
    )

    opencode_ok = bool(getattr(opencode_result, "ok", False))
    opencode_session_id = getattr(opencode_result, "session_id", None)
    opencode_output = getattr(opencode_result, "final_output", None)
    if opencode_error is None:
        opencode_error = getattr(opencode_result, "error", None)

    if submit_payload is not None:
        return await _build_agent_bundle_from_submit(
            rag,
            agent_search_id=agent_search_id,
            workspace=workspace,
            retrieval_target=retrieval_target,
            submit_payload=submit_payload,
            resolved_workspace=resolved_workspace,
            status="completed" if opencode_ok else "failed",
            prior_rag_context=prior_rag_context,
            prior_rag_counts=prior_rag_counts,
            opencode_session_id=opencode_session_id,
            opencode_output=opencode_output,
            error=None if opencode_ok else opencode_error,
        )

    if submit_payload is None:
        return _build_agent_bundle_without_submit(
            agent_search_id=agent_search_id,
            workspace=workspace,
            retrieval_target=retrieval_target,
            resolved_workspace=resolved_workspace,
            status="missing_submit" if opencode_ok else "failed",
            prior_rag_context=prior_rag_context,
            prior_rag_counts=prior_rag_counts,
            opencode_session_id=opencode_session_id,
            opencode_output=opencode_output,
            error=(
                "OpenCode finished but no matching /api/agent/submit payload was found"
                if opencode_ok
                else opencode_error
            ),
        )


async def _run_agent_search_pipeline_guarded(
    rag: Any,
    request: QueryRequest,
    workspace: str,
    param: QueryParam,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
    ignore_history: bool = False,
) -> AgentSearchMergeBundle | None:
    state: dict[str, Any] = {}
    state_token = _agent_search_run_state.set(state)
    try:
        return await asyncio.wait_for(
            _run_agent_search_pipeline(
                rag,
                request,
                workspace=workspace,
                param=param,
                status_queue=status_queue,
                ignore_history=ignore_history,
            ),
            timeout=_get_agent_search_timeout_seconds(),
        )
    except asyncio.TimeoutError:
        agent_search_id = str(state.get("agent_search_id") or _build_agent_search_id())
        retrieval_target = str(
            state.get("retrieval_target")
            or (
                _build_latest_query_retrieval_target(request)
                if ignore_history
                else _build_retrieval_target(request)
            )
        )
        prior_rag_context = state.get("prior_rag_context")
        prior_rag_counts = state.get(
            "prior_rag_counts",
            {"entities": 0, "relationships": 0, "chunks": 0, "doc_qa": 0},
        )
        submit_payload, resolved_workspace = await _wait_for_agent_submit_payload(
            rag,
            agent_search_id,
            timeout_seconds=0.0,
        )
        if status_queue is not None:
            await status_queue.put(
                {
                    "agent_search_id": agent_search_id,
                    "phase": "failed",
                    "message": "agent_search timed out",
                    "event_type": "timeout",
                    "opencode_session_id": None,
                }
            )

        if submit_payload is not None:
            return await _build_agent_bundle_from_submit(
                rag,
                agent_search_id=agent_search_id,
                workspace=workspace,
                retrieval_target=retrieval_target,
                status="failed",
                submit_payload=submit_payload,
                resolved_workspace=resolved_workspace,
                prior_rag_context=(
                    prior_rag_context if isinstance(prior_rag_context, str) else None
                ),
                prior_rag_counts=prior_rag_counts,
                error="agent_search timed out",
            )

        return _build_agent_bundle_without_submit(
            agent_search_id=agent_search_id,
            workspace=workspace,
            retrieval_target=retrieval_target,
            resolved_workspace=resolved_workspace,
            status="failed",
            prior_rag_context=(
                prior_rag_context if isinstance(prior_rag_context, str) else None
            ),
            prior_rag_counts=prior_rag_counts,
            error="agent_search timed out",
        )
    except Exception as exc:
        agent_search_id = str(state.get("agent_search_id") or _build_agent_search_id())
        retrieval_target = str(
            state.get("retrieval_target")
            or (
                _build_latest_query_retrieval_target(request)
                if ignore_history
                else _build_retrieval_target(request)
            )
        )
        prior_rag_context = state.get("prior_rag_context")
        prior_rag_counts = state.get(
            "prior_rag_counts",
            {"entities": 0, "relationships": 0, "chunks": 0, "doc_qa": 0},
        )
        submit_payload, resolved_workspace = await _wait_for_agent_submit_payload(
            rag,
            agent_search_id,
            timeout_seconds=0.0,
        )
        if submit_payload is not None:
            return await _build_agent_bundle_from_submit(
                rag,
                agent_search_id=agent_search_id,
                workspace=workspace,
                retrieval_target=retrieval_target,
                submit_payload=submit_payload,
                resolved_workspace=resolved_workspace,
                status="failed",
                prior_rag_context=(
                    prior_rag_context if isinstance(prior_rag_context, str) else None
                ),
                prior_rag_counts=prior_rag_counts,
                error=str(exc),
            )
        return _build_agent_bundle_without_submit(
            agent_search_id=agent_search_id,
            workspace=workspace,
            retrieval_target=retrieval_target,
            resolved_workspace=resolved_workspace,
            status="failed",
            prior_rag_context=(
                prior_rag_context if isinstance(prior_rag_context, str) else None
            ),
            prior_rag_counts=prior_rag_counts,
            error=str(exc),
        )
    finally:
        _agent_search_run_state.reset(state_token)


async def _run_query_with_agent_context(
    rag: Any,
    request: QueryRequest,
    workspace: str,
    *,
    param: QueryParam,
    data_only: bool = False,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], AgentSearchMergeBundle | None]:
    agent_bundle = (
        await _run_agent_search_pipeline_guarded(
            rag,
            request,
            workspace=workspace,
            param=param,
            status_queue=status_queue,
            ignore_history=data_only,
        )
        if request.agent_search
        else None
    )
    agent_context = _build_agent_context(agent_bundle)

    if data_only:
        result = await rag.aquery_data(
            request.get_query_text(),
            param=param,
            agent_context=agent_context,
        )
    else:
        result = await rag.aquery_llm(
            request.query,
            param=param,
            agent_context=agent_context,
        )

    return result, agent_bundle


def create_query_routes(
    rag_by_workspace: dict[str, Any],
    api_key: Optional[str] = None,
    top_k: int = 60,
    workspace: str = "",
    workspace_aliases: Mapping[str, str] | None = None,
):
    if not rag_by_workspace:
        raise ValueError("rag_by_workspace cannot be empty")

    default_workspace = workspace.strip() or next(iter(rag_by_workspace.keys()))
    if default_workspace not in rag_by_workspace:
        raise ValueError(
            f"Default workspace '{default_workspace}' not found in rag_by_workspace"
        )

    workspace_context: ContextVar[str] = ContextVar(
        "query_workspace", default=default_workspace
    )
    rag = WorkspaceObjectProxy(
        rag_by_workspace, default_workspace=default_workspace, workspace_context=workspace_context
    )
    workspace_scope = create_workspace_scope_dependency(
        rag_by_workspace,
        workspace_context=workspace_context,
        default_workspace=default_workspace,
        workspace_aliases=workspace_aliases,
    )
    combined_auth = get_combined_auth_dependency(api_key)
    router = APIRouter(tags=["query"], dependencies=[Depends(workspace_scope)])

    @router.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Successful RAG query response",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "response": {
                                    "type": "string",
                                    "description": "The generated response from the RAG system",
                                },
                                "references": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "reference_id": {"type": "string"},
                                            "chunk_id": {"type": "string"},
                                            "file_path": {"type": "string"},
                                            "file_id": {"type": "string"},
                                            "page_id": {"type": "integer"},
                                            "bbox": {
                                                "type": "array",
                                                "items": {"type": "number"},
                                            },
                                            "content": {
                                                "type": "string",
                                                "description": "Preview text from the referenced chunk",
                                            },
                                        },
                                    },
                                    "description": "Reference list (only included when include_references=True)",
                                },
                            },
                            "required": ["response"],
                        },
                        "examples": {
                            "with_references": {
                                "summary": "Response with references",
                                "description": "Example response when include_references=True",
                                "value": {
                                    "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
                                    "references": [
                                        {
                                            "reference_id": "1",
                                            "chunk_id": "chunk-1",
                                            "file_path": "/documents/ai_overview.pdf",
                                            "page_id": 0,
                                            "bbox": [10.0, 20.0, 120.0, 60.0],
                                            "content": "Artificial Intelligence (AI) represents a transformative field in computer science focused on creating systems that can perform tasks requiring human-like intelligence....",
                                        },
                                        {
                                            "reference_id": "2",
                                            "chunk_id": "chunk-2",
                                            "file_path": "/documents/machine_learning.txt",
                                            "page_id": 1,
                                            "bbox": [15.0, 80.0, 150.0, 120.0],
                                            "content": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed....",
                                        },
                                    ],
                                },
                            },
                            "with_reference_previews": {
                                "summary": "Response with reference previews",
                                "description": "Example response when include_references=True. Each reference always includes a preview string from the retrieved chunk.",
                                "value": {
                                    "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
                                    "references": [
                                        {
                                            "reference_id": "1",
                                            "chunk_id": "chunk-1",
                                            "file_path": "/documents/ai_overview.pdf",
                                            "page_id": 0,
                                            "bbox": [10.0, 20.0, 120.0, 60.0],
                                            "content": "Artificial Intelligence (AI) represents a transformative field in computer science focused on creating systems that can perform tasks requiring human-like intelligence. These tasks include learning from experience, understanding natural language, recognizing patterns, and making decisions....",
                                        },
                                        {
                                            "reference_id": "2",
                                            "chunk_id": "chunk-2",
                                            "file_path": "/documents/machine_learning.txt",
                                            "page_id": 1,
                                            "bbox": [15.0, 80.0, 150.0, 120.0],
                                            "content": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It focuses on the development of algorithms that can access data and use it to learn for themselves.",
                                        },
                                    ],
                                },
                            },
                            "without_references": {
                                "summary": "Response without references",
                                "description": "Example response when include_references=False",
                                "value": {
                                    "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."
                                },
                            },
                            "different_modes": {
                                "summary": "Different query modes",
                                "description": "Examples of responses from different query modes",
                                "value": {
                                    "local_mode": "Focuses on specific entities and their relationships",
                                    "global_mode": "Provides broader context from relationship patterns",
                                    "hybrid_mode": "Combines local and global approaches",
                                    "naive_mode": "Simple vector similarity search",
                                    "mix_mode": "Integrates knowledge graph and vector retrieval",
                                },
                            },
                        },
                    }
                },
            },
            400: {
                "description": "Bad Request - Invalid input parameters",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Query text must be at least 3 characters long"
                        },
                    }
                },
            },
            500: {
                "description": "Internal Server Error - Query processing failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Failed to process query: LLM service unavailable"
                        },
                    }
                },
            },
        },
    )
    async def query_text(request: QueryRequest):
        """
        Comprehensive RAG query endpoint with non-streaming response. Parameter "stream" is ignored.

        This endpoint performs Retrieval-Augmented Generation (RAG) queries using various modes
        to provide intelligent responses based on your knowledge base.

        **Query Modes:**
        - **local**: Focuses on specific entities and their direct relationships
        - **global**: Analyzes broader patterns and relationships across the knowledge graph
        - **hybrid**: Combines local and global approaches for comprehensive results
        - **naive**: Simple vector similarity search without knowledge graph
        - **mix**: Integrates knowledge graph retrieval with vector search (recommended)
        - **bypass**: Direct LLM query without knowledge retrieval

        conversation_history parameteris sent to LLM only, does not affect retrieval results.

        **Usage Examples:**

        Basic query:
        ```json
        {
            "query": "What is machine learning?",
            "mode": "mix"
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        Advanced query with references:
        ```json
        {
            "query": "Explain neural networks",
            "mode": "hybrid",
            "include_references": true,
            "response_type": "Multiple Paragraphs",
            "top_k": 10
        }
        ```

        Conversation with history:
        ```json
        {
            "query": "Can you give me more details?",
            "conversation_history": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence..."}
            ]
        }
        ```

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The question or prompt to process (min 3 characters)
                - **mode**: Query strategy - "mix" recommended for best results
                - **include_references**: Whether to include source citations
                - **response_type**: Format preference (e.g., "Multiple Paragraphs")
                - **top_k**: Number of top entities/relations to retrieve
                - **conversation_history**: Previous dialogue context
                - **max_total_tokens**: Token budget for the entire response

        Returns:
            QueryResponse: JSON response containing:
                - **response**: The generated answer to your query
                - **references**: Source citations (if include_references=True)

        Raises:
            HTTPException:
                - 400: Invalid input parameters (e.g., query too short)
                - 500: Internal processing error (e.g., LLM service unavailable)
        """
        try:
            _log_query_request("/query", request)
            current_rag = rag.resolve_current()
            current_workspace = str(
                getattr(getattr(current_rag, "text_chunks", None), "workspace", default_workspace)
                or default_workspace
            )
            param = request.to_query_params(
                False
            )  # Ensure stream=False for non-streaming endpoint
            # Force stream=False for /query endpoint regardless of include_references setting
            param.stream = False
            result, agent_bundle = await _run_query_with_agent_context(
                current_rag,
                request,
                current_workspace,
                param=param,
            )

            # Extract LLM response and references from unified result
            llm_response = result.get("llm_response", {})
            data = result.get("data", {})
            references = _normalize_chunk_level_references(data)

            # Get the non-streaming response content
            response_content = llm_response.get("content", "")
            if not response_content:
                response_content = "No relevant context found for the query."
            _log_query_response("/query", response_content)

            if request.include_references:
                references = _enrich_references_with_chunk_preview(
                    references, data.get("chunks", [])
                )

            # Return response with or without references based on request
            if request.include_references:
                return QueryResponse(
                    response=response_content,
                    references=references,
                    agent_search_result=(
                        agent_bundle.public_result if agent_bundle is not None else None
                    ),
                )
            else:
                return QueryResponse(
                    response=response_content,
                    references=None,
                    agent_search_result=(
                        agent_bundle.public_result if agent_bundle is not None else None
                    ),
                )
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/query/stream",
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Flexible RAG query response - format depends on stream parameter",
                "content": {
                    "application/x-ndjson": {
                        "schema": {
                            "type": "string",
                            "format": "ndjson",
                            "description": "Newline-delimited JSON (NDJSON) format used for both streaming and non-streaming responses. For streaming: multiple lines with separate JSON objects. For non-streaming: single line with complete JSON object.",
                            "example": '{"references": [{"reference_id": "1", "file_path": "/documents/ai.pdf"}]}\n{"response": "Artificial Intelligence is"}\n{"response": " a field of computer science"}\n{"response": " that focuses on creating intelligent machines."}',
                        },
                        "examples": {
                            "streaming_with_references": {
                                "summary": "Streaming mode with references (stream=true)",
                                "description": "Multiple NDJSON lines when stream=True and include_references=True. First line contains references, subsequent lines contain response chunks.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/ai_overview.pdf"}, {"reference_id": "2", "file_path": "/documents/ml_basics.txt"}]}\n{"response": "Artificial Intelligence (AI) is a branch of computer science"}\n{"response": " that aims to create intelligent machines capable of performing"}\n{"response": " tasks that typically require human intelligence, such as learning,"}\n{"response": " reasoning, and problem-solving."}',
                            },
                            "streaming_with_reference_previews": {
                                "summary": "Streaming mode with reference previews (stream=true)",
                                "description": "Multiple NDJSON lines when stream=True and include_references=True. The first line contains references with preview text, followed by response chunks.",
                                "value": "{\"references\": [{\"reference_id\": \"1\", \"file_path\": \"/documents/ai_overview.pdf\", \"content\": \"Artificial Intelligence (AI) represents a transformative field...\"}, {\"reference_id\": \"2\", \"file_path\": \"/documents/ml_basics.txt\", \"content\": \"Machine learning is a subset of AI that enables computers to learn...\"}]}\n{\"response\": \"Artificial Intelligence (AI) is a branch of computer science\"}\n{\"response\": \" that aims to create intelligent machines capable of performing\"}\n{\"response\": \" tasks that typically require human intelligence.\"}",
                            },
                            "streaming_without_references": {
                                "summary": "Streaming mode without references (stream=true)",
                                "description": "Multiple NDJSON lines when stream=True and include_references=False. Only response chunks are sent.",
                                "value": '{"response": "Machine learning is a subset of artificial intelligence"}\n{"response": " that enables computers to learn and improve from experience"}\n{"response": " without being explicitly programmed for every task."}',
                            },
                            "non_streaming_with_references": {
                                "summary": "Non-streaming mode with references (stream=false)",
                                "description": "Single NDJSON line when stream=False and include_references=True. Complete response with references in one message.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/neural_networks.pdf", "content": "Neural networks are computational models inspired by biological neural networks..."}], "response": "Neural networks are computational models inspired by biological neural networks that consist of interconnected nodes (neurons) organized in layers. They are fundamental to deep learning and can learn complex patterns from data through training processes."}',
                            },
                            "non_streaming_without_references": {
                                "summary": "Non-streaming mode without references (stream=false)",
                                "description": "Single NDJSON line when stream=False and include_references=False. Complete response only.",
                                "value": '{"response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence deep) to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition."}',
                            },
                            "error_response": {
                                "summary": "Error during streaming",
                                "description": "Error handling in NDJSON format when an error occurs during processing.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/ai.pdf", "content": "Artificial Intelligence is a field of computer science..."}]}\n{"response": "Artificial Intelligence is"}\n{"error": "LLM service temporarily unavailable"}',
                            },
                        },
                    }
                },
            },
            400: {
                "description": "Bad Request - Invalid input parameters",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Query text must be at least 3 characters long"
                        },
                    }
                },
            },
            500: {
                "description": "Internal Server Error - Query processing failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Failed to process streaming query: Knowledge graph unavailable"
                        },
                    }
                },
            },
        },
    )
    async def query_text_stream(request: QueryRequest):
        """
        Advanced RAG query endpoint with flexible streaming response.

        This endpoint provides the most flexible querying experience, supporting both real-time streaming
        and complete response delivery based on your integration needs.

        **Response Modes:**
        - Real-time response delivery as content is generated
        - NDJSON format: each line is a separate JSON object
        - First line: `{"references": [...]}` (if include_references=True)
        - Subsequent lines: `{"response": "content chunk"}`
        - Error handling: `{"error": "error message"}`

        > If stream parameter is False, or the query hit LLM cache, complete response delivered in a single streaming message.

        **Response Format Details**
        - **Content-Type**: `application/x-ndjson` (Newline-Delimited JSON)
        - **Structure**: Each line is an independent, valid JSON object
        - **Parsing**: Process line-by-line, each line is self-contained
        - **Headers**: Includes cache control and connection management

        **Query Modes (same as /query endpoint)**
        - **local**: Entity-focused retrieval with direct relationships
        - **global**: Pattern analysis across the knowledge graph
        - **hybrid**: Combined local and global strategies
        - **naive**: Vector similarity search only
        - **mix**: Integrated knowledge graph + vector retrieval (recommended)
        - **bypass**: Direct LLM query without knowledge retrieval

        conversation_history parameteris sent to LLM only, does not affect retrieval results.

        **Usage Examples**

        Real-time streaming query:
        ```json
        {
            "query": "Explain machine learning algorithms",
            "mode": "mix",
            "stream": true,
            "include_references": true
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        Complete response query:
        ```json
        {
            "query": "What is deep learning?",
            "mode": "hybrid",
            "stream": false,
            "response_type": "Multiple Paragraphs"
        }
        ```

        Conversation with context:
        ```json
        {
            "query": "Can you elaborate on that?",
            "stream": true,
            "conversation_history": [
                {"role": "user", "content": "What is neural network?"},
                {"role": "assistant", "content": "A neural network is..."}
            ]
        }
        ```

        **Response Processing:**

        ```python
        async for line in response.iter_lines():
            data = json.loads(line)
            if "references" in data:
                # Handle references (first message)
                references = data["references"]
            if "response" in data:
                # Handle content chunk
                content_chunk = data["response"]
            if "error" in data:
                # Handle error
                error_message = data["error"]
        ```

        **Error Handling:**
        - Streaming errors are delivered as `{"error": "message"}` lines
        - Non-streaming errors raise HTTP exceptions
        - Partial responses may be delivered before errors in streaming mode
        - Always check for error objects when processing streaming responses

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The question or prompt to process (min 3 characters)
                - **mode**: Query strategy - "mix" recommended for best results
                - **stream**: Enable streaming (True) or complete response (False)
                - **include_references**: Whether to include source citations
                - **response_type**: Format preference (e.g., "Multiple Paragraphs")
                - **top_k**: Number of top entities/relations to retrieve
                - **conversation_history**: Previous dialogue context for multi-turn conversations
                - **max_total_tokens**: Token budget for the entire response

        Returns:
            StreamingResponse: NDJSON streaming response containing:
                - **Streaming mode**: Multiple JSON objects, one per line
                  - References object (if requested): `{"references": [...]}`
                  - Content chunks: `{"response": "chunk content"}`
                  - Error objects: `{"error": "error message"}`
                - **Non-streaming mode**: Single JSON object
                  - Complete response: `{"references": [...], "response": "complete content"}`

        Raises:
            HTTPException:
                - 400: Invalid input parameters (e.g., query too short, invalid mode)
                - 500: Internal processing error (e.g., LLM service unavailable)

        Note:
            This endpoint is ideal for applications requiring flexible response delivery.
            Use streaming mode for real-time interfaces and non-streaming for batch processing.
        """
        try:
            _log_query_request("/query/stream", request)
            # Use the stream parameter from the request, defaulting to True if not specified
            stream_mode = request.stream if request.stream is not None else True
            current_rag = rag.resolve_current()
            current_workspace = str(
                getattr(getattr(current_rag, "text_chunks", None), "workspace", default_workspace)
                or default_workspace
            )
            param = request.to_query_params(stream_mode)

            from fastapi.responses import StreamingResponse

            async def stream_generator():
                status_queue: asyncio.Queue[dict[str, Any]] | None = (
                    asyncio.Queue()
                    if request.agent_search and request.stream_agent_status
                    else None
                )
                agent_bundle: AgentSearchMergeBundle | None = None
                result: dict[str, Any] | None = None

                try:
                    if request.agent_search:
                        loop = asyncio.get_running_loop()
                        heartbeat_interval_seconds = 5.0
                        last_heartbeat_at = loop.time()
                        if request.stream_agent_status:
                            yield (
                                f"{json.dumps({'agent_status': {'agent_search_id': None, 'phase': 'starting', 'message': 'agent_search started', 'event_type': 'local.started', 'opencode_session_id': None}})}\n"
                            )
                        agent_task = asyncio.create_task(
                            _run_agent_search_pipeline_guarded(
                                current_rag,
                                request,
                                workspace=current_workspace,
                                param=param,
                                status_queue=status_queue,
                            )
                        )

                        while not agent_task.done():
                            if status_queue is None:
                                await asyncio.sleep(0.01)
                                continue
                            try:
                                agent_status = await asyncio.wait_for(
                                    status_queue.get(), timeout=0.1
                                )
                                last_heartbeat_at = loop.time()
                                yield f"{json.dumps({'agent_status': agent_status})}\n"
                            except asyncio.TimeoutError:
                                now = loop.time()
                                if (
                                    now - last_heartbeat_at
                                    >= heartbeat_interval_seconds
                                ):
                                    last_heartbeat_at = now
                                    yield (
                                        f"{json.dumps({'agent_status': {'agent_search_id': None, 'phase': 'waiting', 'message': 'agent_search is still running', 'event_type': 'local.heartbeat', 'opencode_session_id': None}})}\n"
                                    )
                                await asyncio.sleep(0.01)

                        agent_bundle = await agent_task
                        if status_queue is not None:
                            while not status_queue.empty():
                                agent_status = status_queue.get_nowait()
                                yield f"{json.dumps({'agent_status': agent_status})}\n"

                    if agent_bundle is not None and request.stream_agent_status:
                        yield (
                            f"{json.dumps({'agent_status': {'agent_search_id': agent_bundle.public_result.agent_search_id, 'phase': 'rag_started', 'message': 'agent_search completed, starting rag query', 'event_type': 'local.rag_started', 'opencode_session_id': agent_bundle.public_result.opencode_session_id}})}\n"
                        )

                    result = await current_rag.aquery_llm(
                        request.query,
                        param=param,
                        agent_context=_build_agent_context(agent_bundle),
                    )

                    # Extract references and LLM response from unified result
                    data = (result or {}).get("data", {})
                    references = _normalize_chunk_level_references(data)
                    llm_response = (result or {}).get("llm_response", {})

                    if request.include_references:
                        references = _enrich_references_with_chunk_preview(
                            references, data.get("chunks", [])
                        )

                    if llm_response.get("is_streaming"):
                        first_payload: dict[str, Any] = {}
                        if request.include_references:
                            first_payload["references"] = references
                        if agent_bundle is not None:
                            first_payload["agent_search_result"] = (
                                agent_bundle.public_result.model_dump(exclude_none=True)
                            )
                        if first_payload:
                            yield f"{json.dumps(first_payload)}\n"

                        response_stream = llm_response.get("response_iterator")
                        if response_stream:
                            last_response_chunk = ""
                            try:
                                async for chunk in response_stream:
                                    if chunk:
                                        last_response_chunk = str(chunk)
                                        yield f"{json.dumps({'response': chunk})}\n"
                                _log_query_response(
                                    "/query/stream",
                                    last_response_chunk,
                                )
                            except Exception as e:
                                logger.error(f"Streaming error: {str(e)}")
                                yield f"{json.dumps({'error': str(e)})}\n"
                    else:
                        response_content = llm_response.get("content", "")
                        if not response_content:
                            response_content = "No relevant context found for the query."
                        _log_query_response("/query/stream", response_content)

                        complete_response: dict[str, Any] = {"response": response_content}
                        if request.include_references:
                            complete_response["references"] = references
                        if agent_bundle is not None:
                            complete_response["agent_search_result"] = (
                                agent_bundle.public_result.model_dump(exclude_none=True)
                            )

                        yield f"{json.dumps(complete_response)}\n"
                except Exception as e:
                    logger.error(
                        f"Error during streaming query generation: {str(e)}",
                        exc_info=True,
                    )
                    yield f"{json.dumps({'error': str(e)})}\n"

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "application/x-ndjson",
                    "X-Accel-Buffering": "no",  # Ensure proper handling of streaming response when proxied by Nginx
                },
            )
        except Exception as e:
            logger.error(f"Error processing streaming query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/query/data",
        response_model=QueryDataResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Successful data retrieval response with structured RAG data",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["success", "failure"],
                                    "description": "Query execution status",
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Status message describing the result",
                                },
                                "data": {
                                    "type": "object",
                                    "properties": {
                                        "entities": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "string"},
                                                    "description": {"type": "string"},
                                                },
                                                "required": ["id", "description"],
                                            },
                                            "description": "Retrieved entities with only identifier and description for LLM agent consumption",
                                        },
                                        "relationships": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "string"},
                                                    "description": {"type": "string"},
                                                },
                                                "required": ["id", "description"],
                                            },
                                            "description": "Retrieved relationships with only identifier and description for LLM agent consumption",
                                        },
                                        "chunks": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "chunk_id": {"type": "string"},
                                                    "full_doc_id": {
                                                        "type": "string",
                                                        "nullable": True,
                                                    },
                                                    "image_text": {
                                                        "type": "string",
                                                        "nullable": True,
                                                    },
                                                    "content_type": {
                                                        "type": "string",
                                                        "nullable": True,
                                                    },
                                                },
                                                "required": [
                                                    "chunk_id",
                                                    "full_doc_id",
                                                    "image_text",
                                                    "content_type",
                                                ],
                                            },
                                            "description": "Retrieved chunks with compact source and OCR metadata only",
                                        },
                                    },
                                    "description": "Agent-focused structured retrieval data. Unlike /query, this endpoint does not return a prompt string or reference list.",
                                },
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "query_mode": {"type": "string"},
                                        "keywords": {
                                            "type": "object",
                                            "properties": {
                                                "high_level": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "low_level": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                        "processing_info": {
                                            "type": "object",
                                            "properties": {
                                                "total_entities_found": {
                                                    "type": "integer"
                                                },
                                                "total_relations_found": {
                                                    "type": "integer"
                                                },
                                                "entities_after_truncation": {
                                                    "type": "integer"
                                                },
                                                "relations_after_truncation": {
                                                    "type": "integer"
                                                },
                                                "final_chunks_count": {
                                                    "type": "integer"
                                                },
                                            },
                                        },
                                    },
                                    "description": "Query metadata including mode, keywords, and processing information",
                                },
                            },
                            "required": ["status", "message", "data", "metadata"],
                        },
                        "examples": {
                            "successful_local_mode": {
                                "summary": "Local mode data retrieval",
                                "description": "Example of structured data from local mode query focusing on specific entities",
                                "value": {
                                    "status": "success",
                                    "message": "Query executed successfully",
                                    "data": {
                                        "entities": [
                                            {
                                                "id": "Neural Networks",
                                                "description": "Computational models inspired by biological neural networks",
                                            }
                                        ],
                                        "relationships": [
                                            {
                                                "id": "Neural Networks -> Machine Learning",
                                                "description": "Neural networks are a subset of machine learning algorithms",
                                            }
                                        ],
                                        "chunks": [
                                            {
                                                "chunk_id": "chunk-123",
                                                "full_doc_id": "doc-ai-basics",
                                                "image_text": None,
                                                "content_type": "text",
                                            }
                                        ],
                                    },
                                    "metadata": {
                                        "query_mode": "local",
                                        "keywords": {
                                            "high_level": ["neural", "networks"],
                                            "low_level": [
                                                "computation",
                                                "model",
                                                "algorithm",
                                            ],
                                        },
                                        "processing_info": {
                                            "total_entities_found": 5,
                                            "total_relations_found": 3,
                                            "entities_after_truncation": 1,
                                            "relations_after_truncation": 1,
                                            "final_chunks_count": 1,
                                        },
                                    },
                                },
                            },
                            "global_mode": {
                                "summary": "Global mode data retrieval",
                                "description": "Example of structured data from global mode query analyzing broader patterns",
                                "value": {
                                    "status": "success",
                                    "message": "Query executed successfully",
                                    "data": {
                                        "entities": [],
                                        "relationships": [
                                            {
                                                "id": "Artificial Intelligence -> Machine Learning",
                                                "description": "AI encompasses machine learning as a core component",
                                            }
                                        ],
                                        "chunks": [],
                                    },
                                    "metadata": {
                                        "query_mode": "global",
                                        "keywords": {
                                            "high_level": [
                                                "artificial",
                                                "intelligence",
                                                "overview",
                                            ],
                                            "low_level": [],
                                        },
                                    },
                                },
                            },
                            "naive_mode": {
                                "summary": "Naive mode data retrieval",
                                "description": "Example of structured data from naive mode using only vector search",
                                "value": {
                                    "status": "success",
                                    "message": "Query executed successfully",
                                    "data": {
                                        "entities": [],
                                        "relationships": [],
                                        "chunks": [
                                            {
                                                "chunk_id": "chunk-789",
                                                "full_doc_id": "doc-deep-learning",
                                                "image_text": None,
                                                "content_type": "text",
                                            }
                                        ],
                                    },
                                    "metadata": {
                                        "query_mode": "naive",
                                        "keywords": {"high_level": [], "low_level": []},
                                    },
                                },
                            },
                        },
                    }
                },
            },
            400: {
                "description": "Bad Request - Invalid input parameters",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Query text must be at least 3 characters long"
                        },
                    }
                },
            },
            500: {
                "description": "Internal Server Error - Data retrieval failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Failed to retrieve data: Knowledge graph unavailable"
                        },
                    }
                },
            },
        },
    )
    async def query_data(request: QueryRequest):
        """
        Advanced data retrieval endpoint for LLM-agent structured RAG data.

        This endpoint follows the same request parameters, controls, and retrieval flow
        as /query, but returns compact structured data instead of an LLM response prompt.
        It is intended for:
        - **LLM Agents**: Consume separated entities, relationships, and chunks
        - **System Integration**: Get structured retrieval inputs for custom processing
        - **Debugging**: Understand which structured items retrieval selected

        **Key Features:**
        - No LLM generation - pure data retrieval
        - Same retrieval behavior as /query for the same parameters
        - Compact structured output with entities, relationships, and chunks
        - No prompt string and no reference list in data
        - Detailed metadata about processing and keywords
        - Compatible with all query modes and parameters

        **Query Mode Behaviors:**
        - **local**: Returns entities and their direct relationships plus related chunks
        - **global**: Returns relationship patterns across the knowledge graph
        - **hybrid**: Combines local and global retrieval strategies
        - **naive**: Returns only vector-retrieved text chunks (no knowledge graph)
        - **mix**: Integrates knowledge graph data with vector-retrieved chunks
        - **bypass**: Returns empty data arrays (used for direct LLM queries)

        **Data Structure:**
        - **entities**: Objects with `id` and `description`
        - **relationships**: Objects with `id` and `description`
        - **chunks**: Objects with `chunk_id`, `full_doc_id`, `image_text`, and `content_type`
        - **metadata**: Processing information, keywords, and query statistics

        **Usage Examples:**

        Analyze entity relationships:
        ```json
        {
            "query": "machine learning algorithms",
            "mode": "local",
            "top_k": 10
        }
        ```

        Explore global patterns:
        ```json
        {
            "query": "artificial intelligence trends",
            "mode": "global",
            "max_relation_tokens": 2000
        }
        ```

        Vector similarity search:
        ```json
        {
            "query": "neural network architectures",
            "mode": "naive",
            "chunk_top_k": 5
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        **Response Analysis:**
        - **Empty arrays**: Normal for certain modes (e.g., naive mode has no entities/relationships)
        - **Processing info**: Shows retrieval statistics and token usage
        - **Keywords**: High-level and low-level keywords extracted from query

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The search query to analyze (min 3 characters)
                - **mode**: Retrieval strategy affecting data types returned
                - **top_k**: Number of top entities/relationships to retrieve
                - **chunk_top_k**: Number of text chunks to retrieve
                - **max_entity_tokens**: Token limit for entity context
                - **max_relation_tokens**: Token limit for relationship context
                - **max_total_tokens**: Overall token budget for retrieval

        Returns:
            QueryDataResponse: Structured JSON response containing:
                - **status**: "success" or "failure"
                - **message**: Human-readable status description
                - **data**: Agent-focused retrieval results with entities, relationships, and chunks
                - **metadata**: Query processing information and statistics

        Raises:
            HTTPException:
                - 400: Invalid input parameters (e.g., query too short, invalid mode)
                - 500: Internal processing error (e.g., knowledge graph unavailable)

        Note:
            The `include_references` parameter only affects /query and /query/stream.
            /query/data always returns the compact agent-focused structure above.
        """
        try:
            _log_query_request("/query/data", request)
            current_rag = rag.resolve_current()
            current_workspace = str(
                getattr(getattr(current_rag, "text_chunks", None), "workspace", default_workspace)
                or default_workspace
            )
            param = request.to_query_params(False)  # No streaming for data endpoint
            param.conversation_history = []
            response, agent_bundle = await _run_query_with_agent_context(
                current_rag,
                request,
                current_workspace,
                param=param,
                data_only=True,
            )

            # aquery_data returns the new format with status, message, data, and metadata
            if isinstance(response, dict):
                if agent_bundle is not None:
                    response["agent_search_result"] = agent_bundle.public_result.model_dump(
                        exclude_none=True
                    )
                response = _format_query_data_for_agent(response)
                return QueryDataResponse(**response)
            else:
                # Handle unexpected response format
                return QueryDataResponse(
                    status="failure",
                    message="Invalid response type",
                    data={},
                    metadata={},
                )
        except Exception as e:
            logger.error(f"Error processing data query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return router

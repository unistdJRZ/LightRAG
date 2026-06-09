"""
Knowledge-base-level QA routes.
"""

from contextvars import ContextVar
from typing import Any, Literal, Mapping, Optional
import traceback

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from lightrag.api.utils_api import (
    WorkspaceObjectProxy,
    create_workspace_scope_dependency,
    get_combined_auth_dependency,
)
from lightrag.utils import logger


class KnowledgeBaseQASearchRequest(BaseModel):
    workspace: Optional[str] = Field(
        default=None,
        description="Target workspace for this request.",
    )
    query: str = Field(
        min_length=1,
        description="Question text used for rule matching or vector retrieval.",
    )
    mode: Literal["rule", "vector"] = Field(
        default="vector",
        description="Retrieval mode. 'rule' uses deterministic question matching; 'vector' uses embedding search.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of QA rows to return.",
    )

    @field_validator("query", mode="after")
    @classmethod
    def query_strip_after(cls, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        return query


class KnowledgeBaseQAItem(BaseModel):
    id: str = Field(description="Knowledge-base QA id")
    workspace: str = Field(description="Owning workspace")
    question: str = Field(description="Preset question")
    answer: str = Field(description="Preset answer")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: int | None = None
    updated_at: int | None = None
    distance: float | None = Field(
        default=None,
        description="Vector distance when mode='vector'.",
    )
    match_rank: int | None = Field(
        default=None,
        description="Rule match rank when mode='rule'. Lower is better.",
    )


class KnowledgeBaseQASearchResponse(BaseModel):
    workspace: str = Field(description="Resolved workspace")
    mode: Literal["rule", "vector"]
    results: list[KnowledgeBaseQAItem]


def create_knowledge_base_qa_routes(
    rag_by_workspace: dict[str, Any],
    api_key: Optional[str] = None,
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
        "knowledge_base_qa_workspace", default=default_workspace
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
    router = APIRouter(
        tags=["knowledge-base-qa"],
        dependencies=[Depends(workspace_scope)],
    )

    @router.post(
        "/knowledge-base-qa/search",
        response_model=KnowledgeBaseQASearchResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def search_knowledge_base_qa(request: KnowledgeBaseQASearchRequest):
        try:
            rows = await rag.query_knowledge_base_qa(
                query=request.query,
                top_k=request.top_k,
                mode=request.mode,
            )
            resolved_workspace = getattr(rag, "workspace", default_workspace)
            return {
                "workspace": resolved_workspace,
                "mode": request.mode,
                "results": rows,
            }
        except ValueError as e:
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            logger.error(f"Error searching knowledge-base QA: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error searching knowledge-base QA: {str(e)}",
            )

    return router

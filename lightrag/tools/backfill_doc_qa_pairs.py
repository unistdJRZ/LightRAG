#!/usr/bin/env python3
"""
Backfill document QA pairs into PostgreSQL.

Usage:
    python -m lightrag.tools.backfill_doc_qa_pairs --workspace default
    python -m lightrag.tools.backfill_doc_qa_pairs --doc-id doc-xxx --force
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from functools import partial
from typing import Any, Callable

import numpy as np
from dotenv import load_dotenv

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.base import DocStatus, EmbeddingFunc
from lightrag.kg import STORAGES
from lightrag.kg.postgres_impl import ClientManager
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace
from lightrag.qa_extraction import (
    build_doc_qa_vector_data,
    extract_doc_qa_pairs,
    replace_doc_qa_pairs,
)
from lightrag.utils import logger, priority_limit_async_func_call, setup_logger


load_dotenv(dotenv_path=".env", override=False)
setup_logger("lightrag", level=os.getenv("LOG_LEVEL", "INFO"))


def _default_host(binding: str) -> str:
    return {
        "ollama": "http://localhost:11434",
        "lollms": "http://localhost:9600",
        "openai": "https://api.openai.com/v1",
        "azure_openai": "",
        "gemini": "https://generativelanguage.googleapis.com",
    }.get(binding, os.getenv("LLM_BINDING_HOST", ""))


def _create_llm_model_func(
    *,
    binding: str,
    model: str,
    host: str,
    api_key: str | None,
    timeout: int,
    max_async: int,
) -> Callable[..., Any]:
    async def openai_like(prompt: str, **kwargs: Any) -> str:
        from lightrag.llm.openai import openai_complete_if_cache

        return await openai_complete_if_cache(
            model,
            prompt,
            base_url=host,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )

    async def azure_openai(prompt: str, **kwargs: Any) -> str:
        from lightrag.llm.azure_openai import azure_openai_complete_if_cache

        return await azure_openai_complete_if_cache(
            model,
            prompt,
            base_url=host,
            api_key=os.getenv("AZURE_OPENAI_API_KEY", api_key),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            timeout=timeout,
            **kwargs,
        )

    async def gemini(prompt: str, **kwargs: Any) -> str:
        from lightrag.llm.gemini import gemini_complete_if_cache

        return await gemini_complete_if_cache(
            model,
            prompt,
            api_key=api_key,
            base_url=host,
            timeout=timeout,
            **kwargs,
        )

    if binding == "ollama":
        from lightrag.llm.ollama import ollama_model_complete

        func = partial(
            ollama_model_complete,
            llm_model_name=model,
            host=host,
            api_key=api_key,
            timeout=timeout,
        )
    elif binding == "azure_openai":
        func = azure_openai
    elif binding == "gemini":
        func = gemini
    else:
        func = openai_like

    return priority_limit_async_func_call(
        max_async,
        llm_timeout=timeout,
        queue_name="QA backfill LLM",
    )(func)


def _create_embedding_func(
    *,
    binding: str,
    model: str | None,
    host: str,
    api_key: str | None,
) -> EmbeddingFunc:
    provider_func = None
    if binding == "ollama":
        from lightrag.llm.ollama import ollama_embed

        provider_func = ollama_embed
    elif binding == "azure_openai":
        from lightrag.llm.azure_openai import azure_openai_embed

        provider_func = azure_openai_embed
    elif binding == "gemini":
        from lightrag.llm.gemini import gemini_embed

        provider_func = gemini_embed
    elif binding == "jina":
        from lightrag.llm.jina import jina_embed

        provider_func = jina_embed
    elif binding == "qwen":
        from lightrag.llm.transforms_qwen import qwen_embed

        provider_func = qwen_embed
    else:
        from lightrag.llm.openai import openai_embed

        provider_func = openai_embed

    provider_dim = (
        provider_func.embedding_dim
        if isinstance(provider_func, EmbeddingFunc)
        else None
    )
    provider_max_tokens = (
        provider_func.max_token_size
        if isinstance(provider_func, EmbeddingFunc)
        else None
    )
    embedding_dim = int(os.getenv("EMBEDDING_DIM") or provider_dim or 1024)
    max_token_size = int(os.getenv("EMBEDDING_TOKEN_LIMIT") or provider_max_tokens or 8192)
    actual_func = (
        provider_func.func if isinstance(provider_func, EmbeddingFunc) else provider_func
    )

    async def embed(texts: list[str], embedding_dim: int | None = None) -> np.ndarray:
        kwargs: dict[str, Any] = {"texts": texts}
        if binding in {"openai", "jina"}:
            kwargs.update({"base_url": host, "api_key": api_key})
        elif binding == "ollama":
            kwargs.update({"host": host, "api_key": api_key})
        elif binding == "gemini":
            kwargs.update({"base_url": host, "api_key": api_key})
        elif binding == "azure_openai":
            kwargs.update({"api_key": os.getenv("AZURE_OPENAI_API_KEY", api_key or "")})
        if model:
            if binding == "ollama":
                kwargs["embed_model"] = model
            else:
                kwargs["model"] = model
        if binding in {"openai", "jina", "gemini", "qwen"}:
            kwargs["embedding_dim"] = embedding_dim

        result = await actual_func(**kwargs)
        arr = np.asarray(result, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=embed,
        model_name=model,
    )


def _get_vector_storage_class(storage_name: str):
    module_name = STORAGES[storage_name]
    module = importlib.import_module(f"lightrag{module_name}")
    return getattr(module, storage_name)


async def _create_qa_vector_storage(
    *,
    args: argparse.Namespace,
    workspace: str,
):
    storage_name = args.vector_storage or os.getenv(
        "LIGHTRAG_VECTOR_STORAGE", "MilvusVectorDBStorage"
    )
    binding = args.embedding_binding or os.getenv("EMBEDDING_BINDING", "ollama")
    model = args.embedding_model or os.getenv("EMBEDDING_MODEL")
    host = args.embedding_binding_host or os.getenv("EMBEDDING_BINDING_HOST") or _default_host(binding)
    api_key = args.embedding_binding_api_key or os.getenv("EMBEDDING_BINDING_API_KEY")
    embedding_func = _create_embedding_func(
        binding=binding,
        model=model,
        host=host,
        api_key=api_key,
    )
    storage_cls = _get_vector_storage_class(storage_name)
    storage = storage_cls(
        namespace=NameSpace.VECTOR_STORE_QA_PAIRS,
        workspace=workspace,
        global_config={
            "working_dir": os.getenv("WORKING_DIR", "./rag_storage"),
            "embedding_batch_num": int(os.getenv("EMBEDDING_BATCH_NUM", "10")),
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": float(
                    os.getenv("COSINE_THRESHOLD", "0.2")
                )
            },
        },
        embedding_func=embedding_func,
        meta_fields={"doc_id", "qa_id"},
    )
    await storage.initialize()
    return storage


async def _fetch_docs(
    db: Any,
    *,
    workspace: str,
    doc_id: str | None,
    force: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    params: list[Any] = [workspace, DocStatus.PROCESSED.value]
    where = ["f.workspace=$1", "s.status=$2"]

    if doc_id:
        params.append(doc_id)
        where.append(f"f.id=${len(params)}")

    if not force:
        where.append(
            """
            NOT EXISTS (
                SELECT 1 FROM LIGHTRAG_DOC_QA_PAIRS q
                WHERE q.workspace=f.workspace AND q.doc_id=f.id
            )
            """
        )

    sql = f"""
        SELECT f.id, COALESCE(f.content, '') AS content
        FROM LIGHTRAG_DOC_FULL f
        JOIN LIGHTRAG_DOC_STATUS s
          ON s.workspace=f.workspace AND s.id=f.id
        WHERE {' AND '.join(where)}
        ORDER BY f.update_time DESC
    """
    if limit is not None:
        params.append(limit)
        sql += f" LIMIT ${len(params)}"

    return await db.query(sql, params, multirows=True)


async def _run(args: argparse.Namespace) -> int:
    initialize_share_data()
    db = None
    qa_vector_storage = None
    try:
        config = ClientManager.get_config()
        workspace = args.workspace or config.get("workspace") or os.getenv("WORKSPACE") or "default"
        binding = args.llm_binding or os.getenv("LLM_BINDING", "ollama")
        model = args.llm_model or os.getenv("LLM_MODEL", "mistral-nemo:latest")
        host = args.llm_binding_host or os.getenv("LLM_BINDING_HOST") or _default_host(binding)
        api_key = args.llm_binding_api_key or os.getenv("LLM_BINDING_API_KEY")
        timeout = args.llm_timeout or int(os.getenv("LLM_TIMEOUT", "240"))
        max_async = args.max_async or int(os.getenv("MAX_ASYNC", "4"))

        db = await ClientManager.get_client()
        llm_model_func = _create_llm_model_func(
            binding=binding,
            model=model,
            host=host,
            api_key=api_key,
            timeout=timeout,
            max_async=max_async,
        )
        docs = await _fetch_docs(
            db,
            workspace=workspace,
            doc_id=args.doc_id,
            force=args.force,
            limit=args.limit,
        )
        logger.info("Found %d document(s) for QA backfill in workspace=%s", len(docs), workspace)
        if not docs:
            return 0
        qa_vector_storage = await _create_qa_vector_storage(args=args, workspace=workspace)

        semaphore = asyncio.Semaphore(args.doc_concurrency)
        success_count = 0
        failure_count = 0

        async def process_doc(row: dict[str, Any]) -> None:
            nonlocal success_count, failure_count
            async with semaphore:
                doc_id = row["id"]
                try:
                    old_vector_ids: list[str] = []
                    if args.force:
                        old_rows = await db.query(
                            "SELECT id FROM LIGHTRAG_DOC_QA_PAIRS WHERE workspace=$1 AND doc_id=$2",
                            [workspace, doc_id],
                            multirows=True,
                        )
                        old_vector_ids = [str(item["id"]) for item in old_rows or []]
                    extraction_result = await extract_doc_qa_pairs(
                        doc_id=doc_id,
                        document_text=row.get("content") or "",
                        llm_model_func=llm_model_func,
                    )
                    qa_count = await replace_doc_qa_pairs(
                        db=db,
                        workspace=workspace,
                        doc_id=doc_id,
                        extraction_result=extraction_result,
                    )
                    vector_data = build_doc_qa_vector_data(
                        doc_id=doc_id,
                        extraction_result=extraction_result,
                    )
                    if vector_data:
                        if old_vector_ids:
                            await qa_vector_storage.delete(old_vector_ids)
                        await qa_vector_storage.upsert(vector_data)
                    success_count += 1
                    logger.info("Backfilled %d QA pair(s) for %s", qa_count, doc_id)
                except Exception as e:
                    failure_count += 1
                    logger.exception("Failed to backfill QA pairs for %s: %s", doc_id, e)

        await asyncio.gather(*(process_doc(row) for row in docs))
        logger.info(
            "QA backfill finished: success=%d failure=%d total=%d",
            success_count,
            failure_count,
            len(docs),
        )
        return 1 if failure_count else 0
    finally:
        if qa_vector_storage is not None:
            await qa_vector_storage.finalize()
        if db is not None:
            await ClientManager.release_client(db)
        finalize_share_data()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill document QA pairs from LIGHTRAG_DOC_FULL content into PostgreSQL."
    )
    parser.add_argument("--workspace", help="PostgreSQL workspace to process")
    parser.add_argument("--doc-id", help="Only process one document id")
    parser.add_argument("--force", action="store_true", help="Regenerate QA pairs even if rows already exist")
    parser.add_argument("--limit", type=int, help="Maximum number of documents to process")
    parser.add_argument("--doc-concurrency", type=int, default=1, help="Number of documents to process concurrently")
    parser.add_argument("--llm-binding", help="LLM binding, defaults to LLM_BINDING")
    parser.add_argument("--llm-model", help="LLM model, defaults to LLM_MODEL")
    parser.add_argument("--llm-binding-host", help="LLM endpoint, defaults to LLM_BINDING_HOST")
    parser.add_argument("--llm-binding-api-key", help="LLM API key, defaults to LLM_BINDING_API_KEY")
    parser.add_argument("--llm-timeout", type=int, help="LLM timeout seconds")
    parser.add_argument("--max-async", type=int, help="Max concurrent LLM calls")
    parser.add_argument("--vector-storage", help="Vector storage implementation for QA question vectors")
    parser.add_argument("--embedding-binding", help="Embedding binding, defaults to EMBEDDING_BINDING")
    parser.add_argument("--embedding-model", help="Embedding model, defaults to EMBEDDING_MODEL")
    parser.add_argument("--embedding-binding-host", help="Embedding endpoint, defaults to EMBEDDING_BINDING_HOST")
    parser.add_argument("--embedding-binding-api-key", help="Embedding API key, defaults to EMBEDDING_BINDING_API_KEY")
    args = parser.parse_args()

    if args.doc_concurrency < 1:
        parser.error("--doc-concurrency must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import gc
import os
from pathlib import Path
from typing import Any

from lightrag.kg.shared_storage import initialize_share_data
from lightrag.namespace import NameSpace
from lightrag.utils import logger
from lightrag.workspace_config import load_workspace_config

from lightrag.tools.migrate_workspaces_to_postgres import (
    KV_NAMESPACES,
    VECTOR_NAMESPACES,
    build_dummy_embedding,
    build_global_config,
    create_source_graph,
    create_source_vector,
    detect_source_storage,
    infer_target_embedding_model_name,
    migrate_doc_status,
    migrate_graph,
    migrate_json_kv_namespace,
    migrate_vector_namespace,
    namespace_file,
    read_embedding_dim,
)


VECTOR_META_FIELDS: dict[str, set[str]] = {
    NameSpace.VECTOR_STORE_ENTITIES: {"entity_name", "source_id", "content", "file_path"},
    NameSpace.VECTOR_STORE_RELATIONSHIPS: {
        "src_id",
        "tgt_id",
        "source_id",
        "content",
        "file_path",
    },
    NameSpace.VECTOR_STORE_CHUNKS: {
        "full_doc_id",
        "content",
        "content_type",
        "file_path",
        "page_id",
        "bbox",
        "ocr_chunk_id",
        "image_base64",
        "image_text",
    },
}


async def create_target_bundle(
    working_dir: str,
    workspace: str,
    embedding_dim: int,
    batch_size: int,
    model_name: str | None,
) -> dict[str, Any]:
    try:
        from lightrag.kg.milvus_impl import MilvusVectorDBStorage
        from lightrag.kg.neo4j_impl import Neo4JStorage
        from lightrag.kg.postgres_impl import PGDocStatusStorage, PGKVStorage
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import the hybrid storage backends. Ensure PostgreSQL, Milvus, and Neo4j dependencies are installed."
        ) from exc

    global_config = build_global_config(working_dir, batch_size=batch_size)
    embedding = build_dummy_embedding(embedding_dim, model_name=model_name)

    storages: dict[str, Any] = {
        "kv": {
            namespace: PGKVStorage(
                namespace=namespace,
                workspace=workspace,
                global_config=global_config,
                embedding_func=None,
            )
            for namespace in KV_NAMESPACES
        },
        "doc_status": PGDocStatusStorage(
            namespace=NameSpace.DOC_STATUS,
            workspace=workspace,
            global_config=global_config,
            embedding_func=None,
        ),
        "graph": Neo4JStorage(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace=workspace,
            global_config=global_config,
            embedding_func=None,
        ),
        "vector": {
            namespace: MilvusVectorDBStorage(
                namespace=namespace,
                workspace=workspace,
                global_config=global_config,
                embedding_func=embedding,
                meta_fields=VECTOR_META_FIELDS[namespace],
            )
            for namespace in VECTOR_NAMESPACES
        },
    }

    for storage in storages["kv"].values():
        await storage.initialize()
    await storages["doc_status"].initialize()
    await storages["graph"].initialize()
    for storage in storages["vector"].values():
        await storage.initialize()

    return storages


async def finalize_target_bundle(storages: dict[str, Any] | None) -> None:
    if storages is None:
        return
    for storage in storages["kv"].values():
        await storage.finalize()
    await storages["doc_status"].finalize()
    await storages["graph"].finalize()
    for storage in storages["vector"].values():
        await storage.finalize()


async def clear_target_bundle(storages: dict[str, Any], workspace: str) -> None:
    logger.info(
        "[%s] Clearing target workspace data before migration to avoid dirty state",
        workspace,
    )

    for namespace, storage in storages["vector"].items():
        result = await storage.drop()
        if result.get("status") != "success":
            raise RuntimeError(
                f"[{workspace}] Failed to clear Milvus namespace '{namespace}': {result.get('message', 'unknown error')}"
            )

    graph_result = await storages["graph"].drop()
    if graph_result.get("status") != "success":
        raise RuntimeError(
            f"[{workspace}] Failed to clear Neo4j graph storage: {graph_result.get('message', 'unknown error')}"
        )

    doc_status_result = await storages["doc_status"].drop()
    if doc_status_result.get("status") != "success":
        raise RuntimeError(
            f"[{workspace}] Failed to clear PostgreSQL doc_status storage: {doc_status_result.get('message', 'unknown error')}"
        )

    for namespace, storage in storages["kv"].items():
        result = await storage.drop()
        if result.get("status") != "success":
            raise RuntimeError(
                f"[{workspace}] Failed to clear PostgreSQL KV namespace '{namespace}': {result.get('message', 'unknown error')}"
            )

    logger.info("[%s] Target workspace data cleared", workspace)


def write_switch_env_file(output_path: Path) -> None:
    content = "\n".join(
        [
            "LIGHTRAG_KV_STORAGE=PGKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


async def migrate_workspace(
    workspace: str,
    args: argparse.Namespace,
) -> None:
    logger.info("Start migrating workspace '%s'", workspace)

    if args.source_kv_storage != "JsonKVStorage":
        raise ValueError(
            f"Unsupported source KV storage for this migration tool: {args.source_kv_storage}"
        )
    if args.source_graph_storage != "NetworkXStorage":
        raise ValueError(
            f"Unsupported source graph storage for this migration tool: {args.source_graph_storage}"
        )
    if args.source_vector_storage != "NanoVectorDBStorage":
        raise ValueError(
            f"Unsupported source vector storage for this migration tool: {args.source_vector_storage}"
        )
    if args.source_doc_status_storage != "JsonDocStatusStorage":
        raise ValueError(
            "Unsupported source doc status storage for this migration tool: "
            f"{args.source_doc_status_storage}"
        )

    working_dir = Path(args.working_dir)
    vector_file = namespace_file(working_dir, workspace, "vdb", "entities.json")
    embedding_dim = args.target_embedding_dim or read_embedding_dim(vector_file)
    if embedding_dim is None:
        raise ValueError(
            f"Could not determine embedding dimension for workspace '{workspace}'. "
            "Provide --target-embedding-dim explicitly."
        )

    target_bundle = None
    if not args.dry_run:
        target_bundle = await create_target_bundle(
            working_dir=str(working_dir),
            workspace=workspace,
            embedding_dim=embedding_dim,
            batch_size=args.batch_size,
            model_name=args.inferred_embedding_model_name,
        )
        if not args.skip_target_cleanup:
            await clear_target_bundle(target_bundle, workspace)

    source_graph = await create_source_graph(str(working_dir), workspace)

    try:
        for namespace in KV_NAMESPACES:
            source_path = namespace_file(
                working_dir, workspace, "kv_store", f"{namespace}.json"
            )
            count = await migrate_json_kv_namespace(
                workspace,
                namespace,
                source_path,
                None if target_bundle is None else target_bundle["kv"][namespace],
                args.batch_size,
                args.dry_run,
            )
            logger.info("[%s] KV '%s' done: %d records", workspace, namespace, count)
            gc.collect()

        doc_status_path = namespace_file(
            working_dir, workspace, "kv_store", f"{NameSpace.DOC_STATUS}.json"
        )
        doc_count = await migrate_doc_status(
            workspace,
            doc_status_path,
            None if target_bundle is None else target_bundle["doc_status"],
            args.batch_size,
            args.dry_run,
        )
        logger.info("[%s] doc_status done: %d records", workspace, doc_count)
        gc.collect()

        node_count, edge_count = await migrate_graph(
            workspace,
            source_graph,
            None if target_bundle is None else target_bundle["graph"],
            args.dry_run,
        )
        logger.info(
            "[%s] graph done: %d nodes, %d edges",
            workspace,
            node_count,
            edge_count,
        )
        gc.collect()

        for namespace in VECTOR_NAMESPACES:
            source_vector = await create_source_vector(
                str(working_dir), workspace, namespace, embedding_dim
            )
            try:
                count = await migrate_vector_namespace(
                    workspace,
                    namespace,
                    source_vector,
                    None if target_bundle is None else target_bundle["vector"][namespace],
                    args.batch_size,
                    args.dry_run,
                )
            finally:
                await source_vector.finalize()
            logger.info("[%s] vector '%s' done: %d records", workspace, namespace, count)
            gc.collect()
    finally:
        await source_graph.finalize()
        await finalize_target_bundle(target_bundle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate enabled LightRAG workspaces from local JSON/NetworkX/NanoVectorDB "
            "storage to PostgreSQL KV/doc-status, Milvus vector, and Neo4j graph backends."
        )
    )
    parser.add_argument(
        "--workspace-config",
        default="workspace.yaml",
        help="Path to workspace.yaml",
    )
    parser.add_argument(
        "--working-dir",
        default=os.getenv("WORKING_DIR", "./rag_storage"),
        help="Source working directory containing per-workspace storage files",
    )
    parser.add_argument(
        "--workspace",
        action="append",
        dest="workspaces",
        help="Workspace id to migrate. Repeat to migrate multiple workspaces. Defaults to all enabled workspaces.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for KV/vector/doc-status migration",
    )
    parser.add_argument(
        "--target-embedding-dim",
        type=int,
        help="Embedding dimension for Milvus collections. Defaults to the source NanoVectorDB file metadata.",
    )
    parser.add_argument("--source-kv-storage", help="Override detected source KV storage type")
    parser.add_argument(
        "--source-vector-storage", help="Override detected source vector storage type"
    )
    parser.add_argument(
        "--source-graph-storage", help="Override detected source graph storage type"
    )
    parser.add_argument(
        "--source-doc-status-storage",
        help="Override detected source doc status storage type",
    )
    parser.add_argument(
        "--write-switch-env-file",
        help="Optional output file path for hybrid storage env vars",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and count source data without writing to PostgreSQL, Milvus, or Neo4j",
    )
    parser.add_argument(
        "--skip-target-cleanup",
        action="store_true",
        help="Skip clearing existing target workspace data before migration",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    initialize_share_data()
    args.source_kv_storage = detect_source_storage(
        "LIGHTRAG_KV_STORAGE", "JsonKVStorage", args.source_kv_storage
    )
    args.source_vector_storage = detect_source_storage(
        "LIGHTRAG_VECTOR_STORAGE",
        "NanoVectorDBStorage",
        args.source_vector_storage,
    )
    args.source_graph_storage = detect_source_storage(
        "LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage", args.source_graph_storage
    )
    args.source_doc_status_storage = detect_source_storage(
        "LIGHTRAG_DOC_STATUS_STORAGE",
        "JsonDocStatusStorage",
        args.source_doc_status_storage,
    )
    args.inferred_embedding_model_name = infer_target_embedding_model_name()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    config = load_workspace_config(Path(args.workspace_config))
    configured_workspaces = [workspace.id for workspace in config["workspaces"]]
    selected_workspaces = args.workspaces or configured_workspaces
    missing = sorted(set(selected_workspaces) - set(configured_workspaces))
    if missing:
        raise ValueError(
            f"Workspace(s) not found in {args.workspace_config}: {', '.join(missing)}"
        )

    logger.info(
        "Migrating %d workspace(s) from %s to PostgreSQL + Milvus + Neo4j",
        len(selected_workspaces),
        args.working_dir,
    )
    if args.dry_run:
        logger.info("Dry run mode enabled; target storages will not be modified")
    elif args.skip_target_cleanup:
        logger.warning(
            "Target cleanup is disabled; existing workspace data will be preserved"
        )
    else:
        logger.info(
            "Target workspace data will be cleared before each migration run"
        )
    if args.inferred_embedding_model_name is None:
        logger.warning(
            "Could not infer EMBEDDING_MODEL from current config; Milvus collection suffixes will not include a model-specific suffix."
        )
    else:
        logger.info(
            "Using configured embedding model for Milvus collection suffix: %s",
            args.inferred_embedding_model_name,
        )

    if args.write_switch_env_file:
        write_switch_env_file(Path(args.write_switch_env_file))
        logger.info("Wrote hybrid storage env file: %s", args.write_switch_env_file)

    for workspace in selected_workspaces:
        await migrate_workspace(workspace, args)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

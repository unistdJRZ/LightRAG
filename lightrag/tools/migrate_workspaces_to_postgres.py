from __future__ import annotations

import argparse
import asyncio
import gc
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from lightrag.namespace import NameSpace
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc, load_json, logger
from lightrag.workspace_config import load_workspace_config

try:
    import ijson  # type: ignore
except ImportError:
    ijson = None


KV_NAMESPACES = [
    NameSpace.KV_STORE_FULL_DOCS,
    NameSpace.KV_STORE_TEXT_CHUNKS,
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
    NameSpace.KV_STORE_FULL_ENTITIES,
    NameSpace.KV_STORE_FULL_RELATIONS,
    NameSpace.KV_STORE_ENTITY_CHUNKS,
    NameSpace.KV_STORE_RELATION_CHUNKS,
]

VECTOR_NAMESPACES = [
    NameSpace.VECTOR_STORE_ENTITIES,
    NameSpace.VECTOR_STORE_RELATIONSHIPS,
    NameSpace.VECTOR_STORE_CHUNKS,
]


async def _unused_embedding(*_: Any, **__: Any) -> Any:
    raise RuntimeError("Migration path should not call the embedding function")


def build_dummy_embedding(embedding_dim: int, model_name: str | None) -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        func=_unused_embedding,
        model_name=model_name,
    )


def chunk_dict_items(
    items: Iterable[tuple[str, dict[str, Any]]], batch_size: int
) -> Iterator[dict[str, dict[str, Any]]]:
    batch: dict[str, dict[str, Any]] = {}
    for key, value in items:
        batch[key] = value
        if len(batch) >= batch_size:
            yield batch
            batch = {}
    if batch:
        yield batch


def stream_json_dict(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    if not path.exists():
        return iter(())

    if ijson is not None:

        def _generator() -> Iterator[tuple[str, dict[str, Any]]]:
            with path.open("rb") as file_handle:
                yield from ijson.kvitems(file_handle, "")

        return _generator()

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 512:
        logger.warning(
            "Streaming parser 'ijson' is not installed; loading %.1f MB file into memory: %s",
            file_size_mb,
            path,
        )
    data = load_json(path) or {}
    return iter(data.items())


def detect_source_storage(
    env_name: str, default_value: str, explicit_value: str | None
) -> str:
    if explicit_value:
        return explicit_value
    return os.getenv(env_name, default_value)


def read_embedding_dim(vdb_file: Path) -> int | None:
    if not vdb_file.exists():
        return None

    with vdb_file.open("r", encoding="utf-8") as file_handle:
        head = file_handle.read(4096)
    match = re.search(r'"embedding_dim"\s*:\s*(\d+)', head)
    if match:
        return int(match.group(1))
    return None


def infer_target_embedding_model_name() -> str | None:
    """Infer the configured embedding model name from the current project config."""
    return (
        os.getenv("EMBEDDING_MODEL")
        or os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        or None
    )


def build_global_config(working_dir: str, batch_size: int) -> dict[str, Any]:
    return {
        "working_dir": working_dir,
        "embedding_batch_num": batch_size,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.2,
        },
    }


def namespace_file(working_dir: Path, workspace: str, prefix: str, suffix: str) -> Path:
    workspace_dir = working_dir / workspace if workspace else working_dir
    return workspace_dir / f"{prefix}_{suffix}"


async def migrate_json_kv_namespace(
    workspace: str,
    namespace: str,
    source_path: Path,
    target: Any,
    batch_size: int,
    dry_run: bool,
) -> int:
    if not source_path.exists():
        logger.info("[%s] Skip KV namespace '%s': source file not found", workspace, namespace)
        return 0

    migrated = 0
    for batch in chunk_dict_items(stream_json_dict(source_path), batch_size):
        migrated += len(batch)
        if not dry_run:
            await target.upsert(batch)
        if migrated % max(batch_size * 10, 1000) == 0:
            logger.info("[%s] KV '%s' migrated %d records", workspace, namespace, migrated)
    return migrated


async def migrate_doc_status(
    workspace: str,
    source_path: Path,
    target: Any,
    batch_size: int,
    dry_run: bool,
) -> int:
    if not source_path.exists():
        logger.info("[%s] Skip doc_status: source file not found", workspace)
        return 0

    migrated = 0
    for batch in chunk_dict_items(stream_json_dict(source_path), batch_size):
        migrated += len(batch)
        if not dry_run:
            await target.upsert(batch)
        if migrated % max(batch_size * 10, 1000) == 0:
            logger.info("[%s] doc_status migrated %d records", workspace, migrated)
    return migrated


async def migrate_graph(
    workspace: str,
    source: Any,
    target: Any,
    dry_run: bool,
) -> tuple[int, int]:
    nodes = await source.get_all_nodes()
    edges = await source.get_all_edges()

    if not dry_run:
        for index, node in enumerate(nodes, start=1):
            node_id = str(node.get("id") or node.get("entity_id") or "").strip()
            if not node_id:
                continue
            node_data = dict(node)
            node_data.pop("id", None)
            node_data.setdefault("entity_id", node_id)
            await target.upsert_node(node_id, node_data)
            if index % 1000 == 0:
                logger.info("[%s] graph nodes migrated %d/%d", workspace, index, len(nodes))

        for index, edge in enumerate(edges, start=1):
            source_id = str(edge.get("source") or "").strip()
            target_id = str(edge.get("target") or "").strip()
            if not source_id or not target_id:
                continue
            edge_data = dict(edge)
            edge_data.pop("source", None)
            edge_data.pop("target", None)
            await target.upsert_edge(source_id, target_id, edge_data)
            if index % 1000 == 0:
                logger.info("[%s] graph edges migrated %d/%d", workspace, index, len(edges))

    return len(nodes), len(edges)


async def migrate_vector_namespace(
    workspace: str,
    namespace: str,
    source: Any,
    target: Any,
    batch_size: int,
    dry_run: bool,
) -> int:
    storage = await source.client_storage
    raw_items = storage.get("data", [])
    if not raw_items:
        return 0

    migrated = 0
    for offset in range(0, len(raw_items), batch_size):
        batch_items = raw_items[offset : offset + batch_size]
        batch_ids = [str(item["__id__"]) for item in batch_items]
        vectors = await source.get_vectors_by_ids(batch_ids)
        payload: dict[str, dict[str, Any]] = {}
        for item in batch_items:
            item_id = str(item["__id__"])
            payload[item_id] = {
                key: value
                for key, value in item.items()
                if key != "vector"
            }
            payload[item_id]["__vector__"] = np.asarray(
                vectors[item_id], dtype=np.float32
            )

        migrated += len(payload)
        if not dry_run:
            await target.upsert_precomputed(payload)
        if migrated % max(batch_size * 10, 1000) == 0:
            logger.info("[%s] vector '%s' migrated %d records", workspace, namespace, migrated)
    return migrated


async def create_source_graph(
    working_dir: str,
    workspace: str,
) -> Any:
    try:
        from lightrag.kg.networkx_impl import NetworkXStorage
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import NetworkXStorage. Ensure the project runtime dependencies are installed."
        ) from exc

    storage = NetworkXStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace=workspace,
        global_config={"working_dir": working_dir},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


async def create_source_vector(
    working_dir: str,
    workspace: str,
    namespace: str,
    embedding_dim: int,
) -> Any:
    try:
        from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import NanoVectorDBStorage. Install the source backend dependencies, including 'nano_vectordb'."
        ) from exc

    storage = NanoVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config=build_global_config(working_dir, batch_size=64),
        embedding_func=build_dummy_embedding(embedding_dim, model_name=None),
        meta_fields=set(),
    )
    await storage.initialize()
    return storage


async def create_target_bundle(
    working_dir: str,
    workspace: str,
    embedding_dim: int,
    batch_size: int,
    model_name: str | None,
) -> dict[str, Any]:
    try:
        from lightrag.kg.postgres_impl import (
            PGDocStatusStorage,
            PGGraphStorage,
            PGKVStorage,
            PGVectorStorage,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import PostgreSQL storage backends. Ensure PostgreSQL dependencies are installed."
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
        "graph": PGGraphStorage(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace=workspace,
            global_config=global_config,
            embedding_func=None,
        ),
        "vector": {
            namespace: PGVectorStorage(
                namespace=namespace,
                workspace=workspace,
                global_config=global_config,
                embedding_func=embedding,
                meta_fields=set(),
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


async def finalize_target_bundle(storages: dict[str, Any]) -> None:
    for storage in storages["kv"].values():
        await storage.finalize()
    await storages["doc_status"].finalize()
    await storages["graph"].finalize()
    for storage in storages["vector"].values():
        await storage.finalize()


def write_switch_env_file(output_path: Path) -> None:
    content = "\n".join(
        [
            "LIGHTRAG_KV_STORAGE=PGKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=PGVectorStorage",
            "LIGHTRAG_GRAPH_STORAGE=PGGraphStorage",
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

    target_bundle = await create_target_bundle(
        working_dir=str(working_dir),
        workspace=workspace,
        embedding_dim=embedding_dim,
        batch_size=args.batch_size,
        model_name=args.inferred_embedding_model_name,
    )

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
                target_bundle["kv"][namespace],
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
            target_bundle["doc_status"],
            args.batch_size,
            args.dry_run,
        )
        logger.info("[%s] doc_status done: %d records", workspace, doc_count)
        gc.collect()

        node_count, edge_count = await migrate_graph(
            workspace,
            source_graph,
            target_bundle["graph"],
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
                    target_bundle["vector"][namespace],
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
            "storage to PostgreSQL backends."
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
        help="Embedding dimension for PostgreSQL vector tables. Defaults to the source NanoVectorDB file metadata.",
    )
    parser.add_argument(
        "--source-kv-storage",
        help="Override detected source KV storage type",
    )
    parser.add_argument(
        "--source-vector-storage",
        help="Override detected source vector storage type",
    )
    parser.add_argument(
        "--source-graph-storage",
        help="Override detected source graph storage type",
    )
    parser.add_argument(
        "--source-doc-status-storage",
        help="Override detected source doc status storage type",
    )
    parser.add_argument(
        "--write-switch-env-file",
        help="Optional output file path for PostgreSQL storage env vars",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and count source data without writing to PostgreSQL",
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
        "Migrating %d workspace(s) from %s to PostgreSQL",
        len(selected_workspaces),
        args.working_dir,
    )
    if args.inferred_embedding_model_name is None:
        logger.warning(
            "Could not infer EMBEDDING_MODEL from current config; vector data will "
            "be written to PostgreSQL legacy base tables without model suffix."
        )
    else:
        logger.info(
            "Using configured embedding model for PostgreSQL vector table suffix: %s",
            args.inferred_embedding_model_name,
        )

    if args.write_switch_env_file:
        write_switch_env_file(Path(args.write_switch_env_file))
        logger.info("Wrote PostgreSQL storage env file: %s", args.write_switch_env_file)

    for workspace in selected_workspaces:
        await migrate_workspace(workspace, args)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

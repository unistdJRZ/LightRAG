from __future__ import annotations

import asyncio
import uuid
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from lightrag.utils import EmbeddingFunc, Tokenizer, make_relation_chunk_key
from lightrag.utils_graph import acreate_entity, acreate_relation


class _FakeNanoVectorDB:
    def __init__(self, embedding_dim: int, storage_file: str | None = None):
        self.embedding_dim = embedding_dim
        self.storage_file = storage_file
        self._NanoVectorDB__storage = {"data": []}

    def __len__(self) -> int:
        return len(self._NanoVectorDB__storage["data"])

    def upsert(self, datas: list[dict[str, object]]) -> list[str]:
        existing = {
            item["__id__"]: index
            for index, item in enumerate(self._NanoVectorDB__storage["data"])
        }
        for data in datas:
            item_id = data["__id__"]
            if item_id in existing:
                self._NanoVectorDB__storage["data"][existing[item_id]] = data
            else:
                self._NanoVectorDB__storage["data"].append(data)
        return [data["__id__"] for data in datas]

    def query(
        self, query: list[float], top_k: int, better_than_threshold: float = 0.2
    ) -> list[dict[str, object]]:
        query_vector = np.array(query, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        results: list[dict[str, object]] = []
        for data in self._NanoVectorDB__storage["data"]:
            vector = np.array(data["__vector__"], dtype=np.float32)
            denominator = float(np.linalg.norm(vector)) * query_norm
            similarity = (
                float(np.dot(vector, query_vector) / denominator)
                if denominator
                else 0.0
            )
            if similarity >= better_than_threshold:
                results.append({**data, "__metrics__": similarity})
        results.sort(key=lambda item: item["__metrics__"], reverse=True)
        return results[:top_k]

    def get(self, ids: list[str]) -> list[dict[str, object]]:
        requested = set(ids)
        return [
            data
            for data in self._NanoVectorDB__storage["data"]
            if data["__id__"] in requested
        ]

    def delete(self, ids: list[str]) -> None:
        requested = set(ids)
        self._NanoVectorDB__storage["data"] = [
            data
            for data in self._NanoVectorDB__storage["data"]
            if data["__id__"] not in requested
        ]

    def save(self) -> None:
        return None


_fake_nano_module = types.ModuleType("nano_vectordb")
_fake_nano_module.NanoVectorDB = _FakeNanoVectorDB
sys.modules.setdefault("nano_vectordb", _fake_nano_module)

from lightrag import LightRAG  # noqa: E402


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _build_embedding(text: str) -> list[float]:
    if "Alpha 100" in text or "Alpha-100 Device" in text:
        return [1.0, 0.0, 0.0]
    if "Sensor Hub" in text:
        return [0.0, 1.0, 0.0]
    if "SKIP-900" in text or "SKIP 900" in text:
        return [0.0, 0.0, 1.0]
    return [0.1, 0.1, 0.1]


async def _mock_embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    return np.array([_build_embedding(text) for text in texts], dtype=np.float32)


async def _mock_llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    **kwargs,
) -> str:
    if "Embedding Similarity Score" in prompt:
        if "Alpha 100" in prompt and "Alpha-100 Device" in prompt:
            return (
                '{"same_entity": true, "reason": "Same device naming variant", '
                '"confidence": 0.98}'
            )
        return '{"same_entity": false, "reason": "Different entity", "confidence": 0.05}'

    if "Entity Group:" in prompt and "Alpha 100" in prompt and "Alpha-100 Device" in prompt:
        return (
            '{"entity_name": "Alpha-100 Device", '
            '"description": "Alpha-100 Device is the canonical device entity merged from the Alpha 100 naming variants."}'
        )

    return '{"entity_name": "Unknown", "description": "Unknown"}'


@pytest.mark.offline
def test_merge_similar_entities_merges_workspace_group_and_preserves_edges():
    async def _run() -> None:
        test_dir = Path(f"temp/test_entity_merge_similarity_{uuid.uuid4().hex}")
        test_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = Tokenizer("mock-tokenizer", _SimpleTokenizerImpl())
        rag = LightRAG(
            working_dir=str(test_dir),
            workspace="merge_test",
            llm_model_func=_mock_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=3,
                max_token_size=256,
                func=_mock_embedding_func,
                model_name="merge-test",
            ),
            tokenizer=tokenizer,
        )

        await rag.initialize_storages()
        try:
            await acreate_entity(
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                "Alpha 100",
                {
                    "entity_type": "DEVICE",
                    "description": "Alpha 100 is a device model.",
                    "source_id": "chunk-a",
                },
                rag.entity_chunks,
                rag.relation_chunks,
            )
            await acreate_entity(
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                "Alpha-100 Device",
                {
                    "entity_type": "DEVICE",
                    "description": "Alpha-100 Device is the same device written with a hyphen.",
                    "source_id": "chunk-b",
                },
                rag.entity_chunks,
                rag.relation_chunks,
            )
            await acreate_entity(
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                "Sensor Hub",
                {
                    "entity_type": "DEVICE",
                    "description": "Sensor Hub is connected to field devices.",
                    "source_id": "chunk-c",
                },
                rag.entity_chunks,
                rag.relation_chunks,
            )
            await acreate_entity(
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                "SKIP-900",
                {
                    "entity_type": "SPECIAL",
                    "description": "SKIP-900 should remain isolated.",
                    "source_id": "chunk-d",
                },
                rag.entity_chunks,
                rag.relation_chunks,
            )
            await acreate_entity(
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                "SKIP 900",
                {
                    "entity_type": "SPECIAL",
                    "description": "SKIP 900 is similar but should be skipped by type whitelist.",
                    "source_id": "chunk-e",
                },
                rag.entity_chunks,
                rag.relation_chunks,
            )
            await acreate_relation(
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                "Alpha 100",
                "Sensor Hub",
                {
                    "description": "Alpha 100 reports telemetry to Sensor Hub.",
                    "keywords": "telemetry",
                    "source_id": "rel-chunk-1",
                },
                rag.relation_chunks,
            )

            merge_result = await rag.amerge_similar_entities(
                type_whitelist=["SPECIAL"]
            )

            assert merge_result["merged_groups"] == 1
            assert merge_result["merged_entities"] == 1

            merged_entity = await rag.get_entity_info(
                "Alpha-100 Device", include_vector_data=True
            )
            assert merged_entity["graph_data"] is not None
            assert (
                merged_entity["graph_data"]["description"]
                == "Alpha-100 Device is the canonical device entity merged from the Alpha 100 naming variants."
            )

            old_entity = await rag.get_entity_info("Alpha 100", include_vector_data=True)
            assert old_entity["graph_data"] is None

            edge_data = await rag.chunk_entity_relation_graph.get_edge(
                "Alpha-100 Device", "Sensor Hub"
            )
            assert edge_data is not None
            assert (
                edge_data["description"]
                == "Alpha 100 reports telemetry to Sensor Hub."
            )

            entity_chunks = await rag.entity_chunks.get_by_id("Alpha-100 Device")
            assert entity_chunks is not None
            assert entity_chunks["chunk_ids"] == ["chunk-a", "chunk-b"]

            relation_chunks = await rag.relation_chunks.get_by_id(
                make_relation_chunk_key("Alpha-100 Device", "Sensor Hub")
            )
            assert relation_chunks is not None
            assert relation_chunks["chunk_ids"] == ["rel-chunk-1"]

            skip_a = await rag.get_entity_info("SKIP-900")
            skip_b = await rag.get_entity_info("SKIP 900")
            assert skip_a["graph_data"] is not None
            assert skip_b["graph_data"] is not None
        finally:
            await rag.finalize_storages()
            shutil.rmtree(test_dir, ignore_errors=True)

    asyncio.run(_run())

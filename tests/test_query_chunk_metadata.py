import asyncio

from lightrag.operate import _merge_all_chunks
from lightrag.utils import generate_reference_list_from_chunks


def test_merge_all_chunks_preserves_page_id_and_bbox(monkeypatch):
    async def _fake_entity_chunks(*args, **kwargs):
        return [
            {
                "content": "entity chunk",
                "file_path": "/tmp/entity.pdf",
                "full_doc_id": "doc-entity",
                "page_id": 2,
                "bbox": [20.0, 21.0, 22.0, 23.0],
                "chunk_id": "chunk-entity",
            }
        ]

    async def _fake_relation_chunks(*args, **kwargs):
        return [
            {
                "content": "relation chunk",
                "file_path": "/tmp/relation.pdf",
                "full_doc_id": "doc-relation",
                "page_id": 3,
                "bbox": [30.0, 31.0, 32.0, 33.0],
                "chunk_id": "chunk-relation",
            }
        ]

    monkeypatch.setattr(
        "lightrag.operate._find_related_text_unit_from_entities",
        _fake_entity_chunks,
    )
    monkeypatch.setattr(
        "lightrag.operate._find_related_text_unit_from_relations",
        _fake_relation_chunks,
    )

    merged_chunks = asyncio.run(
        _merge_all_chunks(
            filtered_entities=[{"entity_name": "Entity"}],
            filtered_relations=[{"src_id": "A", "tgt_id": "B"}],
            vector_chunks=[
                {
                    "content": "vector chunk",
                    "file_path": "/tmp/vector.pdf",
                    "full_doc_id": "doc-vector",
                    "page_id": 1,
                    "bbox": [10.0, 11.0, 12.0, 13.0],
                    "chunk_id": "chunk-vector",
                }
            ],
            query="test query",
            knowledge_graph_inst=object(),
            text_chunks_db=object(),
            query_param=object(),
            chunks_vdb=object(),
        )
    )

    reference_entries, _ = asyncio.run(generate_reference_list_from_chunks(merged_chunks))

    assert merged_chunks == [
        {
            "content": "vector chunk",
            "file_path": "/tmp/vector.pdf",
            "full_doc_id": "doc-vector",
            "page_id": 1,
            "bbox": [10.0, 11.0, 12.0, 13.0],
            "chunk_id": "chunk-vector",
        },
        {
            "content": "entity chunk",
            "file_path": "/tmp/entity.pdf",
            "full_doc_id": "doc-entity",
            "page_id": 2,
            "bbox": [20.0, 21.0, 22.0, 23.0],
            "chunk_id": "chunk-entity",
        },
        {
            "content": "relation chunk",
            "file_path": "/tmp/relation.pdf",
            "full_doc_id": "doc-relation",
            "page_id": 3,
            "bbox": [30.0, 31.0, 32.0, 33.0],
            "chunk_id": "chunk-relation",
        },
    ]
    assert reference_entries == [
        {
            "reference_id": "1",
            "chunk_id": "chunk-vector",
            "file_path": "/tmp/vector.pdf",
            "page_id": 1,
            "bbox": [10.0, 11.0, 12.0, 13.0],
            "file_id": None,
        },
        {
            "reference_id": "2",
            "chunk_id": "chunk-entity",
            "file_path": "/tmp/entity.pdf",
            "page_id": 2,
            "bbox": [20.0, 21.0, 22.0, 23.0],
            "file_id": None,
        },
        {
            "reference_id": "3",
            "chunk_id": "chunk-relation",
            "file_path": "/tmp/relation.pdf",
            "page_id": 3,
            "bbox": [30.0, 31.0, 32.0, 33.0],
            "file_id": None,
        },
    ]

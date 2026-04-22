import asyncio

from lightrag.operate import (
    _build_chunk_reference_lookup,
    _format_preserved_entities_for_user,
    _generate_preserved_reference_list_from_chunks,
)


def test_generate_preserved_reference_list_keeps_duplicate_chunk_ids():
    references, chunks = asyncio.run(
        _generate_preserved_reference_list_from_chunks(
            [
                {
                    "chunk_id": "chunk-1",
                    "file_path": "/tmp/a.txt",
                    "content": "first copy",
                },
                {
                    "chunk_id": "chunk-1",
                    "file_path": "/tmp/a.txt",
                    "content": "second copy",
                },
            ],
            start_reference_index=3,
            doc_status_storage=None,
        )
    )

    assert [reference["reference_id"] for reference in references] == ["3", "4"]
    assert [reference["chunk_id"] for reference in references] == ["chunk-1", "chunk-1"]
    assert [chunk["reference_id"] for chunk in chunks] == ["3", "4"]
    assert [chunk["content"] for chunk in chunks] == ["first copy", "second copy"]


def test_format_preserved_entities_resolves_related_chunk_from_combined_lookup():
    chunk_lookup = _build_chunk_reference_lookup(
        [
            {"chunk_id": "chunk-rag-1", "reference_id": "1"},
            {"chunk_id": "chunk-agent-1", "reference_id": "2"},
        ]
    )

    formatted_entities = _format_preserved_entities_for_user(
        [
            {
                "entity_name": "AgentEntity",
                "entity_type": "concept",
                "description": "selected by agent",
                "source_id": "chunk-agent-1<SEP>chunk-rag-1",
                "file_path": "/tmp/entity.txt",
            }
        ],
        chunk_lookup,
    )

    assert len(formatted_entities) == 1
    assert formatted_entities[0]["entity_name"] == "AgentEntity"
    assert formatted_entities[0]["related_chunk"] == "2"

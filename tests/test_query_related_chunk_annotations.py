from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import convert_to_user_format, truncate_source_id_for_display


def test_convert_to_user_format_adds_related_chunk_reference_ids():
    result = convert_to_user_format(
        entities_context=[
            {
                "entity": "Entity A",
                "type": "PERSON",
                "description": "desc-a",
                "source_id": f"chunk-entity{GRAPH_FIELD_SEP}chunk-entity-missing",
                "file_path": "entity.txt",
                "created_at": "2026-04-01T00:00:00Z",
            },
            {
                "entity": "Entity B",
                "type": "ORG",
                "description": "desc-b",
                "source_id": "chunk-missing",
                "file_path": "entity.txt",
                "created_at": "2026-04-01T00:00:00Z",
            },
        ],
        relations_context=[
            {
                "entity1": "Entity A",
                "entity2": "Entity B",
                "description": "rel-desc",
                "keywords": "kw",
                "weight": 1.0,
                "source_id": f"chunk-relation{GRAPH_FIELD_SEP}chunk-relation-missing",
                "file_path": "relation.txt",
                "created_at": "2026-04-01T00:00:00Z",
            },
            {
                "entity1": "Entity B",
                "entity2": "Entity C",
                "description": "rel-desc-2",
                "keywords": "kw2",
                "weight": 1.0,
                "source_id": "chunk-missing",
                "file_path": "relation.txt",
                "created_at": "2026-04-01T00:00:00Z",
            },
        ],
        chunks=[
            {
                "reference_id": "2",
                "chunk_id": "chunk-entity",
                "content": "entity chunk",
                "file_path": "entity.txt",
            },
            {
                "reference_id": "5",
                "chunk_id": "chunk-relation",
                "content": "relation chunk",
                "file_path": "relation.txt",
            },
        ],
        references=[],
        query_mode="hybrid",
    )

    entities = result["data"]["entities"]
    relationships = result["data"]["relationships"]

    assert entities[0]["related_chunk"] == "2"
    assert entities[1]["related_chunk"] == "本实体 仅用于知识补充，禁止引用。"
    assert relationships[0]["related_chunk"] == "5"
    assert relationships[1]["related_chunk"] == "本边 仅用于知识补充，禁止引用。"


def test_convert_to_user_format_uses_original_source_ids_for_related_chunk():
    result = convert_to_user_format(
        entities_context=[{"entity": "Entity A"}],
        relations_context=[{"entity1": "Entity A", "entity2": "Entity B"}],
        chunks=[
            {
                "reference_id": "7",
                "chunk_id": "chunk-original",
                "content": "original chunk",
                "file_path": "source.txt",
            }
        ],
        references=[],
        query_mode="local",
        entity_id_to_original={
            "Entity A": {
                "entity_name": "Entity A",
                "entity_type": "PERSON",
                "description": "desc",
                "source_id": "chunk-original",
                "file_path": "source.txt",
                "created_at": "2026-04-01T00:00:00Z",
            }
        },
        relation_id_to_original={
            ("Entity A", "Entity B"): {
                "src_id": "Entity A",
                "tgt_id": "Entity B",
                "description": "rel",
                "keywords": "kw",
                "weight": 1.0,
                "source_id": "chunk-original",
                "file_path": "source.txt",
                "created_at": "2026-04-01T00:00:00Z",
            }
        },
    )

    assert result["data"]["entities"][0]["related_chunk"] == "7"
    assert result["data"]["relationships"][0]["related_chunk"] == "7"


def test_convert_to_user_format_limits_displayed_source_ids_to_ten():
    source_ids = [f"chunk-{index}" for index in range(12)]
    long_source_id = GRAPH_FIELD_SEP.join(source_ids)

    result = convert_to_user_format(
        entities_context=[{"entity": "Entity A", "source_id": long_source_id}],
        relations_context=[
            {
                "entity1": "Entity A",
                "entity2": "Entity B",
                "source_id": long_source_id,
            }
        ],
        chunks=[
            {
                "reference_id": "11",
                "chunk_id": "chunk-10",
                "content": "hidden by display truncation but still resolvable",
            }
        ],
        references=[],
        query_mode="mix",
    )

    expected_display = GRAPH_FIELD_SEP.join(source_ids[:10])
    assert result["data"]["entities"][0]["source_id"] == expected_display
    assert result["data"]["relationships"][0]["source_id"] == expected_display
    assert result["data"]["entities"][0]["related_chunk"] == "11"
    assert result["data"]["relationships"][0]["related_chunk"] == "11"


def test_truncate_source_id_for_display_leaves_short_lists_unchanged():
    source_id = GRAPH_FIELD_SEP.join([f"chunk-{index}" for index in range(10)])

    assert truncate_source_id_for_display(source_id) == source_id

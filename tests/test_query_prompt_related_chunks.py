import asyncio

from lightrag.base import QueryParam
from lightrag.operate import _build_context_str


class _DummyTokenizer:
    def encode(self, text: str) -> list[str]:
        return list(text)


def test_build_context_str_includes_related_chunk_in_prompt(monkeypatch):
    async def _fake_process_chunks_unified(
        query,
        unique_chunks,
        query_param,
        global_config,
        source_type="mixed",
        chunk_token_limit=None,
    ):
        return [chunk.copy() for chunk in unique_chunks]

    monkeypatch.setattr(
        "lightrag.operate.process_chunks_unified",
        _fake_process_chunks_unified,
    )

    context, raw_data, response_context = asyncio.run(
        _build_context_str(
            entities_context=[
                {
                    "entity": "Entity A",
                    "type": "PERSON",
                    "description": "desc-a",
                    "source_id": "chunk-entity",
                    "file_path": "entity.txt",
                },
                {
                    "entity": "Entity Missing",
                    "type": "ORG",
                    "description": "desc-missing",
                    "source_id": "chunk-entity-missing",
                    "file_path": "entity.txt",
                }
            ],
            relations_context=[
                {
                    "entity1": "Entity A",
                    "entity2": "Entity B",
                    "description": "rel-desc",
                    "keywords": "kw",
                    "weight": 1.0,
                    "source_id": "chunk-relation",
                    "file_path": "relation.txt",
                },
                {
                    "entity1": "Entity Missing",
                    "entity2": "Entity Missing 2",
                    "description": "rel-missing",
                    "keywords": "kw-missing",
                    "weight": 1.0,
                    "source_id": "chunk-relation-missing",
                    "file_path": "relation.txt",
                }
            ],
            merged_chunks=[
                {
                    "chunk_id": "chunk-entity",
                    "content": "entity chunk",
                    "file_path": "entity.txt",
                },
                {
                    "chunk_id": "chunk-relation",
                    "content": "relation chunk",
                    "file_path": "relation.txt",
                },
            ],
            query="test query",
            query_param=QueryParam(
                mode="mix",
                only_need_prompt=True,
                enable_rerank=False,
                max_total_tokens=10000,
            ),
            global_config={
                "tokenizer": _DummyTokenizer(),
                "system_prompt_template": "{context_data}\n{response_type}\n{user_prompt}",
            },
        )
    )

    assert '"entity": "Entity A"' in context
    assert '"entity": "Entity A"' in response_context
    assert '"entity1": "Entity A"' in context
    assert '"related_chunk": "1"' in context
    assert '"related_chunk": "2"' in context
    assert "本实体 仅用于知识补充，禁止引用。" in context
    assert "本边 仅用于知识补充，禁止引用。" in context
    assert raw_data["data"]["entities"][0]["related_chunk"] == "1"
    assert raw_data["data"]["relationships"][0]["related_chunk"] == "2"
    assert (
        raw_data["data"]["entities"][1]["related_chunk"]
        == "本实体 仅用于知识补充，禁止引用。"
    )
    assert (
        raw_data["data"]["relationships"][1]["related_chunk"]
        == "本边 仅用于知识补充，禁止引用。"
    )

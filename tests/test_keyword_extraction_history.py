import asyncio

from lightrag.base import QueryParam
from lightrag.operate import extract_keywords_only, sanitize_history_messages_for_llm


class _DummyTokenizer:
    def encode(self, text: str) -> list[str]:
        return list(text)


class _DummyHashingKV:
    def __init__(self):
        self.global_config = {"enable_llm_cache": False}

    async def get_by_id(self, key: str):
        return None


def test_extract_keywords_only_includes_history_in_prompt():
    captured = {}

    async def _fake_model(prompt: str, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return '{"high_level_keywords":["LightRAG"],"low_level_keywords":["keyword extraction"]}'

    result = asyncio.run(
        extract_keywords_only(
            "How does it work?",
            QueryParam(
                mode="mix",
                model_func=_fake_model,
                conversation_history=[
                    {"role": "user", "content": "What is LightRAG?"},
                    {
                        "role": "assistant",
                        "content": "LightRAG is a graph-based retrieval-augmented generation system.",
                    },
                ],
            ),
            {
                "addon_params": {"language": "English"},
                "tokenizer": _DummyTokenizer(),
            },
            hashing_kv=_DummyHashingKV(),
        )
    )

    assert result == (["LightRAG"], ["keyword extraction"])
    assert "Conversation History:" in captured["prompt"]
    assert '"role": "user"' in captured["prompt"]
    assert "What is LightRAG?" in captured["prompt"]
    assert "Latest User Message:" in captured["prompt"]
    assert "How does it work?" in captured["prompt"]
    assert captured["kwargs"]["keyword_extraction"] is True


def test_extract_keywords_only_handles_empty_history():
    captured = {}

    async def _fake_model(prompt: str, **kwargs):
        captured["prompt"] = prompt
        return '{"high_level_keywords":[],"low_level_keywords":["standalone query"]}'

    result = asyncio.run(
        extract_keywords_only(
            "Standalone query",
            QueryParam(
                mode="mix",
                model_func=_fake_model,
                conversation_history=[],
            ),
            {
                "addon_params": {"language": "English"},
                "tokenizer": _DummyTokenizer(),
            },
            hashing_kv=_DummyHashingKV(),
        )
    )

    assert result == ([], ["standalone query"])
    assert "Conversation History:" in captured["prompt"]
    assert "[]" in captured["prompt"]
    assert "Latest User Message:" in captured["prompt"]
    assert "Standalone query" in captured["prompt"]


def test_extract_keywords_only_serializes_history_references():
    captured = {}

    async def _fake_model(prompt: str, **kwargs):
        captured["prompt"] = prompt
        return '{"high_level_keywords":["LightRAG"],"low_level_keywords":["chunk reference"]}'

    result = asyncio.run(
        extract_keywords_only(
            "Explain the previous sources",
            QueryParam(
                mode="mix",
                model_func=_fake_model,
                conversation_history=[
                    {
                        "role": "assistant",
                        "content": "The previous answer used two supporting chunks.",
                        "references": [
                            {
                                "reference_id": "1",
                                "chunk_id": "chunk-1",
                                "workspace": "default",
                                "ignored_field": "ignored",
                            }
                        ],
                    }
                ],
            ),
            {
                "addon_params": {"language": "English"},
                "tokenizer": _DummyTokenizer(),
            },
            hashing_kv=_DummyHashingKV(),
        )
    )

    assert result == (["LightRAG"], ["chunk reference"])
    assert '"references"' in captured["prompt"]
    assert '"reference_id": "1"' in captured["prompt"]
    assert '"chunk_id": "chunk-1"' in captured["prompt"]
    assert '"workspace": "default"' in captured["prompt"]
    assert '"ignored_field": "ignored"' in captured["prompt"]


def test_sanitize_history_messages_for_llm_drops_references():
    assert sanitize_history_messages_for_llm(
        [
            {
                "role": "assistant",
                "content": "The answer cited chunk-1.",
                "references": [
                    {
                        "reference_id": "1",
                        "chunk_id": "chunk-1",
                        "workspace": "default",
                    }
                ],
            }
        ]
    ) == [{"role": "assistant", "content": "The answer cited chunk-1."}]

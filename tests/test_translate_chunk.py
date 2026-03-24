import asyncio
import sys
import types

import pytest

ascii_colors_stub = types.ModuleType("ascii_colors")


class _ASCIIColors:
    @staticmethod
    def yellow(message: str) -> None:
        return None

    @staticmethod
    def red(message: str) -> None:
        return None

    @staticmethod
    def green(message: str) -> None:
        return None

    @staticmethod
    def cyan(message: str) -> None:
        return None


ascii_colors_stub.ASCIIColors = _ASCIIColors
sys.modules.setdefault("ascii_colors", ascii_colors_stub)

pipmaster_stub = types.ModuleType("pipmaster")
pipmaster_stub.is_installed = lambda package: True
pipmaster_stub.install = lambda package: None
sys.modules.setdefault("pipmaster", pipmaster_stub)


def _load_translate_chunk_to_cn():
    from lightrag.api.lightrag_server import translate_chunk_to_cn

    return translate_chunk_to_cn


class _FakeTextChunksStorage:
    def __init__(self, chunk_data: dict | None):
        self.chunk_data = chunk_data
        self.upserts: list[dict] = []

    async def get_by_id(self, chunk_id: str):
        if self.chunk_data is None:
            return None
        return dict(self.chunk_data)

    async def upsert(self, data: dict[str, dict]):
        self.upserts.append(data)
        for payload in data.values():
            self.chunk_data = dict(payload)


class _FakeRAG:
    def __init__(self, chunk_data: dict | None, llm_response: str = "翻译结果"):
        self.text_chunks = _FakeTextChunksStorage(chunk_data)
        self.llm_calls: list[dict] = []
        self._llm_response = llm_response

    async def llm_model_func(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        enable_cot=False,
        _priority=None,
        **kwargs,
    ):
        self.llm_calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "history_messages": history_messages,
                "enable_cot": enable_cot,
                "_priority": _priority,
            }
        )
        return self._llm_response


@pytest.mark.offline
def test_translate_chunk_returns_cached_translation():
    translate_chunk_to_cn = _load_translate_chunk_to_cn()
    rag = _FakeRAG(
        {
            "content": "original text",
            "translated_cn": "已翻译内容",
            "tokens": 1,
            "chunk_order_index": 0,
            "full_doc_id": "doc-1",
            "file_path": "doc.txt",
            "llm_cache_list": [],
        }
    )

    translated_cn, cached = asyncio.run(translate_chunk_to_cn(rag, "chunk-1"))

    assert translated_cn == "已翻译内容"
    assert cached is True
    assert rag.llm_calls == []
    assert rag.text_chunks.upserts == []


@pytest.mark.offline
def test_translate_chunk_generates_and_persists_translation():
    translate_chunk_to_cn = _load_translate_chunk_to_cn()
    rag = _FakeRAG(
        {
            "content": "This is the source chunk.",
            "tokens": 6,
            "chunk_order_index": 0,
            "full_doc_id": "doc-1",
            "file_path": "doc.txt",
            "llm_cache_list": [],
        },
        llm_response="<think>reasoning</think>这是中文翻译。",
    )

    translated_cn, cached = asyncio.run(translate_chunk_to_cn(rag, "chunk-1"))

    assert translated_cn == "这是中文翻译。"
    assert cached is False
    assert len(rag.llm_calls) == 1
    assert rag.llm_calls[0]["prompt"] == "This is the source chunk."
    assert rag.text_chunks.upserts == [
        {
            "chunk-1": {
                "content": "This is the source chunk.",
                "tokens": 6,
                "chunk_order_index": 0,
                "full_doc_id": "doc-1",
                "file_path": "doc.txt",
                "llm_cache_list": [],
                "translated_cn": "这是中文翻译。",
            }
        }
    ]


@pytest.mark.offline
def test_translate_chunk_rejects_image_chunk():
    translate_chunk_to_cn = _load_translate_chunk_to_cn()
    rag = _FakeRAG(
        {
            "content": "YmFzZTY0",
            "content_type": "image",
            "tokens": 0,
            "chunk_order_index": 0,
            "full_doc_id": "doc-1",
            "file_path": "doc.txt",
            "llm_cache_list": [],
        }
    )

    with pytest.raises(ValueError, match="Image chunks do not support translation"):
        asyncio.run(translate_chunk_to_cn(rag, "chunk-1"))

    assert rag.llm_calls == []
    assert rag.text_chunks.upserts == []

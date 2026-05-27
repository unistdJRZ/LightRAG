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
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        from lightrag.api.lightrag_server import translate_chunk_to_cn
    finally:
        sys.argv = argv

    return translate_chunk_to_cn


def _load_chunk_file_id_helpers():
    argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        from lightrag.api.lightrag_server import (
            _extract_file_id_from_doc_status,
            _resolve_chunk_file_id,
        )
    finally:
        sys.argv = argv

    return _extract_file_id_from_doc_status, _resolve_chunk_file_id


class _FakeDocStatusStorage:
    def __init__(self):
        self.by_id: dict[str, dict] = {}
        self.by_file_path: dict[str, dict] = {}

    async def get_by_id(self, doc_id: str):
        return self.by_id.get(doc_id)

    async def get_doc_by_file_path(self, file_path: str):
        return self.by_file_path.get(file_path)


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
        self.doc_status = _FakeDocStatusStorage()
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
def test_chunk_file_id_helpers_resolve_meta_info_and_file_path_fallback():
    extract_file_id, resolve_chunk_file_id = _load_chunk_file_id_helpers()
    assert (
        extract_file_id({"metadata": {"meta_info": {"file_id": "file-meta"}}})
        == "file-meta"
    )
    assert extract_file_id({"metadata": {"file_id": "file-direct"}}) == "file-direct"

    rag = _FakeRAG({"content": "chunk"})
    rag.doc_status.by_id["doc-1"] = {
        "metadata": {"meta_info": {"file_id": "file-from-doc"}}
    }
    assert (
        asyncio.run(
            resolve_chunk_file_id(
                rag,
                {"full_doc_id": "doc-1", "file_path": "/tmp/doc.txt"},
            )
        )
        == "file-from-doc"
    )

    rag.doc_status.by_id.clear()
    rag.doc_status.by_file_path["/tmp/doc.txt"] = {
        "metadata": {"file_id": "file-from-path"}
    }
    assert (
        asyncio.run(
            resolve_chunk_file_id(
                rag,
                {"full_doc_id": "missing-doc", "file_path": "/tmp/doc.txt"},
            )
        )
        == "file-from-path"
    )


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

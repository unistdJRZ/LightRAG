import json
import asyncio

from lightrag.qa_extraction import (
    _call_json_llm,
    _extract_json_object,
    _split_text_for_local_qa,
    build_doc_qa_vector_data,
    extract_doc_qa_pairs,
    replace_doc_qa_pairs,
)
from lightrag.knowledge_base_qa import (
    build_knowledge_base_qa_vector_data,
    query_knowledge_base_qa_by_rule,
    upsert_knowledge_base_qa_rows,
)


def test_extract_json_object_from_fenced_response():
    result = _extract_json_object(
        """```json
        {"final_qas": [{"question": "What is tested?", "answer": "JSON parsing."}]}
        ```"""
    )

    assert result["final_qas"][0]["answer"] == "JSON parsing."


def test_split_text_for_local_qa_uses_sequential_windows():
    chunks = _split_text_for_local_qa("a" * 16001, max_chars=8000)

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "qa-bchunk-000001",
        "qa-bchunk-000002",
        "qa-bchunk-000003",
    ]
    assert [(chunk["char_start"], chunk["char_end"]) for chunk in chunks] == [
        (0, 8000),
        (8000, 16000),
        (16000, 16001),
    ]
    assert [len(chunk["content"]) for chunk in chunks] == [8000, 8000, 1]


def test_call_json_llm_retries_incomplete_json(monkeypatch):
    monkeypatch.setenv("QA_EXTRACT_JSON_RETRY_ATTEMPTS", "3")
    calls = 0

    async def fake_llm(_prompt):
        nonlocal calls
        calls += 1
        if calls < 3:
            return '{"final_qas": [{"question": "broken"'
        return '{"final_qas": [{"question": "ok", "answer": "done"}]}'

    result = asyncio.run(_call_json_llm(fake_llm, "prompt", stage="test"))

    assert calls == 3
    assert result["final_qas"][0]["answer"] == "done"


def test_extract_doc_qa_pairs_skips_failed_local_chunk(monkeypatch):
    monkeypatch.setenv("QA_EXTRACT_JSON_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("QA_EXTRACT_B_CHUNK_MAX_CHARS", "5")
    bad_local_calls = 0

    async def fake_llm(prompt):
        nonlocal bad_local_calls
        if "# Input Document" in prompt:
            return json.dumps(
                {
                    "document_identity": {"title": "Example"},
                    "global_profile": {"global_summary": "Profile"},
                    "global_qas": [],
                }
            )
        if "all_qas:" in prompt:
            return json.dumps(
                {
                    "statistics": {
                        "input_total_qas": 1,
                        "kept_qas": 1,
                        "removed_qas": 0,
                        "merged_qas": 0,
                    },
                    "dedup_summary": {
                        "main_duplicate_patterns": [],
                        "main_removal_reasons": [],
                        "coverage_notes": "ok",
                    },
                    "final_qas": [
                        {
                            "question": "What is in the good chunk?",
                            "answer": "Useful information.",
                            "question_type": "definition",
                            "scope": "local",
                            "source_preference": "qa-bchunk-000002",
                            "edit_action": "kept",
                            "quality_score": 0.8,
                        }
                    ],
                    "removed_or_merged_items": [],
                }
            )
        if "local_chunk:\naaaaa" in prompt:
            bad_local_calls += 1
            return '{"qas": [{"question": "broken"'
        if "local_chunk:\nbbbbb" in prompt:
            return json.dumps(
                {
                    "chunk_assessment": {
                        "is_suitable_for_qa_extraction": True,
                    },
                    "local_information_points": [],
                    "qas": [
                        {
                            "question": "What is in the good chunk?",
                            "answer": "Useful information.",
                            "question_type": "definition",
                            "quality_score": 0.8,
                        }
                    ],
                    "rejected_question_candidates": [],
                }
            )
        raise AssertionError("unexpected prompt")

    result = asyncio.run(
        extract_doc_qa_pairs(
            doc_id="doc-1",
            document_text="aaaaabbbbb",
            llm_model_func=fake_llm,
        )
    )

    assert bad_local_calls == 3
    assert len(result["local_results"]) == 2
    assert result["local_results"][0]["qas"] == []
    assert "qa_extraction_error" in result["local_results"][0]
    assert result["local_results"][1]["qas"][0]["answer"] == "Useful information."
    assert result["final_qas"][0]["answer"] == "Useful information."


def test_extract_doc_qa_pairs_falls_back_when_consolidation_fails(monkeypatch):
    monkeypatch.setenv("QA_EXTRACT_JSON_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("QA_EXTRACT_B_CHUNK_MAX_CHARS", "20")

    async def fake_llm(prompt):
        if "# Input Document" in prompt:
            return json.dumps(
                {
                    "document_identity": {"title": "Example"},
                    "global_profile": {"global_summary": "Profile"},
                    "global_qas": [
                        {
                            "question": "What is the document about?",
                            "answer": "A fallback test.",
                            "question_type": "summary",
                            "confidence": 0.7,
                        }
                    ],
                }
            )
        if "all_qas:" in prompt:
            return '{"final_qas": [{"question": "broken"'
        return json.dumps(
            {
                "chunk_assessment": {"is_suitable_for_qa_extraction": True},
                "local_information_points": [],
                "qas": [
                    {
                        "question": "What does the chunk state?",
                        "answer": "It has local evidence.",
                        "question_type": "result",
                        "quality_score": 0.9,
                    }
                ],
                "rejected_question_candidates": [],
            }
        )

    result = asyncio.run(
        extract_doc_qa_pairs(
            doc_id="doc-1",
            document_text="local evidence",
            llm_model_func=fake_llm,
        )
    )

    assert result["consolidated_result"]["qa_extraction_error"]
    assert [qa["edit_action"] for qa in result["final_qas"]] == [
        "fallback_kept",
        "fallback_kept",
    ]
    assert {qa["scope"] for qa in result["final_qas"]} == {"global", "local"}


def test_build_doc_qa_vector_data_uses_question_content_and_qa_id():
    extraction_result = {
        "final_qas": [
            {
                "question": "What is indexed in the QA vector store?",
                "answer": "Only the question text is embedded.",
            }
        ]
    }

    vector_data = build_doc_qa_vector_data(
        doc_id="doc-1",
        extraction_result=extraction_result,
    )

    assert len(vector_data) == 1
    qa_id, payload = next(iter(vector_data.items()))
    assert qa_id.startswith("qa-")
    assert payload == {
        "content": "What is indexed in the QA vector store?",
        "doc_id": "doc-1",
        "qa_id": qa_id,
    }


def test_replace_doc_qa_pairs_inserts_final_qas():
    class FakeDB:
        def __init__(self):
            self.calls = []

        async def execute(self, sql, data=None):
            self.calls.append((sql, data))

    db = FakeDB()
    extraction_result = {
        "global_result": {
            "document_identity": {"title": "Example"},
            "global_profile": {"global_summary": "Summary"},
        },
        "consolidated_result": {
            "statistics": {"input_total_qas": 1},
            "dedup_summary": {"coverage_notes": "ok"},
        },
        "final_qas": [
            {
                "question": "What is LightRAG?",
                "answer": "A RAG framework.",
                "question_type": "definition",
                "scope": "global",
                "source_preference": "global_qa",
                "edit_action": "kept",
                "quality_score": 0.9,
            }
        ],
    }

    count = asyncio.run(
        replace_doc_qa_pairs(
            db=db,
            workspace="default",
            doc_id="doc-1",
            extraction_result=extraction_result,
        )
    )

    assert count == 1
    insert_call = db.calls[-1]
    assert "INSERT INTO LIGHTRAG_DOC_QA_PAIRS" in insert_call[0]
    assert insert_call[1]["doc_id"] == "doc-1"
    assert insert_call[1]["question"] == "What is LightRAG?"
    metadata = json.loads(insert_call[1]["metadata"])
    assert metadata["document_identity"]["title"] == "Example"


def test_build_knowledge_base_qa_vector_data_uses_question_content():
    vector_data = build_knowledge_base_qa_vector_data(
        [
            {
                "id": "kbqa-1",
                "question": "How many files are in this knowledge base?",
                "answer": "There are 10 files.",
            }
        ]
    )

    assert vector_data == {
        "kbqa-1": {
            "content": "How many files are in this knowledge base?",
            "kbqa_id": "kbqa-1",
        }
    }


def test_upsert_knowledge_base_qa_rows_inserts_rows():
    class FakeDB:
        def __init__(self):
            self.calls = []

        async def execute(self, sql, data=None):
            self.calls.append((sql, data))

    db = FakeDB()

    count_rows = asyncio.run(
        upsert_knowledge_base_qa_rows(
            db=db,
            workspace="default",
            qa_rows=[
                {
                    "id": "kbqa-1",
                    "question": "What is the document count?",
                    "answer": "The knowledge base has 3 documents.",
                }
            ],
        )
    )

    assert count_rows[0]["id"] == "kbqa-1"
    insert_call = db.calls[-1]
    assert "INSERT INTO LIGHTRAG_KNOWLEDGE_BASE_QA" in insert_call[0]
    assert insert_call[1]["workspace"] == "default"
    assert insert_call[1]["question"] == "What is the document count?"


def test_query_knowledge_base_qa_by_rule_uses_workspace_and_query():
    class FakeDB:
        def __init__(self):
            self.query_args = None

        async def execute(self, sql, data=None):
            return None

        async def query(self, sql, params=None, multirows=False):
            self.query_args = (sql, params, multirows)
            return [
                {
                    "id": "kbqa-1",
                    "workspace": "default",
                    "question": "What is the document count?",
                    "answer": "Three.",
                    "metadata": "{}",
                    "match_rank": 1,
                }
            ]

    db = FakeDB()

    rows = asyncio.run(
        query_knowledge_base_qa_by_rule(
            db=db,
            workspace="default",
            query="document count",
            top_k=5,
        )
    )

    assert rows[0]["id"] == "kbqa-1"
    assert rows[0]["metadata"] == {}
    assert db.query_args[1] == ["default", "document count", "%document count%", 5]

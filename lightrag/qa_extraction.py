from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

from lightrag.utils import compute_mdhash_id, logger


QA_STYLE_RULES = """
# QA Style Rules

1. Questions should be natural, clear, and self-contained.
2. Answers should be concise, factual, and directly responsive.
3. Do not copy long spans from the source unless necessary.
4. Avoid vague referents like "this", "it", "they" unless clearly grounded.
5. Prefer explicit nouns over ambiguous pronouns.
6. Avoid yes/no questions unless the answer contains substantial explanation.
7. Avoid overly broad questions that cannot be answered precisely.
8. Avoid overly narrow questions with little standalone value.
"""


GLOBAL_QA_PROMPT = """
You are an expert long-document analysis agent.

Your task is to read the provided near-full document content and produce:
1. a structured global profile for downstream local QA extraction
2. a set of document-level QA pairs that should be generated now, because later stages will no longer have access to the full document

The input may contain most or nearly all of the document, but the structure may be noisy, incomplete, inconsistent, or not strictly aligned with standard sections. You must not assume the document is perfectly structured.

# Core Objectives

## Part A: Build a Global Profile
Extract global semantic information that will help downstream local chunk-based QA generation.

## Part B: Generate Global QA Pairs
Generate high-quality QA pairs that require document-level understanding, cross-section synthesis, or full-document perspective.

# Important Constraints

1. Only use information supported by the provided content.
2. Do not fabricate missing sections, methods, results, or conclusions.
3. If the document appears incomplete or uncertain, explicitly mark uncertainty.
4. Global QA pairs should be truly document-level.
5. Do not generate local-detail QA that is better left to chunk-level extraction.
6. Answers should be concise but informative, and should not include unsupported claims.
7. If the document is not a research paper, adapt to its genre.
8. Output must be valid JSON only.

# Output Schema

{
  "document_identity": {
    "title": "",
    "document_type": "",
    "domain": "",
    "main_topic": "",
    "language": "",
    "completeness_assessment": {
      "is_near_full_document": true,
      "confidence": 0.0,
      "notes": ""
    }
  },
  "global_profile": {
    "core_problem": "",
    "core_goal": "",
    "core_claim_or_main_message": "",
    "document_background": "",
    "global_summary": "",
    "semantic_structure": [
      {
        "section_role": "",
        "description": ""
      }
    ],
    "major_topics": [
      {
        "topic": "",
        "description": "",
        "importance": "high"
      }
    ],
    "key_terms": [
      {
        "term": "",
        "definition_or_role": "",
        "aliases": []
      }
    ],
    "main_entities": [
      {
        "name": "",
        "type": "",
        "role_in_document": ""
      }
    ],
    "question_dimensions_for_local_extraction": [
      ""
    ],
    "cross_chunk_dependency_hints": [
      ""
    ],
    "global_only_question_scopes": [
      ""
    ]
  },
  "global_qas": [
    {
      "question": "",
      "answer": "",
      "question_type": "",
      "coverage_basis": "",
      "confidence": 0.0
    }
  ]
}

# Quality Requirements for global_qas

1. Each question must be meaningful and useful for retrieval or QA tasks.
2. Each answer must reflect full-document or near-full-document understanding.
3. Prefer 3 to 12 high-quality global QA pairs, depending on document richness.
4. Do not produce redundant QA pairs.
5. Do not ask multiple unrelated things in one question.
6. If evidence is partial, phrase the answer conservatively.

{style_rules}

# Input Document
{document_text}
""".strip()


LOCAL_QA_PROMPT = """
You are an expert document chunk QA extraction agent.

You are given:
1. a global profile extracted from the near-full document
2. one local chunk from that document

Your task is to perform an internal two-step process within a single response:
- Step 1: identify the chunk's role and extract its core local information points
- Step 2: generate 1 to 5 high-quality local QA pairs based on those information points

Global information is only for understanding, disambiguation, terminology resolution, identifying the role of the chunk, and avoiding redundant global-only questions. Global information must NOT replace local evidence.

# Hard Constraints

1. Answers must be mainly grounded in the local chunk.
2. Global profile can help interpret terminology, acronyms, references, and the chunk's purpose, but cannot supply missing factual content.
3. Do not generate document-level questions already covered by global-only scopes.
4. Do not generate questions that require combining multiple distant chunks to answer completely.
5. Do not generate vague, trivial, or low-value questions.
6. Prefer questions with clear information gain.
7. If the chunk contains little useful standalone content, generate fewer QA pairs, even zero if necessary.
8. Maximum QA count: 5.
9. Questions should be concise, answerable, and non-redundant.
10. Do not ask one question containing multiple unrelated sub-questions.
11. Output valid JSON only.

# Preferred Local Question Types

Prefer high-value local question types when supported by the chunk:
- definition
- background_reason
- mechanism
- component_role
- process_step
- condition_or_constraint
- experiment_setup
- metric_definition
- result
- comparison
- limitation
- application
- interpretation

# Output Schema

{
  "chunk_assessment": {
    "chunk_role_in_document": "",
    "chunk_main_function": "",
    "is_suitable_for_qa_extraction": true,
    "suitability_reason": "",
    "requires_global_context_for_interpretation": true
  },
  "local_information_points": [
    {
      "point": "",
      "point_type": "",
      "support_level": "strong"
    }
  ],
  "qas": [
    {
      "question": "",
      "answer": "",
      "question_type": "",
      "evidence_scope": "local_only",
      "uses_global_context_for_disambiguation": false,
      "quality_score": 0.0
    }
  ],
  "rejected_question_candidates": [
    {
      "candidate": "",
      "rejection_reason": ""
    }
  ]
}

{style_rules}

# Input Data

global_profile:
{global_profile}

local_chunk:
{local_chunk}
""".strip()


CONSOLIDATE_QA_PROMPT = """
You are an expert QA deduplication and consolidation agent.

You are given a full set of QA pairs extracted from a document processing pipeline, including:
- global QA pairs generated from near-full document input
- local QA pairs generated from individual chunks

Your task is to directly review the entire QA collection in a single pass and produce a cleaned, deduplicated, consolidated final QA set.

# Main Goals

1. remove redundant QA pairs
2. merge near-duplicate QA pairs when appropriate
3. remove low-quality or weakly supported QA pairs
4. preserve useful diversity of question types and topic coverage
5. prefer clearer, more informative, better-supported versions among duplicates
6. keep both global and local QA pairs when they are genuinely different in scope
7. output a final high-quality QA collection

# Output Requirements

Output valid JSON only.

{
  "statistics": {
    "input_total_qas": 0,
    "kept_qas": 0,
    "removed_qas": 0,
    "merged_qas": 0
  },
  "dedup_summary": {
    "main_duplicate_patterns": [
      ""
    ],
    "main_removal_reasons": [
      ""
    ],
    "coverage_notes": ""
  },
  "final_qas": [
    {
      "question": "",
      "answer": "",
      "question_type": "",
      "scope": "global",
      "source_preference": "",
      "edit_action": "kept",
      "quality_score": 0.0
    }
  ],
  "removed_or_merged_items": [
    {
      "original_question": "",
      "action": "removed",
      "reason": "",
      "replaced_by_question": ""
    }
  ]
}

# Decision Rules

When comparing QA pairs, prioritize stronger support, clearer wording, higher information density, better scope control, lower redundancy, and better answer completeness without overreach.

{style_rules}

# Input Data

global_profile:
{global_profile}

all_qas:
{all_qas}
""".strip()


def _max_doc_chars() -> int:
    return max(1, int(os.getenv("QA_EXTRACT_MAX_DOC_CHARS", "120000")))


def _max_b_chunk_chars() -> int:
    return max(1, int(os.getenv("QA_EXTRACT_B_CHUNK_MAX_CHARS", "8000")))


def _qa_json_retry_attempts() -> int:
    return max(1, int(os.getenv("QA_EXTRACT_JSON_RETRY_ATTEMPTS", "3")))


def _truncate_document_text(text: str) -> str:
    max_chars = _max_doc_chars()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return f"{head}\n\n[... middle content truncated for global QA extraction ...]\n\n{tail}"


def _split_text_for_local_qa(text: str, max_chars: int | None = None) -> list[dict[str, Any]]:
    """Split full document text into sequential B-stage character windows."""

    max_chars = max_chars or _max_b_chunk_chars()
    normalized_text = str(text or "")
    chunks: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(normalized_text):
        end = min(start + max_chars, len(normalized_text))
        chunk_text = normalized_text[start:end]
        if chunk_text.strip():
            chunks.append(
                {
                    "chunk_id": f"qa-bchunk-{index + 1:06d}",
                    "content": chunk_text,
                    "chunk_order_index": index,
                    "char_start": start,
                    "char_end": end,
                }
            )
            index += 1
        start = end
    return chunks


def build_doc_qa_vector_data(
    *,
    doc_id: str,
    extraction_result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build independent QA vector payloads keyed by the PostgreSQL QA row id."""

    final_qas = extraction_result.get("final_qas", [])
    if not isinstance(final_qas, list):
        final_qas = []

    vector_data: dict[str, dict[str, Any]] = {}
    for index, qa in enumerate(final_qas):
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if not question or not answer:
            continue
        qa_id = compute_mdhash_id(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "question": question,
                    "answer": answer,
                    "index": index,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            prefix="qa-",
        )
        vector_data[qa_id] = {
            "content": question,
            "doc_id": doc_id,
            "qa_id": qa_id,
        }
    return vector_data


def _extract_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("LLM response does not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("LLM JSON response must be an object")
                return parsed

    raise ValueError("LLM response contains incomplete JSON")


async def _call_json_llm(
    llm_model_func: Callable[..., Any],
    prompt: str,
    *,
    stage: str = "qa",
) -> dict[str, Any]:
    attempts = _qa_json_retry_attempts()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = await llm_model_func(prompt)
            if hasattr(response, "__aiter__"):
                parts = []
                async for part in response:
                    parts.append(str(part))
                response = "".join(parts)
            return _extract_json_object(str(response))
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            logger.warning(
                "QA JSON extraction failed for stage=%s attempt=%d/%d: %s; retrying",
                stage,
                attempt,
                attempts,
                exc,
            )

    logger.warning(
        "QA JSON extraction failed for stage=%s after %d attempts: %s",
        stage,
        attempts,
        last_error,
    )
    if last_error is not None:
        raise last_error
    raise ValueError(f"QA JSON extraction failed for stage={stage}")


def _empty_global_result(reason: str) -> dict[str, Any]:
    return {
        "document_identity": {
            "title": "",
            "document_type": "",
            "domain": "",
            "main_topic": "",
            "language": "",
            "completeness_assessment": {
                "is_near_full_document": False,
                "confidence": 0.0,
                "notes": reason,
            },
        },
        "global_profile": {
            "core_problem": "",
            "core_goal": "",
            "core_claim_or_main_message": "",
            "document_background": "",
            "global_summary": "",
            "semantic_structure": [],
            "major_topics": [],
            "key_terms": [],
            "main_entities": [],
            "question_dimensions_for_local_extraction": [],
            "cross_chunk_dependency_hints": [],
            "global_only_question_scopes": [],
        },
        "global_qas": [],
        "qa_extraction_error": reason,
    }


def _empty_local_result(
    chunk_data: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk_data.get("chunk_id", "")),
        "char_start": chunk_data.get("char_start"),
        "char_end": chunk_data.get("char_end"),
        "chunk_assessment": {
            "chunk_role_in_document": "",
            "chunk_main_function": "",
            "is_suitable_for_qa_extraction": False,
            "suitability_reason": reason,
            "requires_global_context_for_interpretation": False,
        },
        "local_information_points": [],
        "qas": [],
        "rejected_question_candidates": [],
        "qa_extraction_error": reason,
    }


def _fallback_consolidated_result(all_qas: dict[str, Any], reason: str) -> dict[str, Any]:
    final_qas: list[dict[str, Any]] = []

    for qa in all_qas.get("global_qas", []) or []:
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if not question or not answer:
            continue
        final_qas.append(
            {
                "question": question,
                "answer": answer,
                "question_type": qa.get("question_type"),
                "scope": "global",
                "source_preference": "global_qa",
                "edit_action": "fallback_kept",
                "quality_score": float(
                    qa.get("quality_score", qa.get("confidence", 0.0)) or 0.0
                ),
            }
        )

    for local_item in all_qas.get("local_qas", []) or []:
        if not isinstance(local_item, dict):
            continue
        chunk_id = local_item.get("chunk_id")
        for qa in local_item.get("qas", []) or []:
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            answer = str(qa.get("answer") or "").strip()
            if not question or not answer:
                continue
            final_qas.append(
                {
                    "question": question,
                    "answer": answer,
                    "question_type": qa.get("question_type"),
                    "scope": "local",
                    "source_preference": str(chunk_id or ""),
                    "edit_action": "fallback_kept",
                    "quality_score": float(qa.get("quality_score", 0.0) or 0.0),
                }
            )

    return {
        "statistics": {
            "input_total_qas": len(final_qas),
            "kept_qas": len(final_qas),
            "removed_qas": 0,
            "merged_qas": 0,
        },
        "dedup_summary": {
            "main_duplicate_patterns": [],
            "main_removal_reasons": [reason],
            "coverage_notes": "Consolidation failed; using valid pre-consolidation QA pairs.",
        },
        "final_qas": final_qas,
        "removed_or_merged_items": [],
        "qa_extraction_error": reason,
    }


async def extract_doc_qa_pairs(
    *,
    doc_id: str,
    document_text: str,
    chunks: dict[str, dict[str, Any]] | None = None,
    llm_model_func: Callable[..., Any],
) -> dict[str, Any]:
    """Run the A/B/C document QA extraction workflow."""

    if not document_text.strip():
        raise ValueError(f"Document {doc_id} has empty content for QA extraction")

    global_prompt = (
        GLOBAL_QA_PROMPT.replace("{style_rules}", QA_STYLE_RULES).replace(
            "{document_text}", _truncate_document_text(document_text)
        )
    )
    try:
        global_result = await _call_json_llm(
            llm_model_func,
            global_prompt,
            stage=f"global:{doc_id}",
        )
    except Exception as exc:
        logger.warning(
            "Skipping global QA extraction block for doc_id=%s after retries: %s",
            doc_id,
            exc,
        )
        global_result = _empty_global_result(str(exc))
    global_profile = {
        "document_identity": global_result.get("document_identity", {}),
        "global_profile": global_result.get("global_profile", {}),
    }

    local_results: list[dict[str, Any]] = []
    global_profile_text = json.dumps(global_profile, ensure_ascii=False)

    async def process_chunk(chunk_data: dict[str, Any]) -> dict[str, Any]:
        chunk_id = str(chunk_data["chunk_id"])
        chunk_text = str(chunk_data.get("content") or "").strip()
        if not chunk_text:
            return {"chunk_id": chunk_id, "qas": []}
        prompt = (
            LOCAL_QA_PROMPT.replace("{style_rules}", QA_STYLE_RULES)
            .replace("{global_profile}", global_profile_text)
            .replace("{local_chunk}", chunk_text)
        )
        try:
            result = await _call_json_llm(
                llm_model_func,
                prompt,
                stage=f"local:{doc_id}:{chunk_id}",
            )
        except Exception as exc:
            logger.warning(
                "Skipping local QA extraction block doc_id=%s chunk_id=%s after retries: %s",
                doc_id,
                chunk_id,
                exc,
            )
            return _empty_local_result(chunk_data, str(exc))
        result["chunk_id"] = chunk_id
        result["char_start"] = chunk_data.get("char_start")
        result["char_end"] = chunk_data.get("char_end")
        return result

    local_chunks = _split_text_for_local_qa(document_text)
    chunk_tasks = [process_chunk(chunk_data) for chunk_data in local_chunks]
    if chunk_tasks:
        local_results = await asyncio.gather(*chunk_tasks)

    all_qas = {
        "global_qas": global_result.get("global_qas", []),
        "local_qas": [
            {"chunk_id": item.get("chunk_id"), "qas": item.get("qas", [])}
            for item in local_results
        ],
    }
    consolidate_prompt = (
        CONSOLIDATE_QA_PROMPT.replace("{style_rules}", QA_STYLE_RULES)
        .replace("{global_profile}", global_profile_text)
        .replace("{all_qas}", json.dumps(all_qas, ensure_ascii=False))
    )
    try:
        consolidated = await _call_json_llm(
            llm_model_func,
            consolidate_prompt,
            stage=f"consolidate:{doc_id}",
        )
    except Exception as exc:
        logger.warning(
            "Skipping QA consolidation block for doc_id=%s after retries: %s",
            doc_id,
            exc,
        )
        consolidated = _fallback_consolidated_result(all_qas, str(exc))
    return {
        "doc_id": doc_id,
        "global_result": global_result,
        "local_results": local_results,
        "consolidated_result": consolidated,
        "final_qas": consolidated.get("final_qas", []),
    }


async def ensure_doc_qa_table(db: Any) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_QA_PAIRS (
            workspace VARCHAR(255) NOT NULL,
            id VARCHAR(255) NOT NULL,
            doc_id VARCHAR(255) NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            question_type VARCHAR(128) NULL,
            scope VARCHAR(32) NULL,
            source_preference VARCHAR(128) NULL,
            quality_score REAL NULL,
            metadata JSONB NULL DEFAULT '{}'::jsonb,
            create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT LIGHTRAG_DOC_QA_PAIRS_PK PRIMARY KEY (workspace, id)
        )
        """
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_lightrag_doc_qa_pairs_workspace_doc_id ON LIGHTRAG_DOC_QA_PAIRS(workspace, doc_id)"
    )


async def replace_doc_qa_pairs(
    *,
    db: Any,
    workspace: str,
    doc_id: str,
    extraction_result: dict[str, Any],
) -> int:
    """Replace all QA rows for a document in PostgreSQL."""

    await ensure_doc_qa_table(db)
    await db.execute(
        "DELETE FROM LIGHTRAG_DOC_QA_PAIRS WHERE workspace=$1 AND doc_id=$2",
        {"workspace": workspace, "doc_id": doc_id},
    )

    vector_data = build_doc_qa_vector_data(
        doc_id=doc_id,
        extraction_result=extraction_result,
    )
    qa_id_by_question = {
        value["content"]: qa_id for qa_id, value in vector_data.items()
    }
    final_qas = extraction_result.get("final_qas", [])
    if not isinstance(final_qas, list):
        final_qas = []

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    inserted = 0
    for index, qa in enumerate(final_qas):
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if not question or not answer:
            continue

        qa_id = qa_id_by_question.get(question)
        if not qa_id:
            continue
        metadata = {
            "edit_action": qa.get("edit_action"),
            "dedup_statistics": extraction_result.get("consolidated_result", {}).get(
                "statistics", {}
            ),
            "dedup_summary": extraction_result.get("consolidated_result", {}).get(
                "dedup_summary", {}
            ),
            "document_identity": extraction_result.get("global_result", {}).get(
                "document_identity", {}
            ),
            "global_profile": extraction_result.get("global_result", {}).get(
                "global_profile", {}
            ),
        }
        await db.execute(
            """
            INSERT INTO LIGHTRAG_DOC_QA_PAIRS(
                workspace, id, doc_id, question, answer, question_type, scope,
                source_preference, quality_score, metadata, create_time, update_time
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (workspace, id) DO UPDATE
            SET doc_id=EXCLUDED.doc_id,
                question=EXCLUDED.question,
                answer=EXCLUDED.answer,
                question_type=EXCLUDED.question_type,
                scope=EXCLUDED.scope,
                source_preference=EXCLUDED.source_preference,
                quality_score=EXCLUDED.quality_score,
                metadata=EXCLUDED.metadata,
                update_time=EXCLUDED.update_time
            """,
            {
                "workspace": workspace,
                "id": qa_id,
                "doc_id": doc_id,
                "question": question,
                "answer": answer,
                "question_type": qa.get("question_type"),
                "scope": qa.get("scope"),
                "source_preference": qa.get("source_preference"),
                "quality_score": float(qa.get("quality_score", 0.0) or 0.0),
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "create_time": now,
                "update_time": now,
            },
        )
        inserted += 1

    logger.info("Stored %d QA pairs for doc_id=%s", inserted, doc_id)
    return inserted


async def get_doc_qa_pairs_by_ids(
    *,
    db: Any,
    workspace: str,
    qa_ids: list[str],
) -> list[dict[str, Any]]:
    """Fetch full QA rows from PostgreSQL, preserving vector result order."""

    if not qa_ids:
        return []

    rows = await db.query(
        """
        SELECT id, doc_id, question, answer, question_type, scope,
               source_preference, quality_score, metadata,
               EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at,
               EXTRACT(EPOCH FROM update_time)::BIGINT AS updated_at
        FROM LIGHTRAG_DOC_QA_PAIRS
        WHERE workspace=$1 AND id = ANY($2)
        """,
        [workspace, qa_ids],
        multirows=True,
    )
    rows_by_id = {str(row["id"]): row for row in rows or []}
    ordered_rows: list[dict[str, Any]] = []
    for qa_id in qa_ids:
        row = rows_by_id.get(str(qa_id))
        if not row:
            continue
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                row["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                row["metadata"] = {}
        ordered_rows.append(row)
    return ordered_rows

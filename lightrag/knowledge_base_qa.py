from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from lightrag.utils import compute_mdhash_id, logger


def build_knowledge_base_qa_vector_data(
    qa_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build vector payloads keyed by knowledge-base QA id."""

    vector_data: dict[str, dict[str, Any]] = {}
    for row in qa_rows:
        if not isinstance(row, dict):
            continue
        kbqa_id = str(row.get("id") or row.get("kbqa_id") or "").strip()
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        if not kbqa_id:
            kbqa_id = compute_mdhash_id(
                json.dumps(
                    {"question": question, "answer": answer},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                prefix="kbqa-",
            )
        vector_data[kbqa_id] = {
            "content": question,
            "kbqa_id": kbqa_id,
        }
    return vector_data


async def ensure_knowledge_base_qa_table(db: Any) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS LIGHTRAG_KNOWLEDGE_BASE_QA (
            workspace VARCHAR(255) NOT NULL,
            id VARCHAR(255) NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            metadata JSONB NULL DEFAULT '{}'::jsonb,
            create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT LIGHTRAG_KNOWLEDGE_BASE_QA_PK PRIMARY KEY (workspace, id)
        )
        """
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_lightrag_knowledge_base_qa_workspace ON LIGHTRAG_KNOWLEDGE_BASE_QA(workspace)"
    )


async def upsert_knowledge_base_qa_rows(
    *,
    db: Any,
    workspace: str,
    qa_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Insert or update knowledge-base QA rows and return normalized rows."""

    await ensure_knowledge_base_qa_table(db)
    normalized_rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for row in qa_rows:
        if not isinstance(row, dict):
            continue
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        kbqa_id = str(row.get("id") or row.get("kbqa_id") or "").strip()
        if not kbqa_id:
            kbqa_id = compute_mdhash_id(
                json.dumps(
                    {"workspace": workspace, "question": question, "answer": answer},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                prefix="kbqa-",
            )
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        await db.execute(
            """
            INSERT INTO LIGHTRAG_KNOWLEDGE_BASE_QA(
                workspace, id, question, answer, metadata, create_time, update_time
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (workspace, id) DO UPDATE
            SET question=EXCLUDED.question,
                answer=EXCLUDED.answer,
                metadata=EXCLUDED.metadata,
                update_time=EXCLUDED.update_time
            """,
            {
                "workspace": workspace,
                "id": kbqa_id,
                "question": question,
                "answer": answer,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "create_time": now,
                "update_time": now,
            },
        )
        normalized_rows.append(
            {
                "id": kbqa_id,
                "question": question,
                "answer": answer,
                "metadata": metadata,
            }
        )

    logger.info("Stored %d knowledge-base QA pairs for workspace=%s", len(normalized_rows), workspace)
    return normalized_rows


async def get_knowledge_base_qa_by_ids(
    *,
    db: Any,
    workspace: str,
    kbqa_ids: list[str],
) -> list[dict[str, Any]]:
    if not kbqa_ids:
        return []

    await ensure_knowledge_base_qa_table(db)
    rows = await db.query(
        """
        SELECT id, workspace, question, answer, metadata,
               EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at,
               EXTRACT(EPOCH FROM update_time)::BIGINT AS updated_at
        FROM LIGHTRAG_KNOWLEDGE_BASE_QA
        WHERE workspace=$1 AND id = ANY($2)
        """,
        [workspace, kbqa_ids],
        multirows=True,
    )
    rows_by_id = {str(row["id"]): row for row in rows or []}
    ordered_rows: list[dict[str, Any]] = []
    for kbqa_id in kbqa_ids:
        row = rows_by_id.get(str(kbqa_id))
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


async def query_knowledge_base_qa_by_rule(
    *,
    db: Any,
    workspace: str,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    await ensure_knowledge_base_qa_table(db)
    query_text = str(query or "").strip()
    if not query_text:
        return []

    rows = await db.query(
        """
        SELECT id, workspace, question, answer, metadata,
               EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at,
               EXTRACT(EPOCH FROM update_time)::BIGINT AS updated_at,
               CASE
                   WHEN lower(question) = lower($2) THEN 0
                   WHEN question ILIKE $3 THEN 1
                   WHEN $2 ILIKE '%' || question || '%' THEN 2
                   ELSE 3
               END AS match_rank
        FROM LIGHTRAG_KNOWLEDGE_BASE_QA
        WHERE workspace=$1
          AND (
              lower(question) = lower($2)
              OR question ILIKE $3
              OR $2 ILIKE '%' || question || '%'
          )
        ORDER BY match_rank ASC, update_time DESC
        LIMIT $4
        """,
        [workspace, query_text, f"%{query_text}%", top_k],
        multirows=True,
    )
    for row in rows or []:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                row["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                row["metadata"] = {}
    return rows or []

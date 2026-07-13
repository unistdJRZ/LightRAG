"""Find and optionally delete LightRAG documents listed in lackofuuid_file.csv.

Default mode is a dry run. It only reads PostgreSQL and writes a match report.
Use --execute-api to call the LightRAG HTTP deletion endpoint after reviewing
the report.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import asyncpg
from pymilvus import MilvusClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "lackofuuid_file.csv"
DEFAULT_REPORT = ROOT / "temp" / "lackofuuid_doc_matches.csv"


@dataclass(frozen=True)
class CsvRow:
    row_number: int
    workspace: str
    status: str
    track_id: str
    ocr_id: str
    global_uuid: str


@dataclass(frozen=True)
class Match:
    row_number: int
    csv_workspace: str
    csv_status: str
    csv_track_id: str
    csv_ocr_id: str
    csv_global_uuid: str
    doc_id: str
    db_workspace: str
    db_status: str
    file_path: str
    track_id: str
    metadata: dict[str, Any]
    matched_by: str


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def read_rows(path: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                CsvRow(
                    row_number=row_number,
                    workspace=(row.get("lightrag_workspace") or "").strip(),
                    status=(row.get("lightrag_status") or "").strip(),
                    track_id=(row.get("lightrag_track_id") or "").strip(),
                    ocr_id=(row.get("lightrag_ocr_id") or "").strip(),
                    global_uuid=(row.get("global_uuid") or "").strip(),
                )
            )
    return rows


def pg_config() -> dict[str, Any]:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
        "database": os.getenv("POSTGRES_DATABASE", "postgres"),
    }


def milvus_config() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "uri": os.getenv("MILVUS_URI", "http://localhost:19530"),
    }
    user = os.getenv("MILVUS_USER")
    password = os.getenv("MILVUS_PASSWORD")
    token = os.getenv("MILVUS_TOKEN")
    db_name = os.getenv("MILVUS_DB_NAME")
    if user:
        kwargs["user"] = user
    if password:
        kwargs["password"] = password
    if token:
        kwargs["token"] = token
    if db_name:
        kwargs["db_name"] = db_name
    return kwargs


def normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def metadata_values(metadata: dict[str, Any], key: str) -> set[str]:
    values: set[str] = set()
    direct = metadata.get(key)
    if direct not in (None, ""):
        values.add(str(direct))
    meta_info = metadata.get("meta_info")
    if isinstance(meta_info, dict):
        nested = meta_info.get(key)
        if nested not in (None, ""):
            values.add(str(nested))
    return values


def matched_by(row: CsvRow, record: asyncpg.Record) -> str:
    reasons: list[str] = []
    metadata = normalize_metadata(record.get("metadata"))
    if row.track_id and row.track_id == (record.get("track_id") or ""):
        reasons.append("track_id")
    if row.ocr_id and row.ocr_id in metadata_values(metadata, "ocr_id"):
        reasons.append("metadata.ocr_id")
    if row.global_uuid:
        if row.global_uuid in metadata_values(metadata, "global_uuid"):
            reasons.append("metadata.global_uuid")
        if row.global_uuid in metadata_values(metadata, "file_id"):
            reasons.append("metadata.file_id")
    return "+".join(reasons) or "unknown"


async def find_matches(rows: list[CsvRow]) -> list[Match]:
    conn = await asyncpg.connect(**pg_config())
    try:
        matches: list[Match] = []
        sql = """
            SELECT workspace, id, status, file_path, track_id, metadata
            FROM LIGHTRAG_DOC_STATUS
            WHERE workspace = $1
              AND (
                ($2::text <> '' AND track_id = $2)
                OR ($3::text <> '' AND metadata->'meta_info'->>'ocr_id' = $3)
                OR ($3::text <> '' AND metadata->>'ocr_id' = $3)
                OR ($4::text <> '' AND metadata->'meta_info'->>'global_uuid' = $4)
                OR ($4::text <> '' AND metadata->>'global_uuid' = $4)
                OR ($4::text <> '' AND metadata->'meta_info'->>'file_id' = $4)
                OR ($4::text <> '' AND metadata->>'file_id' = $4)
              )
            ORDER BY id
        """
        for row in rows:
            if not row.workspace:
                continue
            records = await conn.fetch(
                sql,
                row.workspace,
                row.track_id,
                row.ocr_id,
                row.global_uuid,
            )
            for record in records:
                metadata = normalize_metadata(record.get("metadata"))
                matches.append(
                    Match(
                        row_number=row.row_number,
                        csv_workspace=row.workspace,
                        csv_status=row.status,
                        csv_track_id=row.track_id,
                        csv_ocr_id=row.ocr_id,
                        csv_global_uuid=row.global_uuid,
                        doc_id=record["id"],
                        db_workspace=record["workspace"],
                        db_status=record["status"],
                        file_path=record.get("file_path") or "",
                        track_id=record.get("track_id") or "",
                        metadata=metadata,
                        matched_by=matched_by(row, record),
                    )
                )
        return matches
    finally:
        await conn.close()


def write_report(matches: list[Match], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_number",
        "csv_workspace",
        "csv_status",
        "csv_track_id",
        "csv_ocr_id",
        "csv_global_uuid",
        "doc_id",
        "db_workspace",
        "db_status",
        "file_path",
        "track_id",
        "matched_by",
        "metadata",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            row = match.__dict__.copy()
            row["metadata"] = json.dumps(match.metadata, ensure_ascii=False)
            writer.writerow(row)


def unique_doc_ids(matches: list[Match]) -> list[str]:
    seen: set[str] = set()
    doc_ids: list[str] = []
    for match in matches:
        if match.doc_id not in seen:
            seen.add(match.doc_id)
            doc_ids.append(match.doc_id)
    return doc_ids


def doc_ids_by_workspace(matches: list[Match]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    seen_by_workspace: dict[str, set[str]] = {}
    for match in matches:
        seen = seen_by_workspace.setdefault(match.db_workspace, set())
        if match.doc_id in seen:
            continue
        seen.add(match.doc_id)
        grouped.setdefault(match.db_workspace, []).append(match.doc_id)
    return grouped


def delete_via_api(
    api_base: str,
    workspace: str,
    doc_ids: list[str],
    *,
    delete_file: bool,
    delete_llm_cache: bool,
    batch_size: int,
) -> None:
    api_key = os.getenv("LIGHTRAG_API_KEY", "")
    url = api_base.rstrip("/") + "/api/documents/delete_document"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    for start in range(0, len(doc_ids), batch_size):
        batch = doc_ids[start : start + batch_size]
        payload = {
            "workspace": workspace,
            "doc_ids": batch,
            "delete_file": delete_file,
            "delete_llm_cache": delete_llm_cache,
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="DELETE",
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API deletion failed: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"API deletion failed: {exc}") from exc
        print(f"API accepted batch {start // batch_size + 1}: {body}")


async def collect_chunk_ids(workspace: str, doc_ids: list[str]) -> list[str]:
    if not doc_ids:
        return []

    conn = await asyncpg.connect(**pg_config())
    try:
        status_rows = await conn.fetch(
            """
            SELECT chunks_list
            FROM LIGHTRAG_DOC_STATUS
            WHERE workspace = $1 AND id = ANY($2::text[])
            """,
            workspace,
            doc_ids,
        )
        chunk_ids: set[str] = set()
        for row in status_rows:
            chunks_list = row.get("chunks_list") or []
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []
            if isinstance(chunks_list, list):
                chunk_ids.update(str(chunk_id) for chunk_id in chunks_list if chunk_id)

        chunk_rows = await conn.fetch(
            """
            SELECT id
            FROM LIGHTRAG_DOC_CHUNKS
            WHERE workspace = $1 AND full_doc_id = ANY($2::text[])
            """,
            workspace,
            doc_ids,
        )
        chunk_ids.update(str(row["id"]) for row in chunk_rows)
        return sorted(chunk_ids)
    finally:
        await conn.close()


async def delete_pg_chunks(workspace: str, chunk_ids: list[str]) -> int:
    if not chunk_ids:
        return 0

    conn = await asyncpg.connect(**pg_config())
    try:
        result = await conn.execute(
            """
            DELETE FROM LIGHTRAG_DOC_CHUNKS
            WHERE workspace = $1 AND id = ANY($2::text[])
            """,
            workspace,
            chunk_ids,
        )
        try:
            return int(result.rsplit(" ", 1)[-1])
        except (ValueError, IndexError):
            return 0
    finally:
        await conn.close()


def delete_milvus_chunks(workspace: str, chunk_ids: list[str]) -> int:
    if not chunk_ids:
        return 0

    collection_name = f"{os.getenv('MILVUS_WORKSPACE', workspace)}_chunks"
    client = MilvusClient(**milvus_config())
    if not client.has_collection(collection_name):
        print(f"Milvus collection not found, skipping: {collection_name}")
        return 0

    client.load_collection(collection_name)
    result = client.delete(collection_name=collection_name, pks=chunk_ids)
    if isinstance(result, dict):
        return int(result.get("delete_count", 0) or 0)
    return 0


async def delete_chunks_only(matches: list[Match], batch_size: int) -> None:
    for workspace, workspace_doc_ids in doc_ids_by_workspace(matches).items():
        print(f"Collecting chunks for {len(workspace_doc_ids)} document(s) in {workspace}")
        chunk_ids = await collect_chunk_ids(workspace, workspace_doc_ids)
        print(f"Found {len(chunk_ids)} chunk id(s) in {workspace}")
        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start : start + batch_size]
            milvus_deleted = delete_milvus_chunks(workspace, batch)
            pg_deleted = await delete_pg_chunks(workspace, batch)
            print(
                f"Deleted chunk batch {start // batch_size + 1}: "
                f"milvus={milvus_deleted}, postgres={pg_deleted}"
            )


async def verify_chunks_only(matches: list[Match]) -> None:
    for workspace, workspace_doc_ids in doc_ids_by_workspace(matches).items():
        conn = await asyncpg.connect(**pg_config())
        try:
            remaining = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM LIGHTRAG_DOC_CHUNKS
                WHERE workspace = $1 AND full_doc_id = ANY($2::text[])
                """,
                workspace,
                workspace_doc_ids,
            )
        finally:
            await conn.close()
        print(
            f"Verification for {workspace}: "
            f"docs={len(workspace_doc_ids)}, remaining_pg_chunks={remaining}"
        )


async def async_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--workspace",
        default="msre",
        help="Only match and delete rows for this workspace. Use empty string to include all.",
    )
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument(
        "--execute-chunks-only",
        action="store_true",
        help="Delete only chunk records/vectors for matched docs; do not rebuild KG.",
    )
    parser.add_argument(
        "--verify-chunks-only",
        action="store_true",
        help="Verify remaining PostgreSQL chunks for matched docs.",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:9621")
    parser.add_argument("--delete-file", action="store_true")
    parser.add_argument("--delete-llm-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    rows = read_rows(args.csv)
    if args.workspace:
        rows = [row for row in rows if row.workspace == args.workspace]
    matches = await find_matches(rows)
    write_report(matches, args.report)
    doc_ids = unique_doc_ids(matches)

    print(f"CSV rows: {len(rows)}")
    print(f"Matched rows: {len({m.row_number for m in matches})}")
    print(f"Matched records: {len(matches)}")
    print(f"Unique doc_ids: {len(doc_ids)}")
    print(f"Report: {args.report}")

    selected_modes = [
        args.execute_api,
        args.execute_chunks_only,
        args.verify_chunks_only,
    ]
    if sum(1 for selected in selected_modes if selected) > 1:
        print(
            "Choose only one mode: --execute-api, "
            "--execute-chunks-only, or --verify-chunks-only"
        )
        return 2

    if not any(selected_modes):
        print("Dry run only. Re-run with --execute-api to delete matched documents.")
        print(
            "For chunk-only removal, re-run with --execute-chunks-only "
            "to delete chunk records/vectors without touching KG."
        )
        print("For verification, re-run with --verify-chunks-only.")
        return 0

    if not doc_ids:
        print("No matched doc_ids to delete.")
        return 0

    if args.execute_chunks_only:
        await delete_chunks_only(matches, args.batch_size)
        return 0

    if args.verify_chunks_only:
        await verify_chunks_only(matches)
        return 0

    for workspace, workspace_doc_ids in doc_ids_by_workspace(matches).items():
        print(f"Deleting {len(workspace_doc_ids)} document(s) in workspace: {workspace}")
        delete_via_api(
            args.api_base,
            workspace,
            workspace_doc_ids,
            delete_file=args.delete_file,
            delete_llm_cache=args.delete_llm_cache,
            batch_size=args.batch_size,
        )
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())

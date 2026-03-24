#!/usr/bin/env python3
"""Upload a PDF to OCR file_parse and test LightRAG OCR chunk loading.

This script stops after the existing OCR chunk loading stage. It does not
run the full OCR polling/content pipeline.

Artifacts written to output dir:
- raw_parse_response.json
- normalized_finished_payload.json
- chunk_<result>.json
- load_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

try:
    import ascii_colors  # type: ignore  # noqa: F401
except ImportError:
    import types

    dummy_module = types.ModuleType("ascii_colors")

    class _DummyASCIIColors:
        @staticmethod
        def yellow(*args, **kwargs):
            return None

        @staticmethod
        def green(*args, **kwargs):
            return None

        @staticmethod
        def red(*args, **kwargs):
            return None

        @staticmethod
        def cyan(*args, **kwargs):
            return None

    dummy_module.ASCIIColors = _DummyASCIIColors
    sys.modules["ascii_colors"] = dummy_module

_ORIGINAL_ARGV = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    from lightrag.api.routers.document_routes import (
        OCR_POLL_INTERVAL_SECONDS,
        _extract_ocr_failure_reason,
        _extract_enqueue_input_from_ocr_payload,
        _extract_structured_segments_from_ocr_chunks,
    )
finally:
    sys.argv = _ORIGINAL_ARGV


load_dotenv(dotenv_path=".env", override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a PDF to OCR /file_parse and test OCR chunk loading."
    )
    parser.add_argument("pdf", help="Path to the PDF file to upload")
    parser.add_argument(
        "--ocr-server-url",
        default=None,
        help="OCR server base URL. Defaults to OCR_SERVER_URL from .env",
    )
    parser.add_argument(
        "--output-dir",
        default="output/ocr_load_test",
        help="Directory to store raw payloads and reports",
    )
    parser.add_argument(
        "--lang-list",
        default="ch",
        help="lang_list form field value",
    )
    parser.add_argument(
        "--parse-method",
        default="auto",
        help="parse_method form field value",
    )
    parser.add_argument(
        "--backend",
        default="pipeline",
        help="backend form field value",
    )
    parser.add_argument(
        "--start-page-id",
        type=int,
        default=0,
        help="start_page_id form field value",
    )
    parser.add_argument(
        "--end-page-id",
        type=int,
        default=99999,
        help="end_page_id form field value",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--server-url-field",
        default="string",
        help="server_url form field value from OCR example API",
    )
    return parser


def resolve_parse_url(server_url: str) -> str:
    base = server_url.rstrip("/")
    if base.endswith("/file_parse"):
        return base
    if base.endswith("/content"):
        base = base[: -len("/content")]
    return f"{base}/file_parse"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_file_parse_payload(
    parse_payload: dict[str, Any], fallback_name: str
) -> dict[str, Any]:
    """Wrap file_parse-style payload into FINISHED/content/results shape."""
    if (
        isinstance(parse_payload.get("content"), dict)
        and isinstance(parse_payload["content"].get("results"), dict)
    ):
        normalized = dict(parse_payload)
        normalized.setdefault("status", "FINISHED")
        return normalized

    if isinstance(parse_payload.get("results"), dict):
        return {
            "status": "FINISHED",
            "content": {"results": parse_payload["results"]},
        }

    if "chunk" in parse_payload or "md_content" in parse_payload:
        return {
            "status": "FINISHED",
            "content": {
                "results": {
                    fallback_name: {
                        "chunk": parse_payload.get("chunk"),
                        "md_content": parse_payload.get("md_content"),
                    }
                }
            },
        }

    content = parse_payload.get("content")
    if isinstance(content, dict) and (
        "chunk" in content or "md_content" in content
    ):
        return {
            "status": "FINISHED",
            "content": {
                "results": {
                    fallback_name: {
                        "chunk": content.get("chunk"),
                        "md_content": content.get("md_content"),
                    }
                }
            },
        }

    candidate_results: dict[str, Any] = {}
    for key, value in parse_payload.items():
        if isinstance(value, dict) and (
            "chunk" in value or "md_content" in value
        ):
            candidate_results[str(key)] = value
    if candidate_results:
        return {
            "status": "FINISHED",
            "content": {"results": candidate_results},
        }

    raise ValueError(
        "Unsupported OCR parse response shape: cannot find results/chunk/md_content"
    )


async def call_file_parse(
    parse_url: str,
    pdf_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    form = aiohttp.FormData()
    form.add_field("return_middle_json", "false")
    form.add_field("return_model_output", "false")
    form.add_field("return_md", "true")
    form.add_field("return_images", "false")
    form.add_field("end_page_id", str(args.end_page_id))
    form.add_field("parse_method", args.parse_method)
    form.add_field("start_page_id", str(args.start_page_id))
    form.add_field("lang_list", args.lang_list)
    form.add_field("output_dir", str(output_dir))
    form.add_field("server_url", args.server_url_field)
    form.add_field("return_content_list", "false")
    form.add_field("backend", args.backend)
    form.add_field("table_enable", "true")
    form.add_field("response_format_zip", "false")
    form.add_field("formula_enable", "true")
    form.add_field(
        "files",
        pdf_path.read_bytes(),
        filename=pdf_path.name,
        content_type="application/pdf",
    )

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            parse_url,
            data=form,
            headers={"accept": "application/json"},
        ) as response:
            body = await response.text()
            if response.status >= 400:
                raise RuntimeError(
                    f"POST {parse_url} failed with status {response.status}: {body}"
                )
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"POST {parse_url} returned non-JSON body: {body[:1000]}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"POST {parse_url} returned non-object JSON: {type(payload).__name__}"
                )
            return payload


async def poll_content_payload(
    server_url: str,
    ocr_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    base = server_url.rstrip("/")
    content_url = base if base.endswith("/content") else f"{base}/content"
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.get(content_url, params={"ocr_id": ocr_id}) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"GET {content_url} failed with status {response.status}: {body}"
                    )
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"GET {content_url} returned non-JSON body: {body[:1000]}"
                    ) from exc

            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"GET {content_url} returned non-object JSON: {type(payload).__name__}"
                )

            status_raw = payload.get("status")
            if not isinstance(status_raw, str) or not status_raw.strip():
                return payload

            status = status_raw.strip().upper()
            if status == "PENDING":
                if loop.time() >= deadline:
                    raise TimeoutError(
                        f"OCR polling timed out after {timeout_seconds} seconds for ocr_id={ocr_id}"
                    )
                await asyncio.sleep(OCR_POLL_INTERVAL_SECONDS)
                continue

            if status == "FAIL":
                raise RuntimeError(
                    f"OCR task failed for ocr_id={ocr_id}: "
                    f"{_extract_ocr_failure_reason(payload)}"
                )

            if status == "FINISHED":
                return payload

            raise RuntimeError(f"Unexpected OCR status '{status}' for ocr_id={ocr_id}")


def sanitize_result_name(result_name: str) -> str:
    safe = []
    for ch in result_name:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe) or "result"


def extract_results_map(
    normalized_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    content = normalized_payload.get("content")
    if not isinstance(content, dict):
        return {}
    results = content.get("results")
    if not isinstance(results, dict):
        return {}
    return {
        str(name): value
        for name, value in results.items()
        if isinstance(value, dict)
    }


async def main() -> int:
    args = build_parser().parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF file not found: {pdf_path}")

    ocr_server_url = args.ocr_server_url
    if not ocr_server_url:
        import os

        ocr_server_url = os.getenv("OCR_SERVER_URL")
    if not ocr_server_url:
        raise SystemExit("OCR_SERVER_URL is not configured")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parse_url = resolve_parse_url(ocr_server_url)
    print(f"Using OCR parse URL: {parse_url}")
    print(f"Uploading PDF: {pdf_path}")

    raw_payload = await call_file_parse(parse_url, pdf_path, output_dir, args)
    write_json(output_dir / "raw_parse_response.json", raw_payload)

    if (
        isinstance(raw_payload.get("ocr_id"), str)
        and raw_payload["ocr_id"].strip()
        and "chunk" not in raw_payload
        and "results" not in raw_payload
        and not isinstance(raw_payload.get("content"), dict)
    ):
        ocr_id = raw_payload["ocr_id"].strip()
        print(f"Received ocr_id from parse API: {ocr_id}")
        raw_payload = await poll_content_payload(ocr_server_url, ocr_id, args.timeout)
        write_json(output_dir / "raw_content_response.json", raw_payload)

    normalized_payload = normalize_file_parse_payload(raw_payload, pdf_path.name)
    write_json(output_dir / "normalized_finished_payload.json", normalized_payload)

    results_map = extract_results_map(normalized_payload)
    report: dict[str, Any] = {
        "pdf_path": str(pdf_path),
        "parse_url": parse_url,
        "result_count": len(results_map),
        "results": {},
        "existing_flow": {},
    }

    for result_name, result_payload in results_map.items():
        ocr_chunks = result_payload.get("chunk")
        md_content = result_payload.get("md_content")
        safe_name = sanitize_result_name(result_name)

        if ocr_chunks is not None:
            write_json(output_dir / f"chunk_{safe_name}.json", ocr_chunks)

        result_report = {
            "has_chunk": ocr_chunks is not None,
            "has_md_content": bool(md_content),
        }

        if ocr_chunks is None:
            result_report["load_status"] = "skipped"
            result_report["error"] = "chunk missing"
            report["results"][result_name] = result_report
            continue

        try:
            segments = _extract_structured_segments_from_ocr_chunks(ocr_chunks)
            result_report["load_status"] = "success"
            result_report["segment_count"] = len(segments)
            result_report["segment_preview"] = segments[:3]
        except Exception as exc:
            result_report["load_status"] = "failed"
            result_report["error_type"] = type(exc).__name__
            result_report["error"] = str(exc)
            result_report["traceback"] = traceback.format_exc()

        report["results"][result_name] = result_report

    try:
        enqueue_input = _extract_enqueue_input_from_ocr_payload(normalized_payload)
        report["existing_flow"] = {
            "status": "success",
            "result_type": type(enqueue_input).__name__,
            "item_count": len(enqueue_input.get("content_segments", []))
            if isinstance(enqueue_input, dict)
            else (len(enqueue_input) if isinstance(enqueue_input, list) else None),
            "preview": (
                {
                    "content_preview": str(enqueue_input.get("content") or "")[:500],
                    "content_segments_preview": enqueue_input.get("content_segments", [])[
                        :3
                    ],
                }
                if isinstance(enqueue_input, dict)
                else (
                    enqueue_input[:3]
                    if isinstance(enqueue_input, list)
                    else str(enqueue_input)[:500]
                )
            ),
        }
    except Exception as exc:
        report["existing_flow"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    write_json(output_dir / "load_report.json", report)

    print(f"Artifacts written to: {output_dir}")
    for result_name, result_report in report["results"].items():
        print(
            f"[result] {result_name}: {result_report.get('load_status')} "
            f"{result_report.get('error', '')}".rstrip()
        )
    print(
        f"[existing_flow] {report['existing_flow'].get('status')} "
        f"{report['existing_flow'].get('error', '')}".rstrip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

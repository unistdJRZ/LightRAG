#!/usr/bin/env python3
"""Run randomized concurrent load against LightRAG query endpoints."""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import math
import random
import statistics
import string
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_ENDPOINTS = ["/api/query/data", "/api/query"]
CSV_VALUES_KEY = "_csv_values"


@dataclass(slots=True)
class RequestResult:
    request_id: int
    endpoint: str
    workspace: str | None
    status_code: int | None
    success: bool
    latency_ms: float
    response_bytes: int
    error: str | None = None
    response_body: Any = None


class RateLimiter:
    """Reserve evenly spaced request start times across all workers."""

    def __init__(self, requests_per_second: float | None) -> None:
        self._interval = (
            1.0 / requests_per_second if requests_per_second else 0.0
        )
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self._interval:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            start_at = max(now, self._next_start)
            self._next_start = start_at + self._interval
        delay = start_at - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send concurrent requests with a random parameter combination to "
            "LightRAG /api/query/data and /api/query."
        )
    )
    parser.add_argument("config", type=Path, help="Path to the JSON configuration")
    parser.add_argument("--requests", type=int, help="Override load.total_requests")
    parser.add_argument("--concurrency", type=int, help="Override load.concurrency")
    parser.add_argument("--seed", type=int, help="Override the random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print three payloads without sending requests",
    )
    return parser


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")

    config.setdefault("base_url", "http://127.0.0.1:9621")
    config.setdefault("endpoints", DEFAULT_ENDPOINTS)
    config.setdefault("headers", {})
    config.setdefault("fixed_parameters", {})
    config.setdefault("dynamic_parameters", {})
    config.setdefault("query_generation", {})
    config.setdefault("load", {})
    config.setdefault("output", {})

    _load_csv_queries(config, path)
    _validate_config(config)
    return config


def _require_object(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _validate_choices(values: Any, name: str) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON array")


def _load_csv_queries(config: dict[str, Any], config_path: Path) -> None:
    query_generation = config["query_generation"]
    if not isinstance(query_generation, dict):
        return

    csv_config = query_generation.get("csv")
    if csv_config is None:
        return
    if not isinstance(csv_config, dict):
        raise ValueError("query_generation.csv must be a JSON object")

    path_value = csv_config.get("path")
    column = csv_config.get("column", "question")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("query_generation.csv.path must be a non-empty string")
    if not isinstance(column, str) or not column.strip():
        raise ValueError("query_generation.csv.column must be a non-empty string")

    csv_path = Path(path_value)
    candidates = [csv_path] if csv_path.is_absolute() else [
        config_path.parent / csv_path,
        Path.cwd() / csv_path,
    ]
    resolved_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved_path is None:
        raise ValueError(f"Query CSV file does not exist: {path_value}")

    try:
        with resolved_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or column not in reader.fieldnames:
                available = ", ".join(reader.fieldnames or []) or "<none>"
                raise ValueError(
                    f"Query CSV column '{column}' does not exist in {resolved_path}; "
                    f"available columns: {available}"
                )
            values = [
                value
                for row in reader
                if (value := str(row.get(column) or "").strip())
            ]
    except UnicodeDecodeError as exc:
        raise ValueError(f"Query CSV must be UTF-8 encoded: {resolved_path}") from exc

    if not values:
        raise ValueError(
            f"Query CSV column '{column}' contains no non-empty values: {resolved_path}"
        )
    query_generation[CSV_VALUES_KEY] = values


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config["base_url"], str) or not config["base_url"].strip():
        raise ValueError("base_url must be a non-empty string")
    _validate_choices(config["endpoints"], "endpoints")
    for endpoint in config["endpoints"]:
        if not isinstance(endpoint, str) or not endpoint.startswith("/"):
            raise ValueError("Every endpoint must be an absolute URL path")

    _require_object(config, "headers")
    fixed = _require_object(config, "fixed_parameters")
    dynamic = _require_object(config, "dynamic_parameters")
    query_generation = _require_object(config, "query_generation")
    load = _require_object(config, "load")
    _require_object(config, "output")

    for field, choices in dynamic.items():
        _validate_choices(choices, f"dynamic_parameters.{field}")

    templates = query_generation.get("templates", [])
    variables = query_generation.get("variables", {})
    csv_config = query_generation.get("csv")
    csv_variable = "question"
    if csv_config is not None:
        if not isinstance(csv_config, dict):
            raise ValueError("query_generation.csv must be a JSON object")
        csv_variable = csv_config.get("variable", "question")
        if not isinstance(csv_variable, str) or not csv_variable.strip():
            raise ValueError(
                "query_generation.csv.variable must be a non-empty string"
            )
    if templates:
        _validate_choices(templates, "query_generation.templates")
        if not isinstance(variables, dict):
            raise ValueError("query_generation.variables must be a JSON object")
        for name, choices in variables.items():
            _validate_choices(choices, f"query_generation.variables.{name}")
        available = set(variables)
        if csv_config is not None:
            if csv_variable in variables:
                raise ValueError(
                    f"query_generation.csv.variable '{csv_variable}' duplicates "
                    "query_generation.variables"
                )
            available.add(csv_variable)
        for template in templates:
            if not isinstance(template, str):
                raise ValueError("Every query template must be a string")
            fields = {
                field_name
                for _, field_name, _, _ in string.Formatter().parse(template)
                if field_name
            }
            missing = fields - available
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"Query template uses undefined variables: {names}")
    elif (
        "query" not in dynamic
        and "query" not in fixed
        and csv_config is None
    ):
        raise ValueError(
            "Configure query_generation.templates, query_generation.csv, "
            "dynamic_parameters.query, or fixed_parameters.query"
        )

    total_requests = load.get("total_requests", 100)
    concurrency = load.get("concurrency", 10)
    timeout = load.get("request_timeout_seconds", 120)
    rate = load.get("requests_per_second")
    if not isinstance(total_requests, int) or total_requests < 1:
        raise ValueError("load.total_requests must be an integer >= 1")
    if not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("load.concurrency must be an integer >= 1")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("load.request_timeout_seconds must be > 0")
    if rate is not None and (not isinstance(rate, (int, float)) or rate <= 0):
        raise ValueError("load.requests_per_second must be null or > 0")

    agent_search = fixed.get("agent_search", dynamic.get("agent_search"))
    if agent_search is not None:
        candidates = agent_search if isinstance(agent_search, list) else [agent_search]
        if any(not isinstance(value, bool) for value in candidates):
            raise ValueError(
                "agent_search is a boolean API field. Put injected text in a "
                "query_generation variable such as agent_context."
            )


def build_payload(config: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    payload = copy.deepcopy(config["fixed_parameters"])
    for field, choices in config["dynamic_parameters"].items():
        payload[field] = copy.deepcopy(rng.choice(choices))

    query_generation = config["query_generation"]
    templates = query_generation.get("templates", [])
    csv_config = query_generation.get("csv")
    csv_values = query_generation.get(CSV_VALUES_KEY, [])
    if templates:
        variables = {
            name: copy.deepcopy(rng.choice(choices))
            for name, choices in query_generation.get("variables", {}).items()
        }
        if csv_config is not None:
            variables[csv_config.get("variable", "question")] = rng.choice(csv_values)
        payload["query"] = rng.choice(templates).format_map(variables)
    elif csv_config is not None:
        payload["query"] = rng.choice(csv_values)
    return payload


def _decode_response(body: str, content_type: str) -> Any:
    if "json" in content_type.lower():
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass
    return body


def _semantic_error(response_body: Any) -> str | None:
    if isinstance(response_body, dict) and response_body.get("status") == "failure":
        return str(response_body.get("message") or "Response status is failure")
    return None


async def send_request(
    session: aiohttp.ClientSession,
    request_id: int,
    endpoint: str,
    payload: dict[str, Any],
    save_response_body: bool,
) -> RequestResult:
    started = time.perf_counter()
    try:
        async with session.post(endpoint, json=payload) as response:
            raw_body = await response.text()
            response_body = _decode_response(
                raw_body, response.headers.get("Content-Type", "")
            )
            semantic_error = _semantic_error(response_body)
            success = 200 <= response.status < 300 and semantic_error is None
            error = semantic_error
            if response.status < 200 or response.status >= 300:
                error = f"HTTP {response.status}: {raw_body[:500]}"
            return RequestResult(
                request_id=request_id,
                endpoint=endpoint,
                workspace=payload.get("workspace"),
                status_code=response.status,
                success=success,
                latency_ms=(time.perf_counter() - started) * 1000,
                response_bytes=len(raw_body.encode("utf-8")),
                error=error,
                response_body=response_body if save_response_body else None,
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return RequestResult(
            request_id=request_id,
            endpoint=endpoint,
            workspace=payload.get("workspace"),
            status_code=None,
            success=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            response_bytes=0,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_load(config: dict[str, Any]) -> tuple[list[RequestResult], float]:
    load = config["load"]
    output = config["output"]
    total_requests = load.get("total_requests", 100)
    concurrency = min(load.get("concurrency", 10), total_requests)
    progress_interval = load.get("progress_interval", 10)
    save_response_body = bool(output.get("save_response_body", False))
    seed = config.get("random_seed")
    rng = random.Random(seed)
    limiter = RateLimiter(load.get("requests_per_second"))
    endpoints = [
        f"{config['base_url'].rstrip('/')}{path}" for path in config["endpoints"]
    ]

    timeout = aiohttp.ClientTimeout(total=load.get("request_timeout_seconds", 120))
    connector = aiohttp.TCPConnector(limit=concurrency)
    queue: asyncio.Queue[int] = asyncio.Queue()
    for request_id in range(1, total_requests + 1):
        queue.put_nowait(request_id)

    results: list[RequestResult] = []
    result_lock = asyncio.Lock()

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers={str(k): str(v) for k, v in config["headers"].items()},
    ) as session:

        async def worker() -> None:
            while True:
                try:
                    request_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                payload = build_payload(config, rng)
                endpoint = endpoints[(request_id - 1) % len(endpoints)]
                await limiter.wait()
                result = await send_request(
                    session, request_id, endpoint, payload, save_response_body
                )
                async with result_lock:
                    results.append(result)
                    completed = len(results)
                    if progress_interval and (
                        completed % progress_interval == 0
                        or completed == total_requests
                    ):
                        failures = sum(not item.success for item in results)
                        print(
                            f"completed={completed}/{total_requests} "
                            f"failures={failures}",
                            flush=True,
                        )
                queue.task_done()

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - started

    results.sort(key=lambda item: item.request_id)
    return results, elapsed


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0, min(len(ordered) - 1, math.ceil(len(ordered) * percentage) - 1)
    )
    return ordered[index]


def summarize_results(
    results: list[RequestResult], elapsed: float, seed: int | None
) -> dict[str, Any]:
    latencies = [item.latency_ms for item in results]
    successes = sum(item.success for item in results)
    status_codes = Counter(
        str(item.status_code) if item.status_code is not None else "network_error"
        for item in results
    )

    def endpoint_summary(endpoint: str) -> dict[str, Any]:
        selected = [item for item in results if item.endpoint == endpoint]
        selected_latencies = [item.latency_ms for item in selected]
        selected_successes = sum(item.success for item in selected)
        return {
            "requests": len(selected),
            "successes": selected_successes,
            "failures": len(selected) - selected_successes,
            "latency_ms": {
                "average": round(statistics.fmean(selected_latencies), 2),
                "p95": round(percentile(selected_latencies, 0.95), 2),
                "maximum": round(max(selected_latencies), 2),
            },
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "elapsed_seconds": round(elapsed, 3),
        "requests": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "requests_per_second": round(len(results) / elapsed, 3) if elapsed else 0,
        "response_bytes": sum(item.response_bytes for item in results),
        "status_codes": dict(sorted(status_codes.items())),
        "latency_ms": {
            "minimum": round(min(latencies), 2),
            "average": round(statistics.fmean(latencies), 2),
            "p50": round(percentile(latencies, 0.50), 2),
            "p90": round(percentile(latencies, 0.90), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "p99": round(percentile(latencies, 0.99), 2),
            "maximum": round(max(latencies), 2),
        },
        "endpoints": {
            endpoint: endpoint_summary(endpoint)
            for endpoint in sorted({item.endpoint for item in results})
        },
    }


def write_reports(
    config: dict[str, Any],
    config_path: Path,
    results: list[RequestResult],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir = Path(config["output"].get("directory", "output/query_stress_test"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = output_dir / f"requests_{timestamp}.jsonl"
    summary_path = output_dir / f"summary_{timestamp}.json"

    with details_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False, default=str))
            handle.write("\n")

    report = {
        "config_file": str(config_path),
        "test_settings": {
            "base_url": config["base_url"],
            "endpoints": config["endpoints"],
            "load": config["load"],
        },
        "summary": summary,
    }
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return details_path, summary_path


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.requests is not None:
        if args.requests < 1:
            raise ValueError("--requests must be >= 1")
        config["load"]["total_requests"] = args.requests
    if args.concurrency is not None:
        if args.concurrency < 1:
            raise ValueError("--concurrency must be >= 1")
        config["load"]["concurrency"] = args.concurrency
    if args.seed is not None:
        config["random_seed"] = args.seed


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        apply_overrides(config, args)
        _validate_config(config)

        if args.dry_run:
            rng = random.Random(config.get("random_seed"))
            samples = [build_payload(config, rng) for _ in range(3)]
            print(json.dumps(samples, ensure_ascii=False, indent=2))
            return 0

        results, elapsed = asyncio.run(run_load(config))
        summary = summarize_results(results, elapsed, config.get("random_seed"))
        details_path, summary_path = write_reports(
            config, args.config, results, summary
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"details: {details_path}")
        print(f"summary: {summary_path}")
        if summary["failures"] and config["load"].get("fail_on_error", True):
            return 1
        return 0
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

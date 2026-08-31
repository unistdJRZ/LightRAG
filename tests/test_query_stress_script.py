import json
import random

import pytest

from scripts.query_stress_test import (
    RequestResult,
    _validate_config,
    build_payload,
    load_config,
    percentile,
    summarize_results,
)


def _config():
    return {
        "base_url": "http://127.0.0.1:9621",
        "endpoints": ["/api/query/data", "/api/query"],
        "headers": {},
        "fixed_parameters": {"mode": "mix", "agent_search": True},
        "dynamic_parameters": {
            "workspace": ["one", "two"],
            "top_k": [5, 10],
        },
        "query_generation": {
            "templates": ["{question}\n{agent_context}"],
            "variables": {
                "question": ["first question", "second question"],
                "agent_context": ["context A", "context B"],
            },
        },
        "load": {"total_requests": 10, "concurrency": 2},
        "output": {},
    }


def test_build_payload_uses_configured_random_choices():
    config = _config()

    payloads = [build_payload(config, random.Random(seed)) for seed in range(20)]

    assert all(payload["workspace"] in {"one", "two"} for payload in payloads)
    assert all(payload["top_k"] in {5, 10} for payload in payloads)
    assert all(payload["mode"] == "mix" for payload in payloads)
    assert all("question" in payload["query"] for payload in payloads)
    assert all("context" in payload["query"] for payload in payloads)
    assert len({payload["query"] for payload in payloads}) > 1


def test_agent_search_rejects_context_string():
    config = _config()
    config["fixed_parameters"]["agent_search"] = "context text"

    with pytest.raises(ValueError, match="agent_search is a boolean"):
        _validate_config(config)


def test_build_payload_samples_query_from_csv(tmp_path):
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text(
        "question\n第一个问题\n第二个问题\n\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "stress.json"
    config_path.write_text(
        json.dumps(
            {
                "query_generation": {
                    "csv": {
                        "path": "questions.csv",
                        "column": "question",
                        "variable": "question",
                    },
                    "templates": ["查询：{question}"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    payloads = [build_payload(config, random.Random(seed)) for seed in range(20)]

    assert {payload["query"] for payload in payloads} == {
        "查询：第一个问题",
        "查询：第二个问题",
    }


def test_load_config_rejects_missing_csv_column(tmp_path):
    (tmp_path / "questions.csv").write_text(
        "title\n问题\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "stress.json"
    config_path.write_text(
        json.dumps(
            {
                "query_generation": {
                    "csv": {
                        "path": "questions.csv",
                        "column": "question",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="available columns: title"):
        load_config(config_path)


def test_summary_contains_latency_and_endpoint_breakdown():
    results = [
        RequestResult(1, "http://host/api/query", "one", 200, True, 10.0, 20),
        RequestResult(2, "http://host/api/query", "two", 500, False, 30.0, 10),
        RequestResult(
            3, "http://host/api/query/data", "one", 200, True, 20.0, 30
        ),
    ]

    summary = summarize_results(results, elapsed=1.5, seed=42)

    assert summary["requests"] == 3
    assert summary["successes"] == 2
    assert summary["failures"] == 1
    assert summary["latency_ms"]["p50"] == 20.0
    assert summary["status_codes"] == {"200": 2, "500": 1}
    assert summary["endpoints"]["http://host/api/query"]["requests"] == 2


def test_percentile_uses_nearest_rank():
    assert percentile([30.0, 10.0, 20.0], 0.50) == 20.0

#!/usr/bin/env python3
"""Smoke test Qwen embedding with the repository's current .env settings."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from dotenv import load_dotenv

# Load repo-local .env without overriding existing shell variables.
load_dotenv(dotenv_path=".env", override=False)

if TYPE_CHECKING:
    from lightrag.utils import EmbeddingFunc


def get_embedding_func_class():
    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc


def get_qwen_embed() -> Any:
    from lightrag.llm.transforms_qwen import qwen_embed

    return qwen_embed


@dataclass
class TestConfig:
    binding: str
    model: str
    embedding_dim: int
    embedding_token_limit: int | None
    qwen_torch_dtype: str | None
    qwen_attn_implementation: str | None
    qwen_image_max_size: str | None


def _parse_optional_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def load_test_config() -> TestConfig:
    qwen_embed = get_qwen_embed()
    binding = (os.getenv("EMBEDDING_BINDING") or "").strip().lower()
    default_model = getattr(qwen_embed, "model_name", "Qwen/Qwen3-VL-Embedding-2B")
    default_dim = int(getattr(qwen_embed, "embedding_dim", 2048))
    default_limit = getattr(qwen_embed, "max_token_size", None)

    return TestConfig(
        binding=binding,
        model=(os.getenv("EMBEDDING_MODEL") or default_model).strip(),
        embedding_dim=_parse_optional_int("EMBEDDING_DIM", default_dim) or default_dim,
        embedding_token_limit=_parse_optional_int(
            "EMBEDDING_TOKEN_LIMIT",
            int(default_limit) if default_limit is not None else None,
        ),
        qwen_torch_dtype=(os.getenv("QWEN_TORCH_DTYPE") or "").strip() or None,
        qwen_attn_implementation=(
            os.getenv("QWEN_ATTN_IMPLEMENTATION") or ""
        ).strip()
        or None,
        qwen_image_max_size=(os.getenv("QWEN_IMAGE_MAX_SIZE") or "").strip() or None,
    )


def build_embedding_func(config: TestConfig) -> "EmbeddingFunc":
    qwen_embed = get_qwen_embed()
    embedding_func_class = get_embedding_func_class()
    actual_func = (
        qwen_embed.func if isinstance(qwen_embed, embedding_func_class) else qwen_embed
    )

    async def current_qwen_embedding(
        texts: list[str | dict[str, Any]],
        embedding_dim: int | None = None,
        max_token_size: int | None = None,
    ) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "texts": texts,
            "embedding_dim": embedding_dim,
        }
        if config.model:
            kwargs["model"] = config.model
        if max_token_size is not None:
            kwargs["max_token_size"] = max_token_size
        if config.qwen_torch_dtype:
            kwargs["torch_dtype"] = config.qwen_torch_dtype
        if config.qwen_attn_implementation:
            kwargs["attn_implementation"] = config.qwen_attn_implementation
        return await actual_func(**kwargs)

    return embedding_func_class(
        embedding_dim=config.embedding_dim,
        func=current_qwen_embedding,
        max_token_size=config.embedding_token_limit,
        send_dimensions=True,
        vlm_enable=True,
        model_name=config.model,
    )


def summarize_vector(vector: np.ndarray, preview_size: int) -> dict[str, Any]:
    as_list = vector.astype(np.float32).tolist()
    finite_count = sum(1 for x in as_list if math.isfinite(x))
    nan_count = sum(1 for x in as_list if math.isnan(x))
    inf_count = sum(1 for x in as_list if math.isinf(x))
    l2_norm = float(np.linalg.norm(vector))
    return {
        "dimension": int(vector.shape[0]),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "l2_norm": l2_norm,
        "preview": as_list[: max(preview_size, 0)],
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


async def run_test(
    config: TestConfig,
    image_path: Path,
    text_query: str,
    multimodal_text: str,
    preview_size: int,
) -> dict[str, Any]:
    if config.binding != "qwen":
        raise RuntimeError(
            f"Current EMBEDDING_BINDING={config.binding!r}, not 'qwen'. "
            "This script is intended for the current qwen embedding configuration."
        )

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    embedding_func = build_embedding_func(config)
    payloads: list[tuple[str, str | dict[str, Any]]] = [
        ("text_query", text_query),
        ("image_only", {"image": str(image_path)}),
        ("text_plus_image", {"text": multimodal_text, "image": str(image_path)}),
    ]

    vectors = await embedding_func([item for _, item in payloads])
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise RuntimeError(f"Expected 2D embedding output, got shape={vectors.shape}")
    if vectors.shape[0] != len(payloads):
        raise RuntimeError(
            f"Expected {len(payloads)} vectors, got shape={vectors.shape}"
        )
    if vectors.shape[1] != config.embedding_dim:
        raise RuntimeError(
            f"Expected embedding dim {config.embedding_dim}, got {vectors.shape[1]}"
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding output contains NaN or Inf")

    summary = {
        "config": asdict(config),
        "image_path": str(image_path),
        "vector_shape": list(vectors.shape),
        "cases": {},
        "similarities": {
            "text_query_vs_image_only": cosine_similarity(vectors[0], vectors[1]),
            "text_query_vs_text_plus_image": cosine_similarity(vectors[0], vectors[2]),
            "image_only_vs_text_plus_image": cosine_similarity(vectors[1], vectors[2]),
        },
    }

    for index, (label, _) in enumerate(payloads):
        summary["cases"][label] = summarize_vector(vectors[index], preview_size)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test current repo Qwen embedding config with test.png."
    )
    parser.add_argument(
        "--image",
        default="test.png",
        help="Image file to embed (default: test.png)",
    )
    parser.add_argument(
        "--text-query",
        default="Find the semantic content of this test image.",
        help="Text query used for similarity comparison",
    )
    parser.add_argument(
        "--multimodal-text",
        default="This image is used to verify that Qwen multimodal embedding works.",
        help="Text paired with the image for multimodal embedding",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=8,
        help="How many vector values to preview per case (default: 8)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print JSON only",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        config = load_test_config()
        image_path = Path(args.image).resolve()
        result = asyncio.run(
            run_test(
                config=config,
                image_path=image_path,
                text_query=args.text_query,
                multimodal_text=args.multimodal_text,
                preview_size=args.preview_size,
            )
        )

        if not args.json_only:
            print(
                f"[ok] Qwen embedding succeeded with model={config.model}, dim={config.embedding_dim}"
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        error_payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if args.json_only:
            print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        else:
            print("[fail] Qwen embedding test failed", file=sys.stderr)
            print(json.dumps(error_payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

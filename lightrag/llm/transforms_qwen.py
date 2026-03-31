from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pipmaster as pm
import torch
import torch.nn.functional as F

if not pm.is_installed("transformers"):
    pm.install("transformers>=4.57.0")
if not pm.is_installed("qwen-vl-utils"):
    pm.install("qwen-vl-utils>=0.0.14")
if not pm.is_installed("Pillow"):
    pm.install("Pillow>=10.0.0")

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)

from lightrag.llm.local_model_manager import (
    dispose_torch_resource,
    local_model_manager,
)
from lightrag.utils import logger, wrap_embedding_func_with_attrs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

VLM_ENABLE = True

_IMAGE_URL_PATTERN = re.compile(r"^(https?://|file://|data:image/)", re.IGNORECASE)
_MARKDOWN_DATA_IMAGE_PATTERN = re.compile(
    r"!\[[^\]]*\]\(\s*data:image/[-a-zA-Z0-9.+]+;base64,",
    re.IGNORECASE,
)
_DATA_IMAGE_URI_PATTERN = re.compile(
    r"^(data:image/[-a-zA-Z0-9.+]+;base64,)(.*)$", re.IGNORECASE | re.DOTALL
)
_RAW_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_EMBED_LOCK = asyncio.Lock()
_RERANK_LOCK = asyncio.Lock()


def _parse_max_image_size(
    raw_value: str | None,
    default: tuple[int, int] = (1920, 1080),
) -> tuple[int, int]:
    if not raw_value:
        return default

    normalized = str(raw_value).strip().lower()
    match = re.fullmatch(r"(\d+)\s*[x,]\s*(\d+)", normalized)
    if not match:
        logger.warning(
            "Invalid QWEN_IMAGE_MAX_SIZE=%r, expected format WIDTHxHEIGHT; using default %sx%s",
            raw_value,
            default[0],
            default[1],
        )
        return default

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        logger.warning(
            "Non-positive QWEN_IMAGE_MAX_SIZE=%r, using default %sx%s",
            raw_value,
            default[0],
            default[1],
        )
        return default
    return width, height


_MAX_IMAGE_SIZE = _parse_max_image_size(os.getenv("QWEN_IMAGE_MAX_SIZE"))


def _normalize_torch_dtype(torch_dtype: str | None) -> torch.dtype | None:
    if not torch_dtype:
        return None

    normalized = str(torch_dtype).strip().lower()
    mapping = {
        "auto": None,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype for Qwen backend: {torch_dtype}")
    return mapping[normalized]


def _looks_like_raw_base64(value: str) -> bool:
    stripped = re.sub(r"\s+", "", value)
    if len(stripped) < 64 or len(stripped) % 4 != 0:
        return False
    if not _RAW_BASE64_PATTERN.fullmatch(stripped):
        return False
    try:
        base64.b64decode(stripped, validate=True)
        return True
    except Exception:
        return False


def _select_output_format_and_mime(
    mime_hint: str | None,
    image: Image.Image,
) -> tuple[str, str]:
    normalized_mime = str(mime_hint or "").strip().lower()
    has_alpha = "A" in image.getbands()

    if "png" in normalized_mime or has_alpha:
        return "PNG", "image/png"
    if "webp" in normalized_mime:
        return "WEBP", "image/webp"
    return "JPEG", "image/jpeg"


def _resize_pil_image(image: Image.Image) -> Image.Image:
    max_width, max_height = _MAX_IMAGE_SIZE
    if image.width <= max_width and image.height <= max_height:
        return image.copy()

    resampling_module = getattr(Image, "Resampling", Image)
    resized = image.copy()
    resized.thumbnail((max_width, max_height), resample=resampling_module.LANCZOS)
    return resized


def _pil_image_to_data_uri(image: Image.Image, mime_hint: str | None = None) -> str:
    output_format, output_mime = _select_output_format_and_mime(mime_hint, image)
    output_image = image
    if output_format == "JPEG" and output_image.mode not in ("RGB", "L"):
        output_image = output_image.convert("RGB")
    elif output_format == "PNG" and output_image.mode not in ("RGB", "RGBA", "L", "LA"):
        output_image = output_image.convert("RGBA")

    save_kwargs: dict[str, Any] = {}
    if output_format == "JPEG":
        save_kwargs["quality"] = 95

    buffer = BytesIO()
    output_image.save(buffer, format=output_format, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{output_mime};base64,{encoded}"


def _resize_data_image_uri(image_uri: str) -> str:
    match = _DATA_IMAGE_URI_PATTERN.match(image_uri.strip())
    if not match:
        return image_uri

    mime_prefix = match.group(1)
    encoded_part = re.sub(r"\s+", "", match.group(2))
    image_bytes = base64.b64decode(encoded_part)
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        resized = _resize_pil_image(image)
        return _pil_image_to_data_uri(resized, mime_hint=mime_prefix)


def _resize_local_image_file(path_str: str) -> str:
    path = Path(path_str)
    with Image.open(path) as image:
        image.load()
        resized = _resize_pil_image(image)
        mime_hint = Image.MIME.get(image.format or "", None)
        return _pil_image_to_data_uri(resized, mime_hint=mime_hint)


def _normalize_image_source(image: Any) -> str:
    image_str = str(image or "").strip()
    if not image_str:
        raise ValueError("Image content cannot be empty")

    if image_str.lower().startswith("data:image/"):
        return _resize_data_image_uri(image_str)

    if image_str.lower().startswith("file://"):
        parsed = urlparse(image_str)
        file_path = unquote(parsed.path or "")
        if parsed.netloc:
            file_path = f"//{parsed.netloc}{file_path}"
        if re.match(r"^/[A-Za-z]:/", file_path):
            file_path = file_path[1:]
        return _resize_local_image_file(file_path)

    if image_str.lower().startswith(("http://", "https://")):
        return image_str

    if _looks_like_raw_base64(image_str):
        compact = re.sub(r"\s+", "", image_str)
        return _resize_data_image_uri(f"data:image/jpeg;base64,{compact}")

    if Path(image_str).exists():
        return _resize_local_image_file(image_str)

    return image_str


def _normalize_multimodal_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            raise ValueError("Embedding text item cannot be empty")
        return {"text": text}

    if not isinstance(item, dict):
        raise TypeError(
            f"Qwen multimodal input must be str or dict, got {type(item).__name__}"
        )

    normalized: dict[str, Any] = {}
    if item.get("text") is not None:
        text = str(item.get("text") or "").strip()
        if text:
            normalized["text"] = text

    if item.get("image") is not None:
        normalized["image"] = _normalize_image_source(item.get("image"))

    if item.get("video") is not None:
        normalized["video"] = item.get("video")

    if not normalized:
        raise ValueError("Qwen multimodal input requires at least one of text/image/video")

    return normalized


def _truncate_preview(value: str, limit: int = 96) -> str:
    text = str(value or "").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _classify_image_source(image_value: Any) -> str:
    image_str = str(image_value or "").strip()
    if not image_str:
        return "empty"
    if image_str.lower().startswith("data:image/"):
        return "data_uri"
    if image_str.lower().startswith("file://"):
        return "file_uri"
    if image_str.lower().startswith(("http://", "https://")):
        return "http_url"
    if Path(image_str).exists():
        return "local_path"
    return "opaque"


def _text_contains_embedded_image_payload(text_value: Any) -> bool:
    text_str = str(text_value or "").strip()
    if not text_str:
        return False
    if text_str.lower().startswith("data:image/"):
        return True
    if _MARKDOWN_DATA_IMAGE_PATTERN.search(text_str):
        return True
    return False


def _summarize_embedding_item(
    raw_item: str | dict[str, Any],
    normalized_item: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "raw_type": type(raw_item).__name__,
        "normalized_keys": sorted(normalized_item.keys()),
    }

    if isinstance(raw_item, dict):
        summary["raw_keys"] = sorted(str(key) for key in raw_item.keys())
        raw_text = raw_item.get("text")
        raw_image = raw_item.get("image")
    else:
        raw_text = raw_item
        raw_image = None

    if raw_text is not None:
        raw_text_str = str(raw_text)
        summary["raw_text_chars"] = len(raw_text_str)
        summary["raw_text_preview"] = _truncate_preview(raw_text_str)
        summary["raw_text_looks_like_data_image"] = raw_text_str.strip().lower().startswith(
            "data:image/"
        )
        summary["raw_text_contains_embedded_image_payload"] = (
            _text_contains_embedded_image_payload(raw_text_str)
        )
        summary["raw_text_looks_like_raw_base64"] = _looks_like_raw_base64(
            raw_text_str
        )

    if raw_image is not None:
        raw_image_str = str(raw_image)
        summary["raw_image_source"] = _classify_image_source(raw_image_str)
        summary["raw_image_chars"] = len(raw_image_str)

    normalized_text = normalized_item.get("text")
    if normalized_text is not None:
        normalized_text_str = str(normalized_text)
        summary["normalized_text_chars"] = len(normalized_text_str)
        summary["normalized_text_preview"] = _truncate_preview(normalized_text_str)
        summary["normalized_text_looks_like_data_image"] = (
            normalized_text_str.strip().lower().startswith("data:image/")
        )
        summary["normalized_text_contains_embedded_image_payload"] = (
            _text_contains_embedded_image_payload(normalized_text_str)
        )
        summary["normalized_text_looks_like_raw_base64"] = _looks_like_raw_base64(
            normalized_text_str
        )

    normalized_image = normalized_item.get("image")
    if normalized_image is not None:
        normalized_image_str = str(normalized_image)
        summary["normalized_image_source"] = _classify_image_source(
            normalized_image_str
        )
        summary["normalized_image_chars"] = len(normalized_image_str)
        summary["normalized_image_preview"] = _truncate_preview(normalized_image_str)

    return summary


def _summarize_processor_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in inputs.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            summary[key] = list(shape)
    return summary


def _collect_cuda_memory_snapshot(device: Any) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        return None

    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return None

    device_index = (
        device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    )
    gb = 1024**3
    return {
        "device_index": device_index,
        "allocated_gib": round(torch.cuda.memory_allocated(device_index) / gb, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(device_index) / gb, 3),
        "max_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / gb, 3
        ),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved(device_index) / gb, 3),
    }


def _log_embedding_diagnostics(
    *,
    stage: str,
    exc: Exception,
    items: list[str | dict[str, Any]],
    normalized_items: list[dict[str, Any]] | None = None,
    texts: list[str] | None = None,
    processor_inputs: dict[str, Any] | None = None,
    model: Any = None,
    model_name_or_path: str,
    embedding_dim: int | None,
    max_token_size: int | None,
) -> None:
    effective_normalized_items = normalized_items or []
    item_summaries = [
        _summarize_embedding_item(raw_item, normalized_item)
        for raw_item, normalized_item in zip(items[:8], effective_normalized_items[:8])
    ]
    suspicious_text_items = sum(
        1
        for item in item_summaries
        if item.get("normalized_text_looks_like_data_image")
        or item.get("normalized_text_contains_embedded_image_payload")
        or item.get("normalized_text_looks_like_raw_base64")
        or item.get("raw_text_looks_like_data_image")
        or item.get("raw_text_contains_embedded_image_payload")
        or item.get("raw_text_looks_like_raw_base64")
    )

    diagnostics = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "model": model_name_or_path,
        "embedding_dim": embedding_dim,
        "max_token_size": max_token_size,
        "batch_size": len(items),
        "item_count_logged": len(item_summaries),
        "items_omitted": max(len(items) - len(item_summaries), 0),
        "suspicious_text_item_count": suspicious_text_items,
        "items": item_summaries,
    }
    if texts is not None:
        diagnostics["chat_template_chars"] = [len(text) for text in texts[:8]]
    if processor_inputs is not None:
        diagnostics["processor_shapes"] = _summarize_processor_inputs(processor_inputs)
    if model is not None:
        diagnostics["model_device"] = str(getattr(model, "device", "unknown"))
        cuda_snapshot = _collect_cuda_memory_snapshot(getattr(model, "device", "cpu"))
        if cuda_snapshot is not None:
            diagnostics["cuda_memory"] = cuda_snapshot

    logger.error(
        "Qwen embedding diagnostics: %s",
        json.dumps(diagnostics, ensure_ascii=False),
    )


def _build_message_content(item: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if "text" in item:
        content.append({"type": "text", "text": item["text"]})
    if "image" in item:
        content.append({"type": "image", "image": item["image"]})
    if "video" in item:
        content.append({"type": "video", "video": item["video"]})
    return content


def _resolve_model_runtime_kwargs(
    torch_dtype: str | None = None,
    attn_implementation: str | None = None,
    device_map: str | None = None,
) -> dict[str, Any]:
    runtime_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map or "auto",
    }
    resolved_dtype = _normalize_torch_dtype(
        torch_dtype or os.getenv("QWEN_TORCH_DTYPE", "auto")
    )
    if resolved_dtype is not None:
        runtime_kwargs["torch_dtype"] = resolved_dtype

    resolved_attn = (
        attn_implementation
        or os.getenv("QWEN_ATTN_IMPLEMENTATION")
        or None
    )
    if resolved_attn:
        runtime_kwargs["attn_implementation"] = resolved_attn

    return runtime_kwargs


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    config_class = Qwen3VLConfig
    _no_split_modules = ["Qwen3VLDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.visual = self.model.visual
        self.model.embed_tokens = self.model.language_model.get_input_embeddings()
        self.post_init()

    def forward(self, attention_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        outputs = self.model(
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = outputs.hidden_states[-1]
        if attention_mask is None:
            eos_indices = torch.full(
                (last_hidden_state.size(0),),
                last_hidden_state.size(1) - 1,
                device=last_hidden_state.device,
                dtype=torch.long,
            )
        else:
            eos_indices = attention_mask.to(torch.long).sum(dim=1) - 1
        batch_indices = torch.arange(
            last_hidden_state.size(0), device=last_hidden_state.device
        )
        return last_hidden_state[batch_indices, eos_indices]


def _load_qwen_embedder(
    model_name_or_path: str,
    torch_dtype: str | None,
    attn_implementation: str | None,
    device_map: str | None,
):
    runtime_kwargs = _resolve_model_runtime_kwargs(
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = Qwen3VLForEmbedding.from_pretrained(
        model_name_or_path,
        **runtime_kwargs,
    )
    model.eval()
    return model, processor


def _load_qwen_reranker(
    model_name_or_path: str,
    torch_dtype: str | None,
    attn_implementation: str | None,
    device_map: str | None,
):
    runtime_kwargs = _resolve_model_runtime_kwargs(
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **runtime_kwargs,
    )
    model.eval()
    tokenizer = processor.tokenizer
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_token_id = tokenizer.encode("no", add_special_tokens=False)[-1]
    return model, processor, yes_token_id, no_token_id


def _build_embedding_messages(
    items: list[str | dict[str, Any]],
    instruction: str | None = None,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    messages: list[list[dict[str, Any]]] = []
    normalized_items: list[dict[str, Any]] = []
    instruction_text = str(instruction or "").strip()
    for raw_item in items:
        item = _normalize_multimodal_item(raw_item)
        normalized_items.append(item)
        content = _build_message_content(item)
        if instruction_text:
            content = [{"type": "text", "text": instruction_text}] + content
        messages.append([{"role": "user", "content": content}])
    return messages, normalized_items


def _run_embedding_sync(
    items: list[str | dict[str, Any]],
    *,
    model_name_or_path: str,
    embedding_dim: int | None,
    instruction: str | None,
    max_token_size: int | None,
    torch_dtype: str | None,
    attn_implementation: str | None,
    device_map: str | None,
) -> np.ndarray:
    cache_key = (
        "qwen_embedder",
        model_name_or_path,
        torch_dtype,
        attn_implementation,
        device_map,
    )
    model_label = f"qwen_embedder:{model_name_or_path}"
    normalized_items: list[dict[str, Any]] | None = None
    texts: list[str] | None = None
    inputs: dict[str, Any] | None = None
    stage = "build_messages"
    try:
        with local_model_manager.lease(
            key=cache_key,
            loader=lambda: _load_qwen_embedder(
                model_name_or_path,
                torch_dtype,
                attn_implementation,
                device_map,
            ),
            disposer=dispose_torch_resource,
            label=model_label,
        ) as resource:
            model, processor = resource
            messages, normalized_items = _build_embedding_messages(
                items, instruction=instruction
            )
            stage = "apply_chat_template"
            texts = [
                processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for message in messages
            ]
            stage = "process_vision"
            image_inputs, video_inputs = process_vision_info(messages)
            processor_kwargs: dict[str, Any] = {
                "text": texts,
                "images": image_inputs,
                "videos": video_inputs,
                "padding": True,
                "return_tensors": "pt",
            }
            if max_token_size and max_token_size > 0:
                processor_kwargs["truncation"] = True
                processor_kwargs["max_length"] = max_token_size

            stage = "processor"
            inputs = processor(**processor_kwargs)
            stage = "to_device"
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            stage = "forward"
            with torch.inference_mode():
                embeddings = model(**inputs)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                if embedding_dim and 0 < embedding_dim < embeddings.shape[1]:
                    embeddings = embeddings[:, :embedding_dim]
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            return embeddings.to(torch.float32).cpu().numpy()
    except Exception as exc:
        _log_embedding_diagnostics(
            stage=stage,
            exc=exc,
            items=items,
            normalized_items=normalized_items,
            texts=texts,
            processor_inputs=inputs,
            model=model,
            model_name_or_path=model_name_or_path,
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
        )
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        raise


def _build_rerank_messages(
    query: str | dict[str, Any],
    documents: list[str | dict[str, Any]],
    instruction: str,
) -> list[list[dict[str, Any]]]:
    normalized_query = _normalize_multimodal_item(query)
    normalized_documents = [_normalize_multimodal_item(doc) for doc in documents]
    messages: list[list[dict[str, Any]]] = []
    for document in normalized_documents:
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Instruction: {instruction}\nQuery:\n",
            }
        ]
        user_content.extend(_build_message_content(normalized_query))
        user_content.append(
            {
                "type": "text",
                "text": "\nDocument:\n",
            }
        )
        user_content.extend(_build_message_content(document))
        user_content.append(
            {
                "type": "text",
                "text": "\nAnswer with yes or no only.",
            }
        )
        messages.append([{"role": "user", "content": user_content}])
    return messages


def _run_rerank_sync(
    *,
    query: str | dict[str, Any],
    documents: list[str | dict[str, Any]],
    model_name_or_path: str,
    instruction: str,
    top_n: int | None,
    max_token_size: int | None,
    torch_dtype: str | None,
    attn_implementation: str | None,
    device_map: str | None,
) -> list[dict[str, float | int]]:
    cache_key = (
        "qwen_reranker",
        model_name_or_path,
        torch_dtype,
        attn_implementation,
        device_map,
    )
    model_label = f"qwen_reranker:{model_name_or_path}"
    with local_model_manager.lease(
        key=cache_key,
        loader=lambda: _load_qwen_reranker(
            model_name_or_path,
            torch_dtype,
            attn_implementation,
            device_map,
        ),
        disposer=dispose_torch_resource,
        label=model_label,
    ) as resource:
        model, processor, yes_token_id, no_token_id = resource
        messages = _build_rerank_messages(query, documents, instruction)
        texts = [
            processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if max_token_size and max_token_size > 0:
            processor_kwargs["truncation"] = True
            processor_kwargs["max_length"] = max_token_size

        inputs = processor(**processor_kwargs)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            yes_no_logits = torch.stack(
                [logits[:, yes_token_id], logits[:, no_token_id]],
                dim=1,
            )
            scores = torch.softmax(yes_no_logits, dim=1)[:, 0]

        indexed_scores = [
            {"index": index, "relevance_score": float(score)}
            for index, score in enumerate(scores.to(torch.float32).cpu().tolist())
        ]
        indexed_scores.sort(key=lambda item: item["relevance_score"], reverse=True)
        return indexed_scores[:top_n] if top_n else indexed_scores


@wrap_embedding_func_with_attrs(
    embedding_dim=2048,
    max_token_size=2048,
    model_name="Qwen/Qwen3-VL-Embedding-2B",
    vlm_enable=VLM_ENABLE,
)
async def qwen_embed(
    texts: list[str | dict[str, Any]],
    model: str = "Qwen/Qwen3-VL-Embedding-2B",
    embedding_dim: int | None = None,
    instruction: str | None = None,
    max_token_size: int | None = None,
    torch_dtype: str | None = None,
    attn_implementation: str | None = None,
    device_map: str | None = None,
) -> np.ndarray:
    async with _EMBED_LOCK:
        from lightrag.llm.local_model_process import call_qwen_embedding_in_worker

        return await asyncio.to_thread(
            call_qwen_embedding_in_worker,
            {
                "items": texts,
                "model_name_or_path": model,
                "embedding_dim": embedding_dim,
                "instruction": instruction,
                "max_token_size": max_token_size,
                "torch_dtype": torch_dtype,
                "attn_implementation": attn_implementation,
                "device_map": device_map,
            },
        )


async def qwen_rerank(
    query: str | dict[str, Any],
    documents: list[str | dict[str, Any]],
    top_n: int | None = None,
    model: str = "Qwen/Qwen3-VL-Reranker-2B",
    instruction: str = "Retrieve images or text relevant to the user's query.",
    max_token_size: int | None = 32768,
    torch_dtype: str | None = None,
    attn_implementation: str | None = None,
    device_map: str | None = None,
    **_: Any,
) -> list[dict[str, float | int]]:
    if not documents:
        return []

    async with _RERANK_LOCK:
        from lightrag.llm.local_model_process import call_qwen_rerank_in_worker

        return await asyncio.to_thread(
            call_qwen_rerank_in_worker,
            {
                "query": query,
                "documents": documents,
                "model_name_or_path": model,
                "instruction": instruction,
                "top_n": top_n,
                "max_token_size": max_token_size,
                "torch_dtype": torch_dtype,
                "attn_implementation": attn_implementation,
                "device_map": device_map,
            },
        )


qwen_rerank.vlm_enable = VLM_ENABLE

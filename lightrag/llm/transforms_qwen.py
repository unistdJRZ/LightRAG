from __future__ import annotations

import asyncio
import base64
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
import pipmaster as pm
import torch
import torch.nn.functional as F

if not pm.is_installed("transformers"):
    pm.install("transformers>=4.57.0")
if not pm.is_installed("qwen-vl-utils"):
    pm.install("qwen-vl-utils>=0.0.14")

from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)

from lightrag.utils import logger, wrap_embedding_func_with_attrs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

VLM_ENABLE = True

_IMAGE_URL_PATTERN = re.compile(r"^(https?://|file://|data:image/)", re.IGNORECASE)
_RAW_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_EMBED_LOCK = asyncio.Lock()
_RERANK_LOCK = asyncio.Lock()


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


def _normalize_image_source(image: Any) -> str:
    image_str = str(image or "").strip()
    if not image_str:
        raise ValueError("Image content cannot be empty")

    if _IMAGE_URL_PATTERN.match(image_str):
        return image_str

    if _looks_like_raw_base64(image_str):
        compact = re.sub(r"\s+", "", image_str)
        return f"data:image;base64,{compact}"

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


@lru_cache(maxsize=2)
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


@lru_cache(maxsize=2)
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
) -> list[list[dict[str, Any]]]:
    messages: list[list[dict[str, Any]]] = []
    instruction_text = str(instruction or "").strip()
    for raw_item in items:
        item = _normalize_multimodal_item(raw_item)
        content = _build_message_content(item)
        if instruction_text:
            content = [{"type": "text", "text": instruction_text}] + content
        messages.append([{"role": "user", "content": content}])
    return messages


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
    model, processor = _load_qwen_embedder(
        model_name_or_path,
        torch_dtype,
        attn_implementation,
        device_map,
    )
    messages = _build_embedding_messages(items, instruction=instruction)
    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
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
        embeddings = model(**inputs)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if embedding_dim and 0 < embedding_dim < embeddings.shape[1]:
            embeddings = embeddings[:, :embedding_dim]
            embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.to(torch.float32).cpu().numpy()


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
    model, processor, yes_token_id, no_token_id = _load_qwen_reranker(
        model_name_or_path,
        torch_dtype,
        attn_implementation,
        device_map,
    )
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
    max_token_size=32768,
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
        return await asyncio.to_thread(
            _run_embedding_sync,
            texts,
            model_name_or_path=model,
            embedding_dim=embedding_dim,
            instruction=instruction,
            max_token_size=max_token_size,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
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
        return await asyncio.to_thread(
            _run_rerank_sync,
            query=query,
            documents=documents,
            model_name_or_path=model,
            instruction=instruction,
            top_n=top_n,
            max_token_size=max_token_size,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )


qwen_rerank.vlm_enable = VLM_ENABLE

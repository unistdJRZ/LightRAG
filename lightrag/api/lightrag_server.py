"""
LightRAG FastAPI Server
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
import os
import logging
import logging.config
import sys
import subprocess
import shutil
from typing import Any, Literal
import numpy as np
import uvicorn
import pipmaster as pm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
import configparser
from ascii_colors import ASCIIColors
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from lightrag.api.utils_api import (
    get_combined_auth_dependency,
    display_splash_screen,
    check_env_file,
)
from .config import (
    global_args,
    update_uvicorn_mode_config,
    get_default_host,
)
from lightrag.utils import get_env_value
from lightrag import LightRAG, __version__ as core_version
from lightrag.api import __api_version__
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc
from lightrag.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
)
from lightrag.api.routers.document_routes import (
    DocumentManager,
    create_document_routes,
)
from lightrag.api.agent import create_agent_routes
from lightrag.api.routers.query_routes import create_query_routes
from lightrag.api.routers.graph_routes import create_graph_routes
from lightrag.api.routers.knowledge_base_qa_routes import (
    create_knowledge_base_qa_routes,
)
from lightrag.api.routers.ollama_api import OllamaAPI

from lightrag.utils import (
    get_chunk_image_fields,
    logger,
    remove_think_tags,
    set_verbose_debug,
)
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_default_workspace,
    # set_default_workspace,
    cleanup_keyed_lock,
    finalize_share_data,
)
from lightrag.llm.local_model_manager import shutdown_local_model_manager
from lightrag.llm.local_model_process import shutdown_local_model_process_manager
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from lightrag.api.auth import auth_handler
from lightrag.workspace_config import (
    WorkspaceDefinition,
    build_workspace_alias_map,
    display_workspace_id,
    load_workspace_config,
    parse_workspace_csv,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


webui_title = os.getenv("WEBUI_TITLE")
webui_description = os.getenv("WEBUI_DESCRIPTION")

# Initialize config parser
config = configparser.ConfigParser()
config.read("config.ini")

# Global authentication configuration
auth_configured = bool(auth_handler.accounts)


class ChunkContentRequest(BaseModel):
    chunk_id: str = Field(min_length=1, description="Chunk identifier to fetch")
    workspace: str | None = Field(
        default=None,
        description="Optional workspace id or alias. If omitted, falls back to query/header/default workspace routing.",
    )


class ChunkContentResponse(BaseModel):
    chunk_id: str = Field(description="Chunk identifier")
    content: str = Field(description="Full text content of the requested chunk")
    content_type: str | None = Field(
        default=None, description="Chunk content type when available"
    )
    full_doc_id: str | None = Field(
        default=None, description="Owning document identifier when available"
    )
    file_id: str | None = Field(
        default=None, description="Document file_id from doc status metadata when available"
    )
    file_path: str | None = Field(
        default=None, description="Document file path when available"
    )
    page_id: int | None = Field(default=None, description="Page index when available")
    bbox: list[float] | None = Field(
        default=None, description="Chunk bounding box when available"
    )
    page_size: list[float] | None = Field(
        default=None, description="Page size when available"
    )
    ocr_chunk_id: str | None = Field(
        default=None, description="OCR chunk identifier when available"
    )
    image_base64: str | None = Field(
        default=None, description="Image payload for image chunks when available"
    )
    image_text: str | None = Field(
        default=None, description="Accompanying OCR text for image chunks when available"
    )


class ChunksPaginatedRequest(BaseModel):
    page: int = Field(default=1, ge=1, description="1-based page number")
    page_size: int = Field(
        default=20, ge=1, le=200, description="Number of chunks per page"
    )
    sort_direction: Literal["asc", "desc"] = Field(
        default="desc",
        description="Chunk ordering follows document updated_at ordering and per-document chunk order",
    )
    workspace: str | None = Field(
        default=None,
        description="Optional workspace id or alias. If omitted, falls back to query/header/default workspace routing.",
    )


class ChunksPaginationInfo(BaseModel):
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of items per page")
    total_count: int = Field(description="Total number of chunks")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_prev: bool = Field(description="Whether there is a previous page")


class ChunkPreviewItem(BaseModel):
    chunk_id: str = Field(description="Chunk identifier")
    doc_id: str = Field(description="Owning document identifier")
    file_path: str = Field(description="Document file path")
    content: str = Field(description="Stored chunk content")
    content_type: str | None = Field(
        default=None, description="Chunk content type when available"
    )
    page_id: int | None = Field(default=None, description="Page index when available")
    bbox: list[float] | None = Field(
        default=None, description="Chunk bounding box when available"
    )
    ocr_chunk_id: str | None = Field(
        default=None, description="OCR chunk identifier when available"
    )
    image_base64: str | None = Field(
        default=None, description="Image payload for image chunks when available"
    )
    image_text: str | None = Field(
        default=None, description="Accompanying OCR text for image chunks when available"
    )
    chunk_order_index: int | None = Field(
        default=None, description="Chunk order within the source document"
    )
    tokens: int | None = Field(default=None, description="Token count when available")


class ChunksPaginatedResponse(BaseModel):
    chunks: list[ChunkPreviewItem] = Field(
        description="Chunks for the current page"
    )
    pagination: ChunksPaginationInfo = Field(description="Pagination information")


class TranslateChunkRequest(BaseModel):
    chunk_id: str = Field(min_length=1, description="Chunk identifier to translate")
    workspace: str | None = Field(
        default=None,
        description="Optional workspace id or alias. If omitted, falls back to query/header/default workspace routing.",
    )


class TranslateChunkResponse(BaseModel):
    chunk_id: str = Field(description="Chunk identifier")
    translated_cn: str = Field(description="Chinese translation of the chunk content")
    cached: bool = Field(
        description="Whether translated_cn already existed on the chunk"
    )


class WorkspaceDescriptionUpdateRequest(BaseModel):
    description: str = Field(
        default="",
        description="Workspace description shown to agents and WebUI users.",
    )


class WorkspaceInfoItem(BaseModel):
    id: str = Field(description="Workspace id")
    alias: str = Field(description="Workspace display alias")
    description: str = Field(default="", description="Workspace description")
    has_description: bool = Field(description="Whether a non-empty description exists")


TRANSLATE_TO_CN_SYSTEM_PROMPT = """You are a professional translator.
Translate the user's text into Simplified Chinese.
Requirements:
- Keep the original meaning accurate and complete.
- Preserve Markdown structure when present.
- Preserve proper nouns, product names, code, URLs, numbers, and file paths when translation would be inappropriate.
- Return only the translated Chinese text without explanations or extra commentary."""


def _extract_file_id_from_doc_status(doc_status: Any) -> str | None:
    if doc_status is None:
        return None

    metadata = getattr(doc_status, "metadata", None)
    if metadata is None and isinstance(doc_status, dict):
        metadata = doc_status.get("metadata")
    if not isinstance(metadata, dict):
        return None

    meta_info = metadata.get("meta_info")
    if isinstance(meta_info, dict) and meta_info.get("file_id") is not None:
        return str(meta_info["file_id"])

    if metadata.get("file_id") is not None:
        return str(metadata["file_id"])

    return None


async def _resolve_chunk_file_id(rag: LightRAG, chunk_data: dict[str, Any]) -> str | None:
    doc_status = getattr(rag, "doc_status", None)
    if doc_status is None:
        return None

    full_doc_id = chunk_data.get("full_doc_id")
    if isinstance(full_doc_id, str) and full_doc_id:
        try:
            file_id = _extract_file_id_from_doc_status(
                await doc_status.get_by_id(full_doc_id)
            )
            if file_id is not None:
                return file_id
        except Exception as exc:
            logger.warning("Failed to resolve file_id for doc %s: %s", full_doc_id, exc)

    file_path = chunk_data.get("file_path")
    if isinstance(file_path, str) and file_path:
        get_doc_by_file_path = getattr(doc_status, "get_doc_by_file_path", None)
        if callable(get_doc_by_file_path):
            try:
                return _extract_file_id_from_doc_status(
                    await get_doc_by_file_path(file_path)
                )
            except Exception as exc:
                logger.warning(
                    "Failed to resolve file_id for file_path %s: %s",
                    file_path,
                    exc,
                )

    return None


async def translate_chunk_to_cn(
    rag: LightRAG,
    chunk_id: str,
) -> tuple[str, bool]:
    """Translate a stored chunk to Chinese and persist translated_cn if needed."""
    if not callable(getattr(rag, "llm_model_func", None)):
        raise ValueError("LLM model function is not configured")

    chunk_data = await rag.text_chunks.get_by_id(chunk_id)
    if not chunk_data or "content" not in chunk_data:
        raise KeyError(f"Chunk '{chunk_id}' not found")

    existing_translation = chunk_data.get("translated_cn")
    if isinstance(existing_translation, str) and existing_translation.strip():
        return existing_translation, True

    content_type = str(chunk_data.get("content_type") or "").strip().lower()
    if content_type == "image":
        raise ValueError("Image chunks do not support translation")

    source_content = str(chunk_data.get("content") or "").strip()
    if not source_content:
        raise ValueError("Chunk content is empty")

    translated = await rag.llm_model_func(
        source_content,
        system_prompt=TRANSLATE_TO_CN_SYSTEM_PROMPT,
        history_messages=[],
        enable_cot=False,
        _priority=3,
    )
    translated_text = remove_think_tags(str(translated or "")).strip()
    if not translated_text:
        raise ValueError("Translation returned empty content")

    chunk_data["translated_cn"] = translated_text
    await rag.text_chunks.upsert({chunk_id: chunk_data})

    return translated_text, False


async def get_chunks_paginated(
    rag: LightRAG,
    page: int,
    page_size: int,
    sort_direction: Literal["asc", "desc"] = "desc",
) -> tuple[list[dict[str, Any]], int]:
    """Build a chunk page by walking paginated document status records."""
    doc_page_size = 200
    doc_page = 1
    total_chunks = 0
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    selected_chunk_meta: list[dict[str, Any]] = []

    while True:
        docs_page, _ = await rag.doc_status.get_docs_paginated(
            page=doc_page,
            page_size=doc_page_size,
            sort_field="updated_at",
            sort_direction=sort_direction,
        )
        if not docs_page:
            break

        for doc_id, doc_status in docs_page:
            chunk_ids = list(getattr(doc_status, "chunks_list", []) or [])
            file_path = str(getattr(doc_status, "file_path", "unknown_source") or "unknown_source")

            for chunk_order_index, chunk_id in enumerate(chunk_ids):
                if not chunk_id:
                    continue

                if start_index <= total_chunks < end_index:
                    selected_chunk_meta.append(
                        {
                            "chunk_id": str(chunk_id),
                            "doc_id": str(doc_id),
                            "file_path": file_path,
                            "chunk_order_index": chunk_order_index,
                        }
                    )
                total_chunks += 1

        if len(docs_page) < doc_page_size:
            break
        doc_page += 1

    if not selected_chunk_meta:
        return [], total_chunks

    chunk_records = await rag.text_chunks.get_by_ids(
        [item["chunk_id"] for item in selected_chunk_meta]
    )

    page_chunks: list[dict[str, Any]] = []
    for meta, record in zip(selected_chunk_meta, chunk_records):
        chunk_record = record or {}
        raw_ocr_chunk_id = chunk_record.get("ocr_chunk_id")
        resolved_image_base64, resolved_image_text = get_chunk_image_fields(chunk_record)
        chunk_content = chunk_record.get("content", "")
        if str(chunk_record.get("content_type", "") or "").strip().lower() == "image":
            chunk_content = resolved_image_base64 or chunk_content
        page_chunks.append(
            {
                "chunk_id": meta["chunk_id"],
                "doc_id": meta["doc_id"],
                "file_path": chunk_record.get("file_path")
                or meta["file_path"],
                "content": str(chunk_content or ""),
                "content_type": chunk_record.get("content_type"),
                "page_id": chunk_record.get("page_id"),
                "bbox": chunk_record.get("bbox"),
                "ocr_chunk_id": (
                    str(raw_ocr_chunk_id)
                    if raw_ocr_chunk_id is not None
                    else None
                ),
                "image_base64": resolved_image_base64,
                "image_text": resolved_image_text,
                "chunk_order_index": chunk_record.get(
                    "chunk_order_index", meta["chunk_order_index"]
                ),
                "tokens": chunk_record.get("tokens"),
            }
        )

    return page_chunks, total_chunks


class LLMConfigCache:
    """Smart LLM and Embedding configuration cache class"""

    def __init__(self, args):
        self.args = args

        # Initialize configurations based on binding conditions
        self.openai_llm_options = None
        self.gemini_llm_options = None
        self.gemini_embedding_options = None
        self.ollama_llm_options = None
        self.ollama_embedding_options = None

        # Only initialize and log OpenAI options when using OpenAI-related bindings
        if args.llm_binding in ["openai", "azure_openai"]:
            from lightrag.llm.binding_options import OpenAILLMOptions

            self.openai_llm_options = OpenAILLMOptions.options_dict(args)
            logger.info(f"OpenAI LLM Options: {self.openai_llm_options}")

        if args.llm_binding == "gemini":
            from lightrag.llm.binding_options import GeminiLLMOptions

            self.gemini_llm_options = GeminiLLMOptions.options_dict(args)
            logger.info(f"Gemini LLM Options: {self.gemini_llm_options}")

        # Only initialize and log Ollama LLM options when using Ollama LLM binding
        if args.llm_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions

                self.ollama_llm_options = OllamaLLMOptions.options_dict(args)
                logger.info(f"Ollama LLM Options: {self.ollama_llm_options}")
            except ImportError:
                logger.warning(
                    "OllamaLLMOptions not available, using default configuration"
                )
                self.ollama_llm_options = {}

        # Only initialize and log Ollama Embedding options when using Ollama Embedding binding
        if args.embedding_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaEmbeddingOptions

                self.ollama_embedding_options = OllamaEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Ollama Embedding Options: {self.ollama_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "OllamaEmbeddingOptions not available, using default configuration"
                )
                self.ollama_embedding_options = {}

        # Only initialize and log Gemini Embedding options when using Gemini Embedding binding
        if args.embedding_binding == "gemini":
            try:
                from lightrag.llm.binding_options import GeminiEmbeddingOptions

                self.gemini_embedding_options = GeminiEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Gemini Embedding Options: {self.gemini_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "GeminiEmbeddingOptions not available, using default configuration"
                )
                self.gemini_embedding_options = {}


def check_frontend_build():
    """Check if frontend is built and optionally check if source is up-to-date

    Returns:
        tuple: (assets_exist: bool, is_outdated: bool)
            - assets_exist: True if WebUI build files exist
            - is_outdated: True if source is newer than build (only in dev environment)
    """
    webui_dir = Path(__file__).parent / "webui"
    index_html = webui_dir / "index.html"

    # 1. Check if build files exist
    if not index_html.exists():
        ASCIIColors.yellow("\n" + "=" * 80)
        ASCIIColors.yellow("WARNING: Frontend Not Built")
        ASCIIColors.yellow("=" * 80)
        ASCIIColors.yellow("The WebUI frontend has not been built yet.")
        ASCIIColors.yellow("The API server will start without the WebUI interface.")
        ASCIIColors.yellow(
            "\nTo enable WebUI, build the frontend using these commands:\n"
        )
        ASCIIColors.cyan("    cd lightrag_webui")
        ASCIIColors.cyan("    bun install --frozen-lockfile")
        ASCIIColors.cyan("    bun run build")
        ASCIIColors.cyan("    cd ..")
        ASCIIColors.yellow("\nThen restart the service.\n")
        ASCIIColors.cyan(
            "Note: Make sure you have Bun installed. Visit https://bun.sh for installation."
        )
        ASCIIColors.yellow("=" * 80 + "\n")
        return (False, False)  # Assets don't exist, not outdated

    # 2. Check if this is a development environment (source directory exists)
    try:
        source_dir = Path(__file__).parent.parent.parent / "lightrag_webui"
        src_dir = source_dir / "src"

        # Determine if this is a development environment: source directory exists and contains src directory
        if not source_dir.exists() or not src_dir.exists():
            # Production environment, skip source code check
            logger.debug(
                "Production environment detected, skipping source freshness check"
            )
            return (True, False)  # Assets exist, not outdated (prod environment)

        # Development environment, perform source code timestamp check
        logger.debug("Development environment detected, checking source freshness")

        # Source code file extensions (files to check)
        source_extensions = {
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",  # TypeScript/JavaScript
            ".css",
            ".scss",
            ".sass",
            ".less",  # Style files
            ".json",
            ".jsonc",  # Configuration/data files
            ".html",
            ".htm",  # Template files
            ".md",
            ".mdx",  # Markdown
        }

        # Key configuration files (in lightrag_webui root directory)
        key_files = [
            source_dir / "package.json",
            source_dir / "bun.lock",
            source_dir / "vite.config.ts",
            source_dir / "tsconfig.json",
            source_dir / "tailraid.config.js",
            source_dir / "index.html",
        ]

        # Get the latest modification time of source code
        latest_source_time = 0

        # Check source code files in src directory
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                # Only check source code files, ignore temporary files and logs
                if file_path.suffix.lower() in source_extensions:
                    mtime = file_path.stat().st_mtime
                    latest_source_time = max(latest_source_time, mtime)

        # Check key configuration files
        for key_file in key_files:
            if key_file.exists():
                mtime = key_file.stat().st_mtime
                latest_source_time = max(latest_source_time, mtime)

        # Get build time
        build_time = index_html.stat().st_mtime

        # Compare timestamps (5 second tolerance to avoid file system time precision issues)
        if latest_source_time > build_time + 5:
            ASCIIColors.yellow("\n" + "=" * 80)
            ASCIIColors.yellow("WARNING: Frontend Source Code Has Been Updated")
            ASCIIColors.yellow("=" * 80)
            ASCIIColors.yellow(
                "The frontend source code is newer than the current build."
            )
            ASCIIColors.yellow(
                "This might happen after 'git pull' or manual code changes.\n"
            )
            ASCIIColors.cyan(
                "Recommended: Rebuild the frontend to use the latest changes:"
            )
            ASCIIColors.cyan("    cd lightrag_webui")
            ASCIIColors.cyan("    bun install --frozen-lockfile")
            ASCIIColors.cyan("    bun run build")
            ASCIIColors.cyan("    cd ..")
            ASCIIColors.yellow("\nThe server will continue with the current build.")
            ASCIIColors.yellow("=" * 80 + "\n")
            return (True, True)  # Assets exist, outdated
        else:
            logger.info("Frontend build is up-to-date")
            return (True, False)  # Assets exist, up-to-date

    except Exception as e:
        # If check fails, log warning but don't affect startup
        logger.warning(f"Failed to check frontend source freshness: {e}")
        return (True, False)  # Assume assets exist and up-to-date on error


def get_webui_dev_public_url(args) -> str:
    """Resolve the browser-facing URL for WebUI dev server redirects."""
    if getattr(args, "webui_dev_public_url", None):
        return str(args.webui_dev_public_url).rstrip("/")

    host = str(getattr(args, "webui_dev_host", "127.0.0.1")).strip()
    if host in ("0.0.0.0", "::", ""):
        host = "localhost"
    port = int(getattr(args, "webui_dev_port", 5173))
    return f"http://{host}:{port}/webui"


def start_webui_dev_server(args) -> subprocess.Popen | None:
    """Start WebUI dev server via command line (`bun run dev`)."""
    if not getattr(args, "webui_dev", False):
        return None

    if "LIGHTRAG_GUNICORN_MODE" in os.environ or "GUNICORN_CMD_ARGS" in os.environ:
        logger.warning(
            "WEBUI_DEV is enabled but server is running under Gunicorn; skip launching WebUI dev process."
        )
        return None

    webui_source_dir = Path(__file__).parent.parent.parent / "lightrag_webui"
    if not webui_source_dir.exists():
        raise RuntimeError(
            f"WEBUI_DEV enabled but WebUI source directory not found: {webui_source_dir}"
        )

    bun_path = shutil.which("bun")
    if not bun_path:
        raise RuntimeError(
            "WEBUI_DEV enabled but `bun` is not available in PATH. Install Bun first."
        )

    host = str(getattr(args, "webui_dev_host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(getattr(args, "webui_dev_port", 5173))
    command = [bun_path, "run", "dev", "--", "--host", host, "--port", str(port)]

    logger.info(
        f"Starting WebUI dev server with command: {' '.join(command)} (cwd={webui_source_dir})"
    )
    process = subprocess.Popen(command, cwd=str(webui_source_dir))
    logger.info(
        f"WebUI dev server started (pid={process.pid}), public URL: {get_webui_dev_public_url(args)}"
    )
    return process


def stop_webui_dev_server(process: subprocess.Popen | None):
    """Stop WebUI dev server process if it's still running."""
    if process is None:
        return
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=8)
        logger.info("WebUI dev server process terminated")
    except subprocess.TimeoutExpired:
        process.kill()
        logger.warning("WebUI dev server did not exit in time; process was killed")
    except Exception as e:
        logger.warning(f"Failed to stop WebUI dev server process cleanly: {e}")


def create_app(args):
    webui_dev_enabled = bool(getattr(args, "webui_dev", False))
    webui_dev_public_url = get_webui_dev_public_url(args)

    # In WEBUI_DEV mode, disable static WebUI mounting and use external Vite dev server.
    webui_assets_exist = False
    is_frontend_outdated = False
    if not webui_dev_enabled:
        webui_assets_exist, is_frontend_outdated = check_frontend_build()
    else:
        logger.info(
            "WEBUI_DEV is enabled. Static WebUI mount is disabled and requests will be redirected to dev server."
        )

    # Create unified API version display with warning symbol if frontend is outdated
    api_version_display = (
        f"{__api_version__}⚠️" if is_frontend_outdated else __api_version__
    )

    # Setup logging
    logger.setLevel(args.log_level)
    set_verbose_debug(args.verbose)

    # Create configuration cache (this will output configuration logs)
    config_cache = LLMConfigCache(args)

    # Verify that bindings are correctly setup
    if args.llm_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "aws_bedrock",
        "gemini",
        "qwen",
    ]:
        raise Exception("llm binding not supported")

    if args.embedding_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "aws_bedrock",
        "jina",
        "gemini",
        "qwen",
    ]:
        raise Exception("embedding binding not supported")

    # Set default hosts if not provided
    if args.llm_binding_host is None:
        args.llm_binding_host = get_default_host(args.llm_binding)

    if args.embedding_binding_host is None:
        args.embedding_binding_host = get_default_host(args.embedding_binding)

    # Add SSL validation
    if args.ssl:
        if not args.ssl_certfile or not args.ssl_keyfile:
            raise Exception(
                "SSL certificate and key files must be provided when SSL is enabled"
            )
        if not os.path.exists(args.ssl_certfile):
            raise Exception(f"SSL certificate file not found: {args.ssl_certfile}")
        if not os.path.exists(args.ssl_keyfile):
            raise Exception(f"SSL key file not found: {args.ssl_keyfile}")

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    workspace_config_path = os.getenv("LIGHTRAG_WORKSPACE_CONFIG", "").strip()
    workspace_config: dict[str, Any] | None = None
    if workspace_config_path:
        config_file = Path(workspace_config_path)
        if config_file.exists():
            try:
                workspace_config = load_workspace_config(config_file)
            except Exception as exc:
                logger.warning(
                    f"Failed to load workspace config from {config_file}: {exc}"
                )

    workspace_definitions: list[WorkspaceDefinition] = parse_workspace_csv(args.workspace)
    if workspace_config and workspace_config.get("workspaces"):
        configured_definitions = {
            item.id: item for item in workspace_config["workspaces"]
        }
        workspace_definitions = [
            configured_definitions.get(item.id, item) for item in workspace_definitions
        ]

    workspace_names = [item.id for item in workspace_definitions]
    default_workspace_name = workspace_names[0]
    workspace_aliases = build_workspace_alias_map(workspace_definitions)
    workspace_metadata = {
        item.id: {
            "id": display_workspace_id(item.id),
            "alias": item.alias,
        }
        for item in workspace_definitions
    }
    workspace_display_names = {
        item.id: item.alias
        for item in workspace_definitions
    }
    workspace_rags: dict[str, LightRAG] = {}
    workspace_doc_managers: dict[str, DocumentManager] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Store background tasks
        app.state.background_tasks = set()
        webui_dev_process: subprocess.Popen | None = None

        try:
            if webui_dev_enabled:
                webui_dev_process = start_webui_dev_server(args)

            # Initialize all workspace-bound LightRAG instances.
            for workspace_name, rag_instance in workspace_rags.items():
                logger.info(
                    f"Initializing storages for workspace '{workspace_name or 'default'}'"
                )
                await rag_instance.initialize_storages()

                # Data migration regardless of storage implementation
                await rag_instance.check_and_migrate_data()

            ASCIIColors.green("\nServer is ready to accept connections! 🚀\n")

            yield

        finally:
            # Clean up all workspace-bound LightRAG instances.
            for workspace_name, rag_instance in workspace_rags.items():
                try:
                    await rag_instance.finalize_storages()
                except Exception as e:
                    logger.error(
                        f"Failed to finalize storages for workspace '{workspace_name or 'default'}': {e}"
                    )

            if "LIGHTRAG_GUNICORN_MODE" not in os.environ:
                # Only perform cleanup in Uvicorn single-process mode
                logger.debug("Unvicorn Mode: finalizing shared storage...")
                finalize_share_data()
            else:
                # In Gunicorn mode with preload_app=True, cleanup is handled by on_exit hooks
                logger.debug(
                    "Gunicorn Mode: postpone shared storage finalization to master process"
                )

            if webui_dev_enabled:
                stop_webui_dev_server(webui_dev_process)

            try:
                shutdown_local_model_manager()
            except Exception as e:
                logger.error(f"Failed to shutdown local model manager: {e}")

            try:
                shutdown_local_model_process_manager()
            except Exception as e:
                logger.error(f"Failed to shutdown local model process manager: {e}")

    # Initialize FastAPI
    base_description = (
        "Providing API for LightRAG core, Web UI and Ollama Model Emulation"
    )
    swagger_description = (
        base_description
        + (" (API-Key Enabled)" if api_key else "")
        + "\n\n[View ReDoc documentation](/redoc)"
    )
    app_kwargs = {
        "title": "LightRAG Server API",
        "description": swagger_description,
        "version": __api_version__,
        "openapi_url": "/openapi.json",  # Explicitly set OpenAPI schema URL
        "docs_url": None,  # Disable default docs, we'll create custom endpoint
        "redoc_url": "/redoc",  # Explicitly set redoc URL
        "lifespan": lifespan,
    }

    # Configure Swagger UI parameters
    # Enable persistAuthorization and tryItOutEnabled for better user experience
    app_kwargs["swagger_ui_parameters"] = {
        "persistAuthorization": True,
        "tryItOutEnabled": True,
    }

    app = FastAPI(**app_kwargs)

    def _suggest_api_prefixed_path(path: str) -> str | None:
        """Suggest `/api`-prefixed path for common API roots when missing."""
        api_roots = ("/documents", "/query", "/graphs", "/graph")
        if path.startswith("/api/"):
            return None
        if any(path.startswith(root) for root in api_roots):
            return f"/api{path}"
        return None

    # Add custom validation error handler for /query/data endpoint
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        # Check if this is a request to /query/data endpoint
        if request.url.path.endswith("/query/data"):
            # Extract error details
            error_details = []
            for error in exc.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                error_details.append(f"{field_path}: {error['msg']}")

            error_message = "; ".join(error_details)

            # Return in the expected format for /query/data
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failure",
                    "message": f"Validation error: {error_message}",
                    "data": {},
                    "metadata": {},
                },
            )
        else:
            # For other endpoints, return the default FastAPI validation error
            return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Return richer error payload for HTTP errors such as 404/405."""
        path = request.url.path
        query = request.url.query
        suggestion = _suggest_api_prefixed_path(path) if exc.status_code == 404 else None

        response_body: dict[str, Any] = {
            "detail": exc.detail if exc.detail is not None else "HTTP error",
            "status_code": exc.status_code,
            "method": request.method,
            "path": path,
            "query": query,
        }
        if suggestion:
            response_body["hint"] = f"Route not found. Did you mean '{suggestion}'?"

        log_message = (
            f"HTTP {exc.status_code} {request.method} {path}"
            + (f"?{query}" if query else "")
        )
        if suggestion:
            logger.warning(f"{log_message} | hint={suggestion}")
        elif exc.status_code >= 500:
            logger.error(log_message)
        else:
            logger.warning(log_message)

        return JSONResponse(status_code=exc.status_code, content=response_body)

    def get_cors_origins():
        """Get allowed origins from global_args
        Returns a normalized list of allowed origins.
        """
        origins_value = global_args.cors_origins
        if isinstance(origins_value, (list, tuple, set)):
            origins = [
                str(origin).strip()
                for origin in origins_value
                if str(origin).strip()
            ]
        else:
            origins_str = str(origins_value or "").strip()
            if not origins_str:
                return []
            origins = [
                origin.strip()
                for origin in origins_str.replace(";", ",").split(",")
                if origin.strip()
            ]

        if "*" in origins:
            return ["*"]
        return origins

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-New-Token"
        ],  # Expose token renewal header for cross-origin requests
    )

    # Create combined auth dependency for all endpoints
    combined_auth = get_combined_auth_dependency(api_key)

    def get_workspace_from_request(request: Request) -> str | None:
        """
        Extract workspace from request query/header.

        Priority:
        1. `workspace` query parameter
        2. `LIGHTRAG-WORKSPACE` header
        3. fallback to default workspace in caller

        Args:
            request: FastAPI Request object

        Returns:
            Workspace identifier (may be empty string for global namespace)
        """
        workspace = request.query_params.get("workspace", "").strip()
        if not workspace:
            workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()
        if workspace and workspace not in workspace_rags:
            workspace = workspace_aliases.get(workspace, workspace)

        if not workspace:
            workspace = None

        return workspace

    def resolve_postgres_workspace_storage(rag_instance: LightRAG, endpoint: str):
        for attr_name in ("text_chunks", "llm_response_cache", "doc_status"):
            storage = getattr(rag_instance, attr_name, None)
            db = getattr(storage, "db", None)
            if callable(getattr(db, "query", None)) and callable(
                getattr(db, "execute", None)
            ):
                return storage

        raise HTTPException(
            status_code=501,
            detail=f"The {endpoint} endpoint currently requires PostgreSQL-backed storage.",
        )

    async def get_workspace_descriptions() -> dict[str, str]:
        if not workspace_rags:
            return {}

        storage = resolve_postgres_workspace_storage(
            workspace_rags[default_workspace_name],
            "/api/workspaces",
        )
        records = await storage.db.query(
            """
            SELECT workspace, COALESCE(description, '') AS description
            FROM LIGHTRAG_WORKSPACE_INFO
            WHERE workspace = ANY($1)
            """,
            [workspace_names],
            multirows=True,
        )
        return {
            str(record.get("workspace") or ""): str(record.get("description") or "")
            for record in (records or [])
        }

    async def upsert_workspace_description(
        workspace_name: str,
        description: str,
    ) -> str:
        storage = resolve_postgres_workspace_storage(
            workspace_rags[workspace_name],
            "/api/workspaces/{workspace}/description",
        )
        normalized_description = str(description or "").strip()
        result = await storage.db.query(
            """
            INSERT INTO LIGHTRAG_WORKSPACE_INFO (workspace, description)
            VALUES ($1, $2)
            ON CONFLICT (workspace) DO UPDATE
            SET
                description = EXCLUDED.description,
                updated_at = CURRENT_TIMESTAMP
            RETURNING COALESCE(description, '') AS description
            """,
            [workspace_name, normalized_description],
        )
        if not result:
            raise RuntimeError("Failed to persist workspace description")
        return str(result.get("description") or "")

    def format_workspace_info_item(
        workspace_name: str,
        description: str,
    ) -> WorkspaceInfoItem:
        meta = workspace_metadata.get(workspace_name) or {
            "id": display_workspace_id(workspace_name),
            "alias": display_workspace_id(workspace_name),
        }
        normalized_description = str(description or "").strip()
        return WorkspaceInfoItem(
            id=meta["id"],
            alias=meta["alias"],
            description=normalized_description,
            has_description=bool(normalized_description),
        )

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    def create_optimized_openai_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized OpenAI LLM function with pre-processed configuration"""

        async def optimized_openai_alike_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            from lightrag.llm.openai import openai_complete_if_cache

            keyword_extraction = kwargs.pop("keyword_extraction", None)
            if keyword_extraction:
                kwargs["response_format"] = GPTKeywordExtractionFormat
            if history_messages is None:
                history_messages = []

            # Use pre-processed configuration to avoid repeated parsing
            kwargs["timeout"] = llm_timeout
            if config_cache.openai_llm_options:
                kwargs.update(config_cache.openai_llm_options)

            return await openai_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=args.llm_binding_host,
                api_key=args.llm_binding_api_key,
                **kwargs,
            )

        return optimized_openai_alike_model_complete

    def create_optimized_azure_openai_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized Azure OpenAI LLM function with pre-processed configuration"""

        async def optimized_azure_openai_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            from lightrag.llm.azure_openai import azure_openai_complete_if_cache

            keyword_extraction = kwargs.pop("keyword_extraction", None)
            if keyword_extraction:
                kwargs["response_format"] = GPTKeywordExtractionFormat
            if history_messages is None:
                history_messages = []

            # Use pre-processed configuration to avoid repeated parsing
            kwargs["timeout"] = llm_timeout
            if config_cache.openai_llm_options:
                kwargs.update(config_cache.openai_llm_options)

            return await azure_openai_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=args.llm_binding_host,
                api_key=os.getenv("AZURE_OPENAI_API_KEY", args.llm_binding_api_key),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                **kwargs,
            )

        return optimized_azure_openai_model_complete

    def create_optimized_gemini_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized Gemini LLM function with cached configuration"""

        async def optimized_gemini_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            from lightrag.llm.gemini import gemini_complete_if_cache

            if history_messages is None:
                history_messages = []

            # Use pre-processed configuration to avoid repeated parsing
            kwargs["timeout"] = llm_timeout
            if (
                config_cache.gemini_llm_options is not None
                and "generation_config" not in kwargs
            ):
                kwargs["generation_config"] = dict(config_cache.gemini_llm_options)

            return await gemini_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=args.llm_binding_api_key,
                base_url=args.llm_binding_host,
                keyword_extraction=keyword_extraction,
                **kwargs,
            )

        return optimized_gemini_model_complete

    def create_llm_model_func(binding: str):
        """
        Create LLM model function based on binding type.
        Uses optimized functions for OpenAI bindings and lazy import for others.
        """
        try:
            if binding == "lollms":
                from lightrag.llm.lollms import lollms_model_complete

                return lollms_model_complete
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_model_complete

                return ollama_model_complete
            elif binding == "aws_bedrock":
                return bedrock_model_complete  # Already defined locally
            elif binding == "azure_openai":
                # Use optimized function with pre-processed configuration
                return create_optimized_azure_openai_llm_func(
                    config_cache, args, llm_timeout
                )
            elif binding == "gemini":
                return create_optimized_gemini_llm_func(config_cache, args, llm_timeout)
            elif binding == "qwen":
                raise Exception(
                    "qwen binding currently supports embedding and reranking only"
                )
            else:  # openai and compatible
                # Use optimized function with pre-processed configuration
                return create_optimized_openai_llm_func(config_cache, args, llm_timeout)
        except ImportError as e:
            raise Exception(f"Failed to import {binding} LLM binding: {e}")

    def create_llm_model_kwargs(binding: str, args, llm_timeout: int) -> dict:
        """
        Create LLM model kwargs based on binding type.
        Uses lazy import for binding-specific options.
        """
        if binding in ["lollms", "ollama"]:
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions

                return {
                    "host": args.llm_binding_host,
                    "timeout": llm_timeout,
                    "options": OllamaLLMOptions.options_dict(args),
                    "api_key": args.llm_binding_api_key,
                }
            except ImportError as e:
                raise Exception(f"Failed to import {binding} options: {e}")
        return {}

    def create_optimized_embedding_function(
        config_cache: LLMConfigCache, binding, model, host, api_key, args
    ) -> EmbeddingFunc:
        """
        Create optimized embedding function and return an EmbeddingFunc instance
        with proper max_token_size inheritance from provider defaults.

        This function:
        1. Imports the provider embedding function
        2. Extracts max_token_size and embedding_dim from provider if it's an EmbeddingFunc
        3. Creates an optimized wrapper that calls the underlying function directly (avoiding double-wrapping)
        4. Returns a properly configured EmbeddingFunc instance

        Configuration Rules:
        - When EMBEDDING_MODEL is not set: Uses provider's default model and dimension
          (e.g., jina-embeddings-v4 with 2048 dims, text-embedding-3-small with 1536 dims)
        - When EMBEDDING_MODEL is set to a custom model: User MUST also set EMBEDDING_DIM
          to match the custom model's dimension (e.g., for jina-embeddings-v3, set EMBEDDING_DIM=1024)

        Note: The embedding_dim parameter is automatically injected by EmbeddingFunc wrapper
        when send_dimensions=True (enabled for Jina and Gemini bindings). This wrapper calls
        the underlying provider function directly (.func) to avoid double-wrapping, so we must
        explicitly pass embedding_dim to the provider's underlying function.
        """

        # Step 1: Import provider function and extract default attributes
        provider_func = None
        provider_max_token_size = None
        provider_embedding_dim = None
        provider_vlm_enable = False

        try:
            if binding == "openai":
                from lightrag.llm.openai import openai_embed

                provider_func = openai_embed
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_embed

                provider_func = ollama_embed
            elif binding == "gemini":
                from lightrag.llm.gemini import gemini_embed

                provider_func = gemini_embed
            elif binding == "jina":
                from lightrag.llm.jina import jina_embed

                provider_func = jina_embed
            elif binding == "azure_openai":
                from lightrag.llm.azure_openai import azure_openai_embed

                provider_func = azure_openai_embed
            elif binding == "aws_bedrock":
                from lightrag.llm.bedrock import bedrock_embed

                provider_func = bedrock_embed
            elif binding == "lollms":
                from lightrag.llm.lollms import lollms_embed

                provider_func = lollms_embed
            elif binding == "qwen":
                from lightrag.llm.transforms_qwen import qwen_embed

                provider_func = qwen_embed

            # Extract attributes if provider is an EmbeddingFunc
            if provider_func and isinstance(provider_func, EmbeddingFunc):
                provider_max_token_size = provider_func.max_token_size
                provider_embedding_dim = provider_func.embedding_dim
                provider_vlm_enable = provider_func.vlm_enable
                logger.debug(
                    f"Extracted from {binding} provider: "
                    f"max_token_size={provider_max_token_size}, "
                    f"embedding_dim={provider_embedding_dim}"
                )
        except ImportError as e:
            logger.warning(f"Could not import provider function for {binding}: {e}")

        # Step 2: Apply priority (user config > provider default)
        # For max_token_size: explicit env var > provider default > None
        final_max_token_size = args.embedding_token_limit or provider_max_token_size
        # For embedding_dim: user config (always has value) takes priority
        # Only use provider default if user config is explicitly None (which shouldn't happen)
        final_embedding_dim = (
            args.embedding_dim if args.embedding_dim else provider_embedding_dim
        )
        fallback_embedding_dim = final_embedding_dim or provider_embedding_dim or 1024

        def _sanitize_embedding_output(raw_result: Any, texts_count: int) -> np.ndarray:
            """Convert embedding output to ndarray and replace non-finite values with 0."""
            arr = np.asarray(raw_result, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            non_finite_mask = ~np.isfinite(arr)
            non_finite_count = int(np.count_nonzero(non_finite_mask))
            if non_finite_count > 0:
                logger.warning(
                    "Embedding output contains %d non-finite values (binding=%s, model=%s, texts=%d); replaced with 0",
                    non_finite_count,
                    binding,
                    model,
                    texts_count,
                )
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            return arr

        # Step 3: Create optimized embedding function (calls underlying function directly)
        # Note: When model is None, each binding will use its own default model
        async def optimized_embedding_function(
            texts, embedding_dim=None, max_token_size=None
        ):
            try:
                if binding == "lollms":
                    from lightrag.llm.lollms import lollms_embed

                    # Get real function, skip EmbeddingFunc wrapper if present
                    actual_func = (
                        lollms_embed.func
                        if isinstance(lollms_embed, EmbeddingFunc)
                        else lollms_embed
                    )
                    # lollms embed_model is not used (server uses configured vectorizer)
                    # Only pass base_url and api_key
                    raw_result = await actual_func(
                        texts, base_url=host, api_key=api_key
                    )
                    return _sanitize_embedding_output(raw_result, len(texts))
                elif binding == "ollama":
                    from lightrag.llm.ollama import ollama_embed

                    # Get real function, skip EmbeddingFunc wrapper if present
                    actual_func = (
                        ollama_embed.func
                        if isinstance(ollama_embed, EmbeddingFunc)
                        else ollama_embed
                    )

                    # Use pre-processed configuration if available
                    if config_cache.ollama_embedding_options is not None:
                        ollama_options = config_cache.ollama_embedding_options
                    else:
                        from lightrag.llm.binding_options import OllamaEmbeddingOptions

                        ollama_options = OllamaEmbeddingOptions.options_dict(args)

                    # Pass embed_model only if provided, let function use its default (bge-m3:latest)
                    kwargs = {
                        "texts": texts,
                        "host": host,
                        "api_key": api_key,
                        "options": ollama_options,
                    }
                    if model:
                        kwargs["embed_model"] = model
                    raw_result = await actual_func(**kwargs)
                    return _sanitize_embedding_output(raw_result, len(texts))
                elif binding == "azure_openai":
                    from lightrag.llm.azure_openai import azure_openai_embed

                    actual_func = (
                        azure_openai_embed.func
                        if isinstance(azure_openai_embed, EmbeddingFunc)
                        else azure_openai_embed
                    )
                    # Pass model only if provided, let function use its default otherwise
                    kwargs = {"texts": texts, "api_key": api_key}
                    if model:
                        kwargs["model"] = model
                    raw_result = await actual_func(**kwargs)
                    return _sanitize_embedding_output(raw_result, len(texts))
                elif binding == "aws_bedrock":
                    from lightrag.llm.bedrock import bedrock_embed

                    actual_func = (
                        bedrock_embed.func
                        if isinstance(bedrock_embed, EmbeddingFunc)
                        else bedrock_embed
                    )
                    # Pass model only if provided, let function use its default otherwise
                    kwargs = {"texts": texts}
                    if model:
                        kwargs["model"] = model
                    raw_result = await actual_func(**kwargs)
                    return _sanitize_embedding_output(raw_result, len(texts))
                elif binding == "jina":
                    from lightrag.llm.jina import jina_embed

                    actual_func = (
                        jina_embed.func
                        if isinstance(jina_embed, EmbeddingFunc)
                        else jina_embed
                    )
                    # Pass model only if provided, let function use its default (jina-embeddings-v4)
                    kwargs = {
                        "texts": texts,
                        "embedding_dim": embedding_dim,
                        "base_url": host,
                        "api_key": api_key,
                    }
                    if model:
                        kwargs["model"] = model
                    raw_result = await actual_func(**kwargs)
                    return _sanitize_embedding_output(raw_result, len(texts))
                elif binding == "gemini":
                    from lightrag.llm.gemini import gemini_embed

                    actual_func = (
                        gemini_embed.func
                        if isinstance(gemini_embed, EmbeddingFunc)
                        else gemini_embed
                    )

                    # Use pre-processed configuration if available
                    if config_cache.gemini_embedding_options is not None:
                        gemini_options = config_cache.gemini_embedding_options
                    else:
                        from lightrag.llm.binding_options import GeminiEmbeddingOptions

                        gemini_options = GeminiEmbeddingOptions.options_dict(args)

                    # Pass model only if provided, let function use its default (gemini-embedding-001)
                    kwargs = {
                        "texts": texts,
                        "base_url": host,
                        "api_key": api_key,
                        "embedding_dim": embedding_dim,
                        "task_type": gemini_options.get(
                            "task_type", "RETRIEVAL_DOCUMENT"
                        ),
                    }
                    if model:
                        kwargs["model"] = model
                    raw_result = await actual_func(**kwargs)
                    return _sanitize_embedding_output(raw_result, len(texts))
                elif binding == "qwen":
                    from lightrag.llm.transforms_qwen import qwen_embed

                    actual_func = (
                        qwen_embed.func
                        if isinstance(qwen_embed, EmbeddingFunc)
                        else qwen_embed
                    )
                    kwargs = {
                        "texts": texts,
                        "embedding_dim": embedding_dim,
                    }
                    if max_token_size is not None:
                        kwargs["max_token_size"] = max_token_size
                    if model:
                        kwargs["model"] = model
                    raw_result = await actual_func(**kwargs)
                    return _sanitize_embedding_output(raw_result, len(texts))
                else:  # openai and compatible
                    from lightrag.llm.openai import openai_embed

                    actual_func = (
                        openai_embed.func
                        if isinstance(openai_embed, EmbeddingFunc)
                        else openai_embed
                    )
                    # Pass model only if provided, let function use its default (text-embedding-3-small)
                    kwargs = {
                        "texts": texts,
                        "base_url": host,
                        "api_key": api_key,
                        "embedding_dim": embedding_dim,
                    }
                    if model:
                        kwargs["model"] = model
                    raw_result = await actual_func(**kwargs)
                    return _sanitize_embedding_output(raw_result, len(texts))
            except ImportError as e:
                raise Exception(f"Failed to import {binding} embedding: {e}")
            except Exception as e:
                # Ollama may fail with HTTP 500 "unsupported value: NaN" on its side.
                # Fallback to zero vectors to keep indexing pipeline alive.
                if binding == "ollama" and "unsupported value: NaN" in str(e):
                    logger.warning(
                        "Ollama embedding failed with NaN serialization error; fallback to zero vectors (texts=%d, dim=%d, model=%s, host=%s)",
                        len(texts),
                        fallback_embedding_dim,
                        model,
                        host,
                    )
                    return np.zeros(
                        (len(texts), fallback_embedding_dim), dtype=np.float32
                    )
                raise

        # Step 4: Wrap in EmbeddingFunc and return
        embedding_func_instance = EmbeddingFunc(
            embedding_dim=final_embedding_dim,
            func=optimized_embedding_function,
            max_token_size=final_max_token_size,
            send_dimensions=False,  # Will be set later based on binding requirements
            vlm_enable=provider_vlm_enable,
            model_name=model,
        )

        # Log final embedding configuration
        logger.info(
            f"Embedding config: binding={binding} model={model} "
            f"embedding_dim={final_embedding_dim} max_token_size={final_max_token_size}"
        )

        return embedding_func_instance

    llm_timeout = get_env_value("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT, int)
    embedding_timeout = get_env_value(
        "EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT, int
    )

    async def bedrock_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        # Lazy import
        from lightrag.llm.bedrock import bedrock_complete_if_cache

        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []

        # Use global temperature for Bedrock
        kwargs["temperature"] = get_env_value("BEDROCK_LLM_TEMPERATURE", 1.0, float)

        return await bedrock_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    # Create embedding function with optimized configuration and max_token_size inheritance
    import inspect

    # Create the EmbeddingFunc instance (now returns complete EmbeddingFunc with max_token_size)
    embedding_func = create_optimized_embedding_function(
        config_cache=config_cache,
        binding=args.embedding_binding,
        model=args.embedding_model,
        host=args.embedding_binding_host,
        api_key=args.embedding_binding_api_key,
        args=args,
    )

    # Get embedding_send_dim from centralized configuration
    embedding_send_dim = args.embedding_send_dim

    # Check if the underlying function signature has embedding_dim parameter
    sig = inspect.signature(embedding_func.func)
    has_embedding_dim_param = "embedding_dim" in sig.parameters

    # Determine send_dimensions value based on binding type
    # Jina and Gemini REQUIRE dimension parameter (forced to True)
    # OpenAI and others: controlled by EMBEDDING_SEND_DIM environment variable
    if args.embedding_binding in ["jina", "gemini", "qwen"]:
        # Jina and Gemini APIs require dimension parameter - always send it
        send_dimensions = has_embedding_dim_param
        dimension_control = f"forced by {args.embedding_binding.title()} API"
    else:
        # For OpenAI and other bindings, respect EMBEDDING_SEND_DIM setting
        send_dimensions = embedding_send_dim and has_embedding_dim_param
        if send_dimensions or not embedding_send_dim:
            dimension_control = "by env var"
        else:
            dimension_control = "by not hasparam"

    # Set send_dimensions on the EmbeddingFunc instance
    embedding_func.send_dimensions = send_dimensions

    logger.info(
        f"Send embedding dimension: {send_dimensions} {dimension_control} "
        f"(dimensions={embedding_func.embedding_dim}, has_param={has_embedding_dim_param}, "
        f"binding={args.embedding_binding})"
    )

    # Log max_token_size source
    if embedding_func.max_token_size:
        source = (
            "env variable"
            if args.embedding_token_limit
            else f"{args.embedding_binding} provider default"
        )
        logger.info(
            f"Embedding max_token_size: {embedding_func.max_token_size} (from {source})"
        )
    else:
        logger.info(
            "Embedding max_token_size: None (Embedding token limit is disabled)."
        )

    # Configure rerank function based on args.rerank_bindingparameter
    rerank_model_func = None
    if args.rerank_binding != "null":
        from lightrag.rerank import cohere_rerank, jina_rerank, ali_rerank
        from lightrag.llm.transforms_qwen import qwen_rerank

        # Map rerank binding to corresponding function
        rerank_functions = {
            "cohere": cohere_rerank,
            "jina": jina_rerank,
            "aliyun": ali_rerank,
            "qwen": qwen_rerank,
        }

        # Select the appropriate rerank function based on binding
        selected_rerank_func = rerank_functions.get(args.rerank_binding)
        if not selected_rerank_func:
            logger.error(f"Unsupported rerank binding: {args.rerank_binding}")
            raise ValueError(f"Unsupported rerank binding: {args.rerank_binding}")

        # Get default values from selected_rerank_func if args values are None
        if args.rerank_model is None or args.rerank_binding_host is None:
            sig = inspect.signature(selected_rerank_func)

            # Set default model if args.rerank_model is None
            if args.rerank_model is None and "model" in sig.parameters:
                default_model = sig.parameters["model"].default
                if default_model != inspect.Parameter.empty:
                    args.rerank_model = default_model

            # Set default base_url if args.rerank_binding_host is None
            if args.rerank_binding_host is None and "base_url" in sig.parameters:
                default_base_url = sig.parameters["base_url"].default
                if default_base_url != inspect.Parameter.empty:
                    args.rerank_binding_host = default_base_url

        async def server_rerank_func(
            query: str, documents: list, top_n: int = None, extra_body: dict = None
        ):
            """Server rerank function with configuration from environment variables"""
            # Prepare kwargs for rerank function
            kwargs = {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "api_key": args.rerank_binding_api_key,
                "model": args.rerank_model,
                "base_url": args.rerank_binding_host,
            }

            # Add Cohere-specific parameters if using cohere binding
            if args.rerank_binding == "cohere":
                # Enable chunking if configured (useful for models with token limits like ColBERT)
                kwargs["enable_chunking"] = (
                    os.getenv("RERANK_ENABLE_CHUNKING", "false").lower() == "true"
                )
                kwargs["max_tokens_per_doc"] = int(
                    os.getenv("RERANK_MAX_TOKENS_PER_DOC", "4096")
                )

            return await selected_rerank_func(**kwargs, extra_body=extra_body)

        server_rerank_func.vlm_enable = bool(
            getattr(selected_rerank_func, "vlm_enable", False)
        )
        rerank_model_func = server_rerank_func
        logger.info(
            f"Reranking is enabled: {args.rerank_model or 'default model'} using {args.rerank_binding} provider"
        )
    else:
        logger.info("Reranking is disabled")

    # Create ollama_server_infos from command line arguments
    from lightrag.api.config import OllamaServerInfos

    ollama_server_infos = OllamaServerInfos(
        name=args.simulated_model_name, tag=args.simulated_model_tag
    )

    # Initialize one LightRAG instance per workspace.
    try:
        for workspace_name in workspace_names:
            rag_instance = LightRAG(
                working_dir=args.working_dir,
                workspace=workspace_name,
                llm_model_func=create_llm_model_func(args.llm_binding),
                llm_model_name=args.llm_model,
                llm_model_max_async=args.max_async,
                summary_max_tokens=args.summary_max_tokens,
                summary_context_size=args.summary_context_size,
                chunk_token_size=int(args.chunk_size),
                chunk_overlap_token_size=int(args.chunk_overlap_size),
                llm_model_kwargs=create_llm_model_kwargs(
                    args.llm_binding, args, llm_timeout
                ),
                embedding_func=embedding_func,
                default_llm_timeout=llm_timeout,
                default_embedding_timeout=embedding_timeout,
                kv_storage=args.kv_storage,
                graph_storage=args.graph_storage,
                vector_storage=args.vector_storage,
                doc_status_storage=args.doc_status_storage,
                vector_db_storage_cls_kwargs={
                    "cosine_better_than_threshold": args.cosine_threshold
                },
                enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
                enable_llm_cache=args.enable_llm_cache,
                rerank_model_func=rerank_model_func,
                max_parallel_insert=args.max_parallel_insert,
                max_graph_nodes=args.max_graph_nodes,
                addon_params={
                    "language": args.summary_language,
                    "entity_types": args.entity_types,
                },
                ollama_server_infos=ollama_server_infos,
            )
            workspace_rags[workspace_name] = rag_instance
            workspace_doc_managers[workspace_name] = DocumentManager(
                args.input_dir, workspace=workspace_name
            )

            logger.info(
                f"Workspace '{workspace_name or 'default'}' initialized for dynamic workspace routing"
            )
    except Exception as e:
        logger.error(f"Failed to initialize LightRAG instances: {e}")
        raise

    # Register a single route set under /api and dispatch by workspace parameter/header.
    app.include_router(
        create_document_routes(
            workspace_rags,
            workspace_doc_managers,
            api_key,
            workspace=default_workspace_name,
            workspace_aliases=workspace_aliases,
        ),
        prefix="/api",
    )
    app.include_router(
        create_query_routes(
            workspace_rags,
            api_key,
            args.top_k,
            workspace=default_workspace_name,
            workspace_aliases=workspace_aliases,
        ),
        prefix="/api",
    )
    app.include_router(
        create_graph_routes(
            workspace_rags,
            api_key,
            workspace=default_workspace_name,
            workspace_aliases=workspace_aliases,
        ),
        prefix="/api",
    )
    app.include_router(
        create_knowledge_base_qa_routes(
            workspace_rags,
            api_key,
            workspace=default_workspace_name,
            workspace_aliases=workspace_aliases,
        ),
        prefix="/api",
    )
    app.include_router(
        create_agent_routes(
            workspace_rags,
            api_key,
            workspace=default_workspace_name,
            workspace_aliases=workspace_aliases,
            workspace_display_names=workspace_display_names,
        ),
        prefix="/api",
    )

    # Keep Ollama API bound to the default workspace instance for compatibility.
    rag = workspace_rags[default_workspace_name]
    ollama_api = OllamaAPI(rag, top_k=args.top_k, api_key=api_key)
    app.include_router(ollama_api.router, prefix="/api")

    # Custom Swagger UI endpoint for offline support
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Custom Swagger UI HTML with local static files"""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url="/docs/oauth2-redirect",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
            swagger_ui_parameters=app.swagger_ui_parameters,
        )

    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    async def swagger_ui_redirect():
        """OAuth2 redirect for Swagger UI"""
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/")
    async def redirect_to_webui():
        """Redirect root path based on WebUI availability"""
        if webui_dev_enabled:
            return RedirectResponse(url=webui_dev_public_url)
        if webui_assets_exist:
            return RedirectResponse(url="/webui")
        else:
            return RedirectResponse(url="/docs")

    @app.get("/auth-status")
    async def get_auth_status():
        """Get authentication status and guest token if auth is not configured"""

        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "auth_configured": False,
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }

        return {
            "auth_configured": True,
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.get(
        "/api/workspaces",
        dependencies=[Depends(combined_auth)],
        summary="List available workspaces",
        description="Returns all loaded workspace identifiers and the default workspace.",
    )
    async def list_workspaces():
        try:
            descriptions = await get_workspace_descriptions()
        except HTTPException as exc:
            if exc.status_code != 501:
                raise
            descriptions = {}
        except Exception as exc:
            logger.warning("Failed to load workspace descriptions: %s", exc)
            descriptions = {}

        workspace_list = [
            format_workspace_info_item(name, descriptions.get(name, "")).model_dump()
            for name in workspace_names
        ]
        default_workspace_meta = format_workspace_info_item(
            default_workspace_name,
            descriptions.get(default_workspace_name, ""),
        )
        return {
            "default_workspace": default_workspace_meta.id,
            "default_workspace_alias": default_workspace_meta.alias,
            "workspaces": workspace_list,
            "count": len(workspace_list),
        }

    def resolve_workspace_or_raise(
        request: Request, explicit_workspace: str | None
    ) -> str:
        workspace = (explicit_workspace or "").strip()
        has_explicit_workspace = bool(workspace)
        if has_explicit_workspace and workspace not in workspace_rags:
            workspace = workspace_aliases.get(workspace, workspace)
        if not has_explicit_workspace:
            workspace = get_workspace_from_request(request)
        if workspace is None:
            workspace = default_workspace_name

        if workspace not in workspace_rags:
            supported = ", ".join(
                workspace_metadata.get(name, {}).get("id", display_workspace_id(name))
                for name in workspace_names
            )
            raise HTTPException(
                status_code=400,
                detail=f"Unknown workspace '{workspace}'. Available workspaces: {supported}",
            )

        return workspace

    @app.put(
        "/api/workspaces/{workspace}/description",
        dependencies=[Depends(combined_auth)],
        response_model=WorkspaceInfoItem,
        summary="Update workspace description",
        description="Stores a human-maintained description for the selected workspace.",
    )
    async def update_workspace_description(
        request: Request,
        workspace: str,
        payload: WorkspaceDescriptionUpdateRequest,
    ):
        resolved_workspace = resolve_workspace_or_raise(request, workspace)
        try:
            description = await upsert_workspace_description(
                resolved_workspace,
                payload.description,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to update description for workspace '%s': %s",
                resolved_workspace,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update workspace description: {exc}",
            ) from exc

        return format_workspace_info_item(resolved_workspace, description)

    @app.post(
        "/api/chunks/paginated",
        dependencies=[Depends(combined_auth)],
        response_model=ChunksPaginatedResponse,
        summary="Get paginated chunk previews",
        description="Returns paginated chunk records for the resolved workspace.",
    )
    async def list_chunks_paginated(
        request: Request, payload: ChunksPaginatedRequest
    ):
        workspace = resolve_workspace_or_raise(request, payload.workspace)

        try:
            chunks, total_count = await get_chunks_paginated(
                workspace_rags[workspace],
                page=payload.page,
                page_size=payload.page_size,
                sort_direction=payload.sort_direction,
            )
        except Exception as exc:
            logger.error(
                f"Failed to fetch paginated chunks in workspace '{workspace}': {exc}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch paginated chunks: {exc}",
            ) from exc

        total_pages = (total_count + payload.page_size - 1) // payload.page_size
        return ChunksPaginatedResponse(
            chunks=chunks,
            pagination=ChunksPaginationInfo(
                page=payload.page,
                page_size=payload.page_size,
                total_count=total_count,
                total_pages=total_pages,
                has_next=payload.page < total_pages,
                has_prev=payload.page > 1,
            ),
        )

    @app.post(
        "/api/chunk_content",
        dependencies=[Depends(combined_auth)],
        response_model=ChunkContentResponse,
        summary="Get full chunk content by chunk_id",
        description="Returns the full stored chunk text for the given chunk_id in the resolved workspace.",
    )
    async def get_chunk_content(request: Request, payload: ChunkContentRequest):
        workspace = resolve_workspace_or_raise(request, payload.workspace)

        chunk_id = payload.chunk_id.strip()
        try:
            chunk_data = await workspace_rags[workspace].text_chunks.get_by_id(chunk_id)
        except Exception as exc:
            logger.error(
                f"Failed to fetch chunk '{chunk_id}' in workspace '{workspace}': {exc}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch chunk content: {exc}",
            ) from exc

        if not chunk_data or "content" not in chunk_data:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk '{chunk_id}' not found in workspace '{display_workspace_id(workspace)}'",
            )

        resolved_image_base64, resolved_image_text = get_chunk_image_fields(chunk_data)
        chunk_content = chunk_data.get("content", "")
        if str(chunk_data.get("content_type", "") or "").strip().lower() == "image":
            chunk_content = resolved_image_base64 or chunk_content
        file_id = await _resolve_chunk_file_id(workspace_rags[workspace], chunk_data)
        return ChunkContentResponse(
            chunk_id=chunk_id,
            content=str(chunk_content or ""),
            content_type=str(chunk_data.get("content_type", "") or "").strip() or None,
            full_doc_id=(
                str(chunk_data.get("full_doc_id"))
                if chunk_data.get("full_doc_id") is not None
                else None
            ),
            file_id=file_id,
            file_path=(
                str(chunk_data.get("file_path"))
                if chunk_data.get("file_path") is not None
                else None
            ),
            page_id=(
                int(chunk_data.get("page_id"))
                if isinstance(chunk_data.get("page_id"), (int, float))
                else None
            ),
            bbox=chunk_data.get("bbox") if isinstance(chunk_data.get("bbox"), list) else None,
            page_size=(
                chunk_data.get("page_size")
                if isinstance(chunk_data.get("page_size"), list)
                else None
            ),
            ocr_chunk_id=(
                str(chunk_data.get("ocr_chunk_id"))
                if chunk_data.get("ocr_chunk_id") is not None
                else None
            ),
            image_base64=resolved_image_base64,
            image_text=resolved_image_text,
        )

    @app.post(
        "/api/translate_chunk",
        dependencies=[Depends(combined_auth)],
        response_model=TranslateChunkResponse,
        summary="Translate a stored chunk into Chinese",
        description="Returns translated_cn for the given chunk. Reuses cached translation when available; otherwise generates and persists it.",
    )
    async def translate_chunk(request: Request, payload: TranslateChunkRequest):
        workspace = resolve_workspace_or_raise(request, payload.workspace)
        chunk_id = payload.chunk_id.strip()

        try:
            translated_cn, cached = await translate_chunk_to_cn(
                workspace_rags[workspace],
                chunk_id,
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk '{chunk_id}' not found in workspace '{display_workspace_id(workspace)}'",
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(
                "Failed to translate chunk '%s' in workspace '%s': %s",
                chunk_id,
                workspace,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to translate chunk: {exc}",
            ) from exc

        return TranslateChunkResponse(
            chunk_id=chunk_id,
            translated_cn=translated_cn,
            cached=cached,
        )

    @app.post("/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        username = form_data.username
        if auth_handler.accounts.get(username) != form_data.password:
            raise HTTPException(status_code=401, detail="Incorrect credentials")

        # Regular user login
        user_token = auth_handler.create_token(
            username=username, role="user", metadata={"auth_mode": "enabled"}
        )
        return {
            "access_token": user_token,
            "token_type": "bearer",
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.get(
        "/health",
        dependencies=[Depends(combined_auth)],
        summary="Get system health and configuration status",
        description="Returns comprehensive system status including WebUI availability, configuration, and operational metrics",
        response_description="System health status with configuration details",
        responses={
            200: {
                "description": "Successful response with system status",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "healthy",
                            "webui_available": True,
                            "working_directory": "/path/to/working/dir",
                            "input_directory": "/path/to/input/dir",
                            "configuration": {
                                "llm_binding": "openai",
                                "llm_model": "gpt-4",
                                "embedding_binding": "openai",
                                "embedding_model": "text-embedding-ada-002",
                                "workspace": "default",
                            },
                            "auth_mode": "enabled",
                            "pipeline_busy": False,
                            "core_version": "0.0.1",
                            "api_version": "0.0.1",
                        }
                    }
                },
            }
        },
    )
    async def get_status(request: Request):
        """Get current system status including WebUI availability"""
        try:
            workspace = get_workspace_from_request(request)
            default_workspace = get_default_workspace()
            if workspace is None:
                workspace = default_workspace
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )

            if not auth_configured:
                auth_mode = "disabled"
            else:
                auth_mode = "enabled"

            # Cleanup expired keyed locks and get status
            keyed_lock_info = cleanup_keyed_lock()

            return {
                "status": "healthy",
                "webui_available": webui_dev_enabled or webui_assets_exist,
                "working_directory": str(args.working_dir),
                "input_directory": str(args.input_dir),
                "configuration": {
                    # LLM configuration binding/host address (if applicable)/model (if applicable)
                    "llm_binding": args.llm_binding,
                    "llm_binding_host": args.llm_binding_host,
                    "llm_model": args.llm_model,
                    # embedding model configuration binding/host address (if applicable)/model (if applicable)
                    "embedding_binding": args.embedding_binding,
                    "embedding_binding_host": args.embedding_binding_host,
                    "embedding_model": args.embedding_model,
                    "summary_max_tokens": args.summary_max_tokens,
                    "summary_context_size": args.summary_context_size,
                    "kv_storage": args.kv_storage,
                    "doc_status_storage": args.doc_status_storage,
                    "graph_storage": args.graph_storage,
                    "vector_storage": args.vector_storage,
                    "enable_llm_cache_for_extract": args.enable_llm_cache_for_extract,
                    "enable_llm_cache": args.enable_llm_cache,
                    "workspace": default_workspace,
                    "max_graph_nodes": args.max_graph_nodes,
                    # Rerank configuration
                    "enable_rerank": rerank_model_func is not None,
                    "rerank_binding": args.rerank_binding,
                    "rerank_model": args.rerank_model if rerank_model_func else None,
                    "rerank_binding_host": args.rerank_binding_host
                    if rerank_model_func
                    else None,
                    # Environment variable status (requested configuration)
                    "summary_language": args.summary_language,
                    "force_llm_summary_on_merge": args.force_llm_summary_on_merge,
                    "max_parallel_insert": args.max_parallel_insert,
                    "cosine_threshold": args.cosine_threshold,
                    "min_rerank_score": args.min_rerank_score,
                    "related_chunk_number": args.related_chunk_number,
                    "max_async": args.max_async,
                    "embedding_func_max_async": args.embedding_func_max_async,
                    "embedding_batch_num": args.embedding_batch_num,
                },
                "auth_mode": auth_mode,
                "pipeline_busy": pipeline_status.get("busy", False),
                "keyed_locks": keyed_lock_info,
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # Custom StaticFiles class for smart caching
    class SmartStaticFiles(StaticFiles):  # Renamed from NoCacheStaticFiles
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)

            is_html = path.endswith(".html") or response.media_type == "text/html"

            if is_html:
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            elif (
                "/assets/" in path
            ):  # Assets (JS, CSS, images, fonts) generated by Vite with hash in filename
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            # Add other rules here if needed for non-HTML, non-asset files

            # Ensure correct Content-Type
            if path.endswith(".js"):
                response.headers["Content-Type"] = "application/javascript"
            elif path.endswith(".css"):
                response.headers["Content-Type"] = "text/css"

            return response

    # Mount Swagger UI static files for offline support
    swagger_static_dir = Path(__file__).parent / "static" / "swagger-ui"
    if swagger_static_dir.exists():
        app.mount(
            "/static/swagger-ui",
            StaticFiles(directory=swagger_static_dir),
            name="swagger-ui-static",
        )

    if webui_dev_enabled:
        logger.info(
            f"WEBUI_DEV is enabled. `/webui` will redirect to {webui_dev_public_url}"
        )

        @app.get("/webui")
        @app.get("/webui/")
        async def webui_redirect_to_dev():
            """Redirect /webui to the external Vite dev server."""
            return RedirectResponse(url=webui_dev_public_url)

    # Conditionally mount WebUI only if assets exist
    elif webui_assets_exist:
        static_dir = Path(__file__).parent / "webui"
        static_dir.mkdir(exist_ok=True)
        app.mount(
            "/webui",
            SmartStaticFiles(
                directory=static_dir, html=True, check_dir=True
            ),  # Use SmartStaticFiles
            name="webui",
        )
        logger.info("WebUI assets mounted at /webui")
    else:
        logger.info("WebUI assets not available, /webui route not mounted")

        # Add redirect for /webui when assets are not available
        @app.get("/webui")
        @app.get("/webui/")
        async def webui_redirect_to_docs():
            """Redirect /webui to /docs when WebUI is not available"""
            return RedirectResponse(url="/docs")

    return app


def get_application(args=None):
    """Factory function for creating the FastAPI application"""
    if args is None:
        args = global_args
    return create_app(args)


def configure_logging():
    """Configure logging for uvicorn startup"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.filters = []

    # Get log directory path from environment variable
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

    print(f"\nLightRAG log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
    log_backup_count = get_env_value("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                # Configure all uvicorn related loggers
                "uvicorn": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
                "uvicorn.error": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
            },
            "filters": {
                "path_filter": {
                    "()": "lightrag.utils.LightragPathFilter",
                },
            },
        }
    )


def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "uvicorn",
        "tiktoken",
        "fastapi",
        # Add other required packages here
    ]

    for package in required_packages:
        if not pm.is_installed(package):
            print(f"Installing {package}...")
            pm.install(package)
            print(f"{package} installed successfully")


def main():
    # Explicitly initialize configuration for clarity
    # (The proxy will auto-initialize anyway, but this makes intent clear)
    from .config import initialize_config

    initialize_config()

    # Check if running under Gunicorn
    if "GUNICORN_CMD_ARGS" in os.environ:
        # If started with Gunicorn, return directly as Gunicorn will call get_application
        print("Running under Gunicorn - worker management handled by Gunicorn")
        return

    # Check .env file
    if not check_env_file():
        sys.exit(1)

    # Check and install dependencies
    check_and_install_dependencies()

    from multiprocessing import freeze_support

    freeze_support()

    # Configure logging before parsing args
    configure_logging()
    update_uvicorn_mode_config()
    display_splash_screen(global_args)

    # Note: Signal handlers are NOT registered here because:
    # - Uvicorn has built-in signal handling that properly calls lifespan shutdown
    # - Custom signal handlers can interfere with uvicorn's graceful shutdown
    # - Cleanup is handled by the lifespan context manager's finally block

    # Create application instance directly instead of using factory function
    app = create_app(global_args)

    # Start Uvicorn in single process mode
    uvicorn_config = {
        "app": app,  # Pass application instance directly instead of string path
        "host": global_args.host,
        "port": global_args.port,
        "log_config": None,  # Disable default config
    }

    if global_args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": global_args.ssl_certfile,
                "ssl_keyfile": global_args.ssl_keyfile,
            }
        )

    print(
        f"Starting Uvicorn server in single-process mode on {global_args.host}:{global_args.port}"
    )
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()

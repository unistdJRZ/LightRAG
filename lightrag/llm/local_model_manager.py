from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch

from lightrag.utils import logger


_DEFAULT_LOCAL_LLM_SLEEP_SECONDS = 180.0
_SLEEP_ENV_NAME = "LOCAL_LLM_SLEEP_TIME"


def _parse_local_llm_sleep_seconds(raw_value: str | None) -> float | None:
    if raw_value is None or str(raw_value).strip() == "":
        return _DEFAULT_LOCAL_LLM_SLEEP_SECONDS

    normalized = str(raw_value).strip().lower()
    if normalized in {"0", "off", "false", "none", "null", "disable", "disabled"}:
        return None

    try:
        parsed_value = float(normalized)
    except ValueError:
        logger.warning(
            "Invalid %s=%r, expected a number of seconds, falling back to default %.0fs",
            _SLEEP_ENV_NAME,
            raw_value,
            _DEFAULT_LOCAL_LLM_SLEEP_SECONDS,
        )
        return _DEFAULT_LOCAL_LLM_SLEEP_SECONDS

    if parsed_value <= 0:
        return None
    return parsed_value


LOCAL_LLM_SLEEP_SECONDS = _parse_local_llm_sleep_seconds(
    os.getenv(_SLEEP_ENV_NAME)
)
if LOCAL_LLM_SLEEP_SECONDS is None:
    logger.info("Local model auto sleep disabled via %s", _SLEEP_ENV_NAME)
else:
    logger.info(
        "Local model auto sleep enabled: %s=%.0fs",
        _SLEEP_ENV_NAME,
        LOCAL_LLM_SLEEP_SECONDS,
    )


def _iter_nested_values(resource: Any) -> Iterator[Any]:
    stack = [resource]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        object_id = id(current)
        if object_id in seen:
            continue
        seen.add(object_id)
        yield current

        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            stack.extend(current)


def _collect_cuda_memory_snapshot() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None

    device_index = torch.cuda.current_device()
    gib = 1024**3
    return {
        "allocated_gib": round(torch.cuda.memory_allocated(device_index) / gib, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(device_index) / gib, 3),
        "max_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / gib, 3
        ),
        "max_reserved_gib": round(
            torch.cuda.max_memory_reserved(device_index) / gib, 3
        ),
    }


def _remove_accelerate_hooks(module: torch.nn.Module, label: str) -> None:
    try:
        from accelerate.hooks import remove_hook_from_module
    except Exception:
        return

    try:
        if hasattr(module, "_hf_hook") or hasattr(module, "hf_device_map"):
            remove_hook_from_module(module, recurse=True)
    except Exception as exc:
        logger.debug(
            "Failed to remove accelerate hooks for %s: %s",
            label,
            exc,
        )


def _evict_module_storage(module: torch.nn.Module, label: str) -> None:
    _remove_accelerate_hooks(module, label)

    try:
        module.to_empty(device="meta", recurse=True)
        return
    except Exception as exc:
        logger.debug("Failed to move %s to meta device: %s", label, exc)

    try:
        module.to("cpu")
    except Exception as exc:
        logger.debug("Failed to move %s to CPU before unload: %s", label, exc)


def dispose_torch_resource(resource: Any, label: str | None = None) -> None:
    label_text = label or "local_model"
    before_snapshot = _collect_cuda_memory_snapshot()
    modules: list[torch.nn.Module] = []
    for current in _iter_nested_values(resource):
        if isinstance(current, torch.nn.Module):
            modules.append(current)

    for module in modules:
        _evict_module_storage(module, label_text)

    try:
        from accelerate.utils import release_memory
    except Exception:
        release_memory = None

    if release_memory is not None:
        try:
            release_memory(resource, *modules)
        except Exception as exc:
            logger.debug("Accelerate release_memory failed for %s: %s", label_text, exc)

    modules.clear()
    del modules
    del resource
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception as exc:
            logger.debug("Failed to clear CUDA cache for %s: %s", label_text, exc)

    after_snapshot = _collect_cuda_memory_snapshot()
    if before_snapshot is not None and after_snapshot is not None:
        logger.info(
            "Local model '%s' CUDA memory: allocated %.3f -> %.3f GiB, reserved %.3f -> %.3f GiB",
            label_text,
            before_snapshot["allocated_gib"],
            after_snapshot["allocated_gib"],
            before_snapshot["reserved_gib"],
            after_snapshot["reserved_gib"],
        )


@dataclass
class _ManagedResource:
    value: Any
    disposer: Callable[[Any, str | None], None] | None
    label: str
    active_count: int = 0
    last_used_monotonic: float = field(default_factory=time.monotonic)


class LocalModelManager:
    def __init__(self, idle_timeout_seconds: float | None) -> None:
        self._idle_timeout_seconds = idle_timeout_seconds
        self._entries: dict[tuple[Any, ...], _ManagedResource] = {}
        self._lock = threading.RLock()
        self._wakeup = threading.Event()
        self._shutdown = False
        self._janitor_thread: threading.Thread | None = None

    def _ensure_janitor_started(self) -> None:
        if self._idle_timeout_seconds is None or self._janitor_thread is not None:
            return

        self._janitor_thread = threading.Thread(
            target=self._janitor_loop,
            name="lightrag-local-model-janitor",
            daemon=True,
        )
        self._janitor_thread.start()

    def _janitor_loop(self) -> None:
        while True:
            expired_entries: list[tuple[tuple[Any, ...], _ManagedResource]] = []
            next_wait: float | None = None

            with self._lock:
                if self._shutdown:
                    return

                if self._idle_timeout_seconds is not None:
                    now = time.monotonic()
                    for key, entry in list(self._entries.items()):
                        if entry.active_count > 0:
                            continue

                        idle_for = now - entry.last_used_monotonic
                        remaining = self._idle_timeout_seconds - idle_for
                        if remaining <= 0:
                            expired_entries.append((key, self._entries.pop(key)))
                        else:
                            next_wait = (
                                remaining
                                if next_wait is None
                                else min(next_wait, remaining)
                            )

                self._wakeup.clear()

            for _, entry in expired_entries:
                self._dispose_entry(entry, reason="idle_timeout")

            if expired_entries:
                continue

            self._wakeup.wait(timeout=next_wait)

    def _dispose_entry(self, entry: _ManagedResource, reason: str) -> None:
        logger.info(
            "Unloading local model '%s' after %s",
            entry.label,
            reason,
        )
        if entry.disposer is not None:
            try:
                entry.disposer(entry.value, entry.label)
            except Exception as exc:
                logger.warning(
                    "Failed to unload local model '%s': %s",
                    entry.label,
                    exc,
                )

    def _acquire(
        self,
        *,
        key: tuple[Any, ...],
        loader: Callable[[], Any],
        disposer: Callable[[Any, str | None], None] | None,
        label: str,
    ) -> Any:
        with self._lock:
            self._ensure_janitor_started()
            entry = self._entries.get(key)
            if entry is None:
                logger.info("Loading local model '%s'", label)
                entry = _ManagedResource(
                    value=loader(),
                    disposer=disposer,
                    label=label,
                )
                self._entries[key] = entry

            entry.active_count += 1
            entry.last_used_monotonic = time.monotonic()
            self._wakeup.set()
            return entry.value

    def _release(self, key: tuple[Any, ...]) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return

            if entry.active_count > 0:
                entry.active_count -= 1
            entry.last_used_monotonic = time.monotonic()
            self._wakeup.set()

    @contextmanager
    def lease(
        self,
        *,
        key: tuple[Any, ...],
        loader: Callable[[], Any],
        disposer: Callable[[Any, str | None], None] | None = dispose_torch_resource,
        label: str,
    ) -> Iterator[Any]:
        resource = self._acquire(
            key=key,
            loader=loader,
            disposer=disposer,
            label=label,
        )
        try:
            yield resource
        finally:
            self._release(key)

    def unload_all(self, reason: str = "shutdown") -> None:
        with self._lock:
            items = list(self._entries.items())
            self._entries.clear()
            self._wakeup.set()

        for _, entry in items:
            self._dispose_entry(entry, reason=reason)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._wakeup.set()

        janitor_thread = self._janitor_thread
        if janitor_thread is not None and janitor_thread.is_alive():
            janitor_thread.join(timeout=1.0)

        self.unload_all(reason="manager_shutdown")


local_model_manager = LocalModelManager(LOCAL_LLM_SLEEP_SECONDS)


def shutdown_local_model_manager() -> None:
    local_model_manager.shutdown()

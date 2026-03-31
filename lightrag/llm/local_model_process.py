from __future__ import annotations

import pickle
import struct
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, BinaryIO

from lightrag.llm.local_model_manager import LOCAL_LLM_SLEEP_SECONDS
from lightrag.utils import logger


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Unexpected EOF while reading local model worker response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(stream: BinaryIO, payload: Any) -> None:
    serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">Q", len(serialized)))
    stream.write(serialized)
    stream.flush()


def _recv_message(stream: BinaryIO) -> Any:
    header = stream.read(8)
    if not header:
        raise EOFError("Local model worker stream closed")
    if len(header) != 8:
        raise EOFError("Incomplete local model worker message header")
    payload_size = struct.unpack(">Q", header)[0]
    return pickle.loads(_read_exact(stream, payload_size))


def _worker_stdio_main() -> int:
    from lightrag.llm.local_model_manager import shutdown_local_model_manager

    stdin_buffer = sys.stdin.buffer
    stdout_buffer = sys.stdout.buffer

    try:
        while True:
            try:
                request = _recv_message(stdin_buffer)
            except EOFError:
                break

            if not request:
                continue

            operation = request.get("operation")
            request_id = request.get("id")
            payload = request.get("payload") or {}

            if operation == "shutdown":
                break

            try:
                if operation == "qwen_embed":
                    from lightrag.llm.transforms_qwen import _run_embedding_sync

                    result = _run_embedding_sync(**payload)
                elif operation == "qwen_rerank":
                    from lightrag.llm.transforms_qwen import _run_rerank_sync

                    result = _run_rerank_sync(**payload)
                else:
                    raise ValueError(f"Unsupported local model operation: {operation}")

                _send_message(
                    stdout_buffer,
                    {
                        "id": request_id,
                        "ok": True,
                        "result": result,
                    },
                )
            except Exception as exc:
                _send_message(
                    stdout_buffer,
                    {
                        "id": request_id,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
    finally:
        try:
            shutdown_local_model_manager()
        except Exception as exc:
            logger.warning("Failed to shutdown local model manager in worker: %s", exc)

    return 0


class LocalModelProcessManager:
    def __init__(self, idle_timeout_seconds: float | None) -> None:
        self._idle_timeout_seconds = idle_timeout_seconds
        self._lock = threading.RLock()
        self._wakeup = threading.Event()
        self._shutdown = False
        self._janitor_thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._request_counter = 0
        self._last_used_monotonic = 0.0

    def _ensure_janitor_started(self) -> None:
        if self._idle_timeout_seconds is None or self._janitor_thread is not None:
            return

        self._janitor_thread = threading.Thread(
            target=self._janitor_loop,
            name="lightrag-local-model-process-janitor",
            daemon=True,
        )
        self._janitor_thread.start()

    def _spawn_process(self) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-m", "lightrag.llm.local_model_process", "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        self._last_used_monotonic = time.monotonic()
        logger.info(
            "Started local model worker subprocess pid=%s",
            self._process.pid,
        )

    def _ensure_process(self) -> None:
        self._ensure_janitor_started()
        if self._process is not None and self._process.poll() is None:
            return
        self._dispose_process_locked(reason="restart")
        self._spawn_process()

    def _dispose_process_locked(self, reason: str) -> None:
        process = self._process
        if process is None:
            return

        logger.info(
            "Stopping local model worker subprocess pid=%s after %s",
            process.pid,
            reason,
        )

        try:
            if process.poll() is None and process.stdin is not None:
                _send_message(process.stdin, {"operation": "shutdown"})
        except Exception:
            pass

        try:
            process.wait(timeout=5.0)
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except Exception:
                    process.kill()
                    process.wait(timeout=2.0)

        for stream in (process.stdin, process.stdout):
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass

        self._process = None

    def _janitor_loop(self) -> None:
        while True:
            timeout = None
            with self._lock:
                if self._shutdown:
                    return

                if self._process is not None and self._idle_timeout_seconds is not None:
                    idle_for = time.monotonic() - self._last_used_monotonic
                    remaining = self._idle_timeout_seconds - idle_for
                    if remaining <= 0:
                        self._dispose_process_locked(reason="idle_timeout")
                    else:
                        timeout = remaining
                self._wakeup.clear()

            self._wakeup.wait(timeout=timeout)

    def call(
        self,
        *,
        operation: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        with self._lock:
            self._ensure_process()
            process = self._process
            if process is None or process.stdin is None or process.stdout is None:
                raise RuntimeError("Local model worker subprocess is not available")

            self._request_counter += 1
            request_id = self._request_counter
            _send_message(
                process.stdin,
                {
                    "id": request_id,
                    "operation": operation,
                    "payload": payload,
                },
            )
            self._last_used_monotonic = time.monotonic()
            self._wakeup.set()

            response = _recv_message(process.stdout)
            self._last_used_monotonic = time.monotonic()
            self._wakeup.set()

            if response.get("id") != request_id:
                raise RuntimeError(
                    f"Unexpected local model worker response id={response.get('id')} expected={request_id}"
                )

            if response.get("ok"):
                return response.get("result")

            traceback_text = response.get("traceback")
            if traceback_text:
                logger.error(
                    "Local model worker error during %s:\n%s",
                    operation,
                    traceback_text,
                )
            raise RuntimeError(
                f"{response.get('error_type', 'LocalModelWorkerError')}: {response.get('error', 'Unknown local model error')}"
            )

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._wakeup.set()
            self._dispose_process_locked(reason="shutdown")

        janitor_thread = self._janitor_thread
        if janitor_thread is not None and janitor_thread.is_alive():
            janitor_thread.join(timeout=1.0)


local_model_process_manager = LocalModelProcessManager(LOCAL_LLM_SLEEP_SECONDS)


def call_qwen_embedding_in_worker(payload: dict[str, Any]) -> Any:
    return local_model_process_manager.call(
        operation="qwen_embed",
        payload=payload,
    )


def call_qwen_rerank_in_worker(payload: dict[str, Any]) -> Any:
    return local_model_process_manager.call(
        operation="qwen_rerank",
        payload=payload,
    )


def shutdown_local_model_process_manager() -> None:
    local_model_process_manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(_worker_stdio_main())

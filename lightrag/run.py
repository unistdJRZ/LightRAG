"""
Development launcher for LightRAG backend + WebUI.

Reads workspace and dev server settings from workspace.yaml and starts:
1) backend: python -m lightrag.api.lightrag_server
2) frontend: bun run dev
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from lightrag.workspace_config import (
    WorkspaceDefinition,
    load_workspace_config,
    parse_workspace_csv,
)


def _terminate_process(name: str, proc: subprocess.Popen[Any] | None):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception as exc:  # pragma: no cover - best effort cleanup
        print(f"[run.py] Failed to stop {name}: {exc}", file=sys.stderr)


def main():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run LightRAG backend + frontend in dev mode."
    )
    parser.add_argument(
        "--config",
        default=str(project_root / "workspace.yaml"),
        help="Path to workspace yaml config file",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace from config (comma-separated for multi-workspace)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run backend with DEBUG logging and verbose request diagnostics.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_workspace_config(config_path)

    configured_workspace_definitions = cfg.get("workspaces", [])
    workspace = (
        args.workspace
        if args.workspace is not None
        else ",".join(item.id for item in configured_workspace_definitions)
    )
    workspace_definitions = parse_workspace_csv(workspace)
    configured_aliases = {
        item.id: item.alias for item in configured_workspace_definitions
    }
    workspace_definitions = [
        WorkspaceDefinition(
            id=item.id,
            alias=configured_aliases.get(item.id, item.alias),
        )
        for item in workspace_definitions
    ]
    backend_host = str(cfg["backend_host"])
    backend_port = int(cfg["backend_port"])
    frontend_host = str(cfg["frontend_host"])
    frontend_port = int(cfg["frontend_port"])

    bun_bin = shutil.which("bun")
    if not bun_bin:
        raise RuntimeError("`bun` not found in PATH, cannot start frontend dev server.")

    backend_cmd = [
        sys.executable,
        "-m",
        "lightrag.api.lightrag_server",
        "--host",
        backend_host,
        "--port",
        str(backend_port),
        "--workspace",
        workspace,
        "--no-webui-dev",
    ]
    if args.debug:
        backend_cmd.extend(["--log-level", "DEBUG", "--verbose"])
    frontend_cmd = [
        bun_bin,
        "run",
        "dev",
        "--",
        "--host",
        frontend_host,
        "--port",
        str(frontend_port),
    ]

    backend_browser_host = "localhost" if backend_host in ("0.0.0.0", "::") else backend_host
    backend_url = f"http://{backend_browser_host}:{backend_port}"
    frontend_url = f"http://{frontend_host}:{frontend_port}/webui/"

    frontend_env = os.environ.copy()
    frontend_env["VITE_BACKEND_URL"] = backend_url
    frontend_env.setdefault("VITE_API_PROXY", "true")
    backend_env = os.environ.copy()
    backend_env["LIGHTRAG_WORKSPACE_CONFIG"] = str(config_path)

    print(f"[run.py] Config file: {config_path}")
    print(f"[run.py] Workspace: {workspace!r}")
    if workspace_definitions:
        workspace_summary = ", ".join(
            f"{item.id} ({item.alias})" for item in workspace_definitions
        )
        print(f"[run.py] Workspace aliases: {workspace_summary}")
    print(f"[run.py] Backend command: {' '.join(backend_cmd)}")
    print(f"[run.py] Frontend command: {' '.join(frontend_cmd)}")
    print(f"[run.py] Backend URL: {backend_url}")
    print(f"[run.py] Frontend URL: {frontend_url}")

    backend_proc: subprocess.Popen[Any] | None = None
    frontend_proc: subprocess.Popen[Any] | None = None

    try:
        backend_proc = subprocess.Popen(
            backend_cmd, cwd=str(project_root), env=backend_env
        )
        time.sleep(1.0)
        frontend_proc = subprocess.Popen(
            frontend_cmd, cwd=str(project_root / "lightrag_webui"), env=frontend_env
        )

        def _handle_signal(signum, _frame):
            print(f"[run.py] Received signal {signum}, shutting down...")
            _terminate_process("frontend", frontend_proc)
            _terminate_process("backend", backend_proc)
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handle_signal)

        while True:
            backend_rc = backend_proc.poll()
            frontend_rc = frontend_proc.poll()
            if backend_rc is not None:
                print(
                    f"[run.py] Backend exited with code {backend_rc}, stopping frontend..."
                )
                _terminate_process("frontend", frontend_proc)
                sys.exit(backend_rc)
            if frontend_rc is not None:
                print(
                    f"[run.py] Frontend exited with code {frontend_rc}, stopping backend..."
                )
                _terminate_process("backend", backend_proc)
                sys.exit(frontend_rc)
            time.sleep(0.5)

    finally:
        _terminate_process("frontend", frontend_proc)
        _terminate_process("backend", backend_proc)


if __name__ == "__main__":
    main()

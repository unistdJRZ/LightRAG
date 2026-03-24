from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_WORKSPACE_DISPLAY_ID = "rag_storage"


@dataclass(frozen=True)
class WorkspaceDefinition:
    id: str
    alias: str


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def normalize_workspace_id(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("/")


def display_workspace_id(workspace_id: str) -> str:
    normalized = normalize_workspace_id(workspace_id)
    return normalized or DEFAULT_WORKSPACE_DISPLAY_ID


def normalize_workspace_alias(alias: str | None, workspace_id: str) -> str:
    normalized_alias = str(alias or "").strip()
    return normalized_alias or display_workspace_id(workspace_id)


def parse_workspace_csv(raw_workspace: str | None) -> list[WorkspaceDefinition]:
    if raw_workspace is None:
        return [WorkspaceDefinition(id="", alias=display_workspace_id(""))]

    items = [normalize_workspace_id(part) for part in str(raw_workspace).split(",")]
    items = [item for item in items if item]
    if not items:
        return [WorkspaceDefinition(id="", alias=display_workspace_id(""))]

    seen: set[str] = set()
    result: list[WorkspaceDefinition] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(WorkspaceDefinition(id=item, alias=display_workspace_id(item)))
    return result


def _parse_workspace_items(lines: list[str], start_index: int) -> tuple[list[dict[str, str]], int]:
    items: list[dict[str, str]] = []
    current_item: dict[str, str] | None = None
    index = start_index

    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            index += 1
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0 and not stripped.startswith("-"):
            break

        if stripped.startswith("-"):
            if current_item:
                items.append(current_item)
            item_body = stripped[1:].strip()
            current_item = {}
            if item_body:
                if ":" in item_body:
                    key, value = item_body.split(":", 1)
                    current_item[key.strip()] = _strip_quotes(value.strip())
                else:
                    current_item["id"] = _strip_quotes(item_body)
            index += 1
            continue

        if current_item is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_item[key.strip()] = _strip_quotes(value.strip())

        index += 1

    if current_item:
        items.append(current_item)

    return items, index


def load_workspace_config(config_path: Path) -> dict[str, Any]:
    """
    Load workspace/dev config from workspace.yaml.

    Supported formats:
    - Legacy:
        workspace: "default,team_a"
    - Structured:
        workspaces:
          - id: default
            alias: Default Workspace
          - id: team_a
            alias: Team A
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg: dict[str, Any] = {
        "workspace": "",
        "workspaces": [],
        "backend_host": "0.0.0.0",
        "backend_port": 9621,
        "frontend_host": "127.0.0.1",
        "frontend_port": 5173,
    }

    lines = config_path.read_text(encoding="utf-8-sig").splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped == "workspaces:":
            workspace_items, index = _parse_workspace_items(lines, index + 1)
            cfg["workspaces"] = workspace_items
            continue

        if ":" not in stripped:
            index += 1
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())

        if key not in cfg or value == "":
            index += 1
            continue

        if key.endswith("_port"):
            cfg[key] = int(value)
        else:
            cfg[key] = value

        index += 1

    cfg["workspaces"] = parse_workspace_definitions(
        cfg.get("workspace"),
        raw_workspaces=cfg.get("workspaces", []),
    )
    return cfg


def parse_workspace_definitions(
    raw_workspace: str | None,
    raw_workspaces: list[dict[str, str]] | None = None,
) -> list[WorkspaceDefinition]:
    if raw_workspaces:
        result: list[WorkspaceDefinition] = []
        index_by_id: dict[str, int] = {}

        for item in raw_workspaces:
            workspace_id = normalize_workspace_id(
                item.get("id") or item.get("workspace") or item.get("name")
            )
            if not workspace_id:
                continue
            alias = normalize_workspace_alias(
                item.get("alias") or item.get("display_name"),
                workspace_id,
            )
            definition = WorkspaceDefinition(id=workspace_id, alias=alias)
            if workspace_id in index_by_id:
                result[index_by_id[workspace_id]] = definition
            else:
                index_by_id[workspace_id] = len(result)
                result.append(definition)

        if result:
            return result

    return parse_workspace_csv(raw_workspace)


def build_workspace_alias_map(
    workspace_definitions: list[WorkspaceDefinition],
) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for workspace in workspace_definitions:
        alias = workspace.alias.strip()
        if alias and alias != workspace.id:
            alias_map[alias] = workspace.id
    return alias_map


from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_WORKSPACE = PROJECT_ROOT / "workspace"


def get_workspace() -> Path:
    """Resolve the configured workspace relative to the project root."""
    workspace_value: str | None = None
    if CONFIG_PATH.is_file():
        try:
            import yaml

            with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
                config: dict[str, Any] = yaml.safe_load(config_file) or {}
            workspace_value = config.get("paths", {}).get("workspace")
        except ModuleNotFoundError:
            workspace_value = None

    workspace_value = os.getenv("CODEPILOT_WORKSPACE", workspace_value)
    workspace = Path(workspace_value or DEFAULT_WORKSPACE)
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace
    return workspace.resolve()


def resolve_workspace_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a path and reject anything outside the configured workspace."""
    workspace = get_workspace()
    path_text = str(path)
    if "{{" in path_text or "}}" in path_text:
        raise ValueError(f"Unresolved template path: {path_text}")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace / candidate

    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(
            f"Path '{path}' is outside the workspace: {workspace}"
        ) from exc
    return resolved


def is_test_path(path: Path, workspace: Path | None = None) -> bool:
    """Return whether a path looks like a test file or test directory."""
    workspace = workspace or get_workspace()
    try:
        relative = path.resolve(strict=False).relative_to(workspace.resolve())
    except ValueError:
        relative = path
    parts = relative.parts
    name = relative.name
    return (
        "tests" in parts
        or "test" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def ensure_editable_source(path: Path) -> None:
    """Protect tests unless explicitly enabled for a trusted workflow."""
    if is_test_path(path) and os.getenv("CODEPILOT_ALLOW_TEST_EDITS") != "1":
        raise ValueError(
            "Test files are read-only by default; set CODEPILOT_ALLOW_TEST_EDITS=1 "
            "to allow test edits"
        )


def is_glob_pattern(value: str) -> bool:
    return any(char in value for char in "*?[]")


def matches_glob(path: Path, pattern: str, workspace: Path | None = None) -> bool:
    workspace = workspace or get_workspace()
    relative = path.relative_to(workspace).as_posix()
    return fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern)


def success_result(
    output: str = "",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "success": True,
        "output": output,
        "error": "",
        **extra,
    }


def failure_result(
    error: str,
    output: str = "",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "success": False,
        "output": output,
        "error": error,
        **extra,
    }

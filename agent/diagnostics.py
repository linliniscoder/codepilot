from __future__ import annotations

import re
from pathlib import Path


_TRACEBACK_PATH_RE = re.compile(r'File "([^"]+\.py)"')
_PATH_WITH_LINE_RE = re.compile(
    r"(?<![A-Za-z0-9_./\\-])(?P<path>(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.-]+\.py)(?::\d+)?"
)
_BARE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_./-])(?P<path>[A-Za-z_][A-Za-z0-9_.-]*\.py)(?::\d+)?")
_FROM_IMPORT_RE = re.compile(
    r"^\s*from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+",
    re.MULTILINE,
)
_IMPORT_RE = re.compile(
    r"^\s*import\s+([A-Za-z_][A-Za-z0-9_.]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_.]*)*)",
    re.MULTILINE,
)


def extract_python_paths(
    text: str,
    workspace: str | Path | None = None,
    max_paths: int = 12,
) -> list[str]:
    """Extract workspace-relative Python paths from pytest output."""
    if not text:
        return []

    workspace_path = Path(workspace).resolve() if workspace else None
    candidates: list[str] = []
    candidates.extend(_TRACEBACK_PATH_RE.findall(text))
    candidates.extend(match.group("path") for match in _PATH_WITH_LINE_RE.finditer(text))
    candidates.extend(match.group("path") for match in _BARE_PATH_RE.finditer(text))

    paths: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_path(candidate, workspace_path)
        if normalized and normalized not in seen:
            seen.add(normalized)
            paths.append(normalized)
            if len(paths) >= max_paths:
                break
    return paths


def infer_imported_python_paths(
    source: str,
    workspace: str | Path,
    max_paths: int = 12,
) -> list[str]:
    """Infer local module files imported by a test or source file."""
    workspace_path = Path(workspace).resolve()
    modules: list[str] = []
    for match in _FROM_IMPORT_RE.finditer(source):
        modules.append(match.group(1))
    for match in _IMPORT_RE.finditer(source):
        modules.extend(
            item.strip().split(" as ", 1)[0]
            for item in match.group(1).split(",")
        )

    paths: list[str] = []
    seen: set[str] = set()
    for module in modules:
        module_path = module.replace(".", "/")
        candidates = (f"{module_path}.py", f"{module_path}/__init__.py")
        for candidate in candidates:
            path = workspace_path / candidate
            if not path.is_file():
                continue
            normalized = path.relative_to(workspace_path).as_posix()
            if normalized not in seen:
                seen.add(normalized)
                paths.append(normalized)
                if len(paths) >= max_paths:
                    return paths
    return paths


def _normalize_path(candidate: str, workspace: Path | None) -> str:
    value = candidate.replace("\\", "/")
    path = Path(value)
    if workspace is None:
        return value.removeprefix("./")

    if path.is_absolute():
        try:
            return path.resolve().relative_to(workspace).as_posix()
        except ValueError:
            return ""

    try:
        return (workspace / path).resolve().relative_to(workspace).as_posix()
    except ValueError:
        return value.removeprefix("./")

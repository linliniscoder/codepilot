from __future__ import annotations

import difflib
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from ._common import (
    failure_result,
    is_glob_pattern,
    get_workspace,
    ensure_editable_source,
    matches_glob,
    resolve_workspace_path,
    success_result,
)
from .registry import ToolDefinition


LOGGER = logging.getLogger(__name__)


def read_file(path: str) -> dict[str, Any]:
    """Read a UTF-8 text file located inside the configured workspace."""
    LOGGER.info("read_file started: path=%s", path)
    try:
        file_path = resolve_workspace_path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
        content = file_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    except (OSError, UnicodeError, ValueError) as exc:
        LOGGER.exception("read_file failed: path=%s", path)
        return failure_result(str(exc))

    LOGGER.info("read_file completed: path=%s bytes=%d", path, len(content.encode()))
    return success_result(content, path=path, sha256=digest)


def write_file(path: str, content: str) -> dict[str, Any]:
    """Create a UTF-8 file inside the workspace without overwriting by default."""
    LOGGER.info("write_file started: path=%s", path)
    try:
        file_path = resolve_workspace_path(path)
        ensure_editable_source(file_path)
        if file_path.exists() and os.getenv("CODEPILOT_ALLOW_OVERWRITE") != "1":
            raise ValueError(
                "Existing files must be modified with edit_file; set "
                "CODEPILOT_ALLOW_OVERWRITE=1 only for trusted workflows"
            )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        LOGGER.exception("write_file failed: path=%s", path)
        return failure_result(str(exc))

    LOGGER.info("write_file completed: path=%s bytes=%d", path, len(content.encode()))
    return success_result(f"File written: {path}")


def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Replace one exact text region in a workspace file."""
    LOGGER.info("edit_file started: path=%s", path)
    try:
        file_path = resolve_workspace_path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
        ensure_editable_source(file_path)
        content = file_path.read_text(encoding="utf-8")
        current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if expected_sha256 and expected_sha256 != current_hash:
            raise ValueError(
                f"File changed since it was read: expected {expected_sha256}, "
                f"found {current_hash}"
            )
        occurrences = content.count(old_text)
        if occurrences != 1:
            raise ValueError(
                f"old_text must occur exactly once, found {occurrences} occurrences"
            )
        updated = content.replace(old_text, new_text, 1)
        file_path.write_text(updated, encoding="utf-8")
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        LOGGER.exception("edit_file failed: path=%s", path)
        return failure_result(str(exc))

    diff = "".join(
        difflib.unified_diff(
            content.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )
    LOGGER.info("edit_file completed: path=%s bytes=%d", path, len(updated.encode()))
    return success_result(
        f"File edited: {path}",
        path=path,
        diff=diff,
        old_sha256=current_hash,
        new_sha256=hashlib.sha256(updated.encode("utf-8")).hexdigest(),
    )


def search_files(keyword: str) -> dict[str, Any]:
    """Search text files recursively inside the configured workspace."""
    LOGGER.info("search_files started: keyword=%s", keyword)
    if not isinstance(keyword, str) or not keyword:
        return failure_result("keyword must be a non-empty string")

    workspace = get_workspace()
    matches: list[str] = []
    try:
        if not workspace.is_dir():
            raise FileNotFoundError(f"Workspace does not exist: {workspace}")

        for candidate in workspace.rglob("*"):
            if not candidate.is_file():
                continue
            try:
                file_path = resolve_workspace_path(candidate)
            except ValueError:
                LOGGER.warning("Skipping path outside workspace: path=%s", candidate)
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                LOGGER.debug("Skipping non-UTF-8 file: path=%s", file_path)
                continue
            if is_glob_pattern(keyword):
                if matches_glob(file_path, keyword, workspace=workspace):
                    matches.append(file_path.relative_to(workspace).as_posix())
                    continue
            if keyword in text or keyword in file_path.name:
                matches.append(file_path.relative_to(workspace).as_posix())
    except (OSError, ValueError) as exc:
        LOGGER.exception("search_files failed: keyword=%s", keyword)
        return failure_result(str(exc))

    output = "\n".join(sorted(matches))
    LOGGER.info("search_files completed: keyword=%s matches=%d", keyword, len(matches))
    return success_result(output)


READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read a UTF-8 text file and return its current SHA-256 hash.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    handler=read_file,
)

WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description=(
        "Create a new UTF-8 file in the workspace. Existing files must use edit_file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "content": {"type": "string", "description": "File content to write."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    handler=write_file,
)

SEARCH_FILES_TOOL = ToolDefinition(
    name="search_files",
    description="Search workspace text files for a keyword.",
    parameters={
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "Text to search for."},
        },
        "required": ["keyword"],
        "additionalProperties": False,
    },
    handler=search_files,
)

EDIT_FILE_TOOL = ToolDefinition(
    name="edit_file",
    description="Replace one exact text region in an existing workspace file.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "old_text": {
                "type": "string",
                "description": "Existing text that must occur exactly once.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text.",
            },
            "expected_sha256": {
                "type": ["string", "null"],
                "description": "Optional hash from the last read of the file.",
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
    handler=edit_file,
)

FILE_TOOLS = (READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, SEARCH_FILES_TOOL)

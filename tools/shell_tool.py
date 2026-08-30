from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from typing import Any

from ._common import failure_result, get_workspace, success_result
from .registry import ToolDefinition


LOGGER = logging.getLogger(__name__)

COMMAND_TIMEOUT = 30
ALLOWED_COMMANDS = {
    "find",
    "git",
    "ls",
    "mypy",
    "pwd",
    "pytest",
    "python",
    "pyright",
    "rg",
    "ruff",
}
FORBIDDEN_OPERATORS = (";", "&&", "||", "|", ">", "<", "`", "$()")
FORBIDDEN_GIT_COMMANDS = {"checkout", "clean", "reset", "restore"}
SAFE_GIT_COMMANDS = {
    "branch",
    "diff",
    "grep",
    "log",
    "ls-files",
    "rev-parse",
    "show",
    "status",
}


def run_command(command: str) -> dict[str, Any]:
    """Run a shell command with the configured workspace as its working directory."""
    LOGGER.info("run_command started: command=%s", command)
    workspace = get_workspace()

    if not isinstance(command, str) or not command.strip():
        return failure_result(
            "command must be a non-empty string",
            stdout="",
            stderr="",
            return_code=-1,
        )
    if not workspace.is_dir():
        error = f"Workspace does not exist: {workspace}"
        LOGGER.error("run_command failed: %s", error)
        return failure_result(error, stdout="", stderr="", return_code=-1)

    try:
        command_args = _validate_command(command)
    except ValueError as exc:
        LOGGER.warning("run_command rejected: command=%s reason=%s", command, exc)
        return failure_result(str(exc), stdout="", stderr="", return_code=-1)

    try:
        completed = subprocess.run(
            command_args,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        error = f"Command timed out after {COMMAND_TIMEOUT} seconds"
        LOGGER.exception("run_command timed out: command=%s", command)
        return failure_result(
            error,
            output=stdout,
            stdout=stdout,
            stderr=stderr,
            return_code=-1,
        )
    except OSError as exc:
        LOGGER.exception("run_command failed: command=%s", command)
        return failure_result(str(exc), stdout="", stderr="", return_code=-1)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    result = (
        success_result(
            stdout,
            stdout=stdout,
            stderr=stderr,
            return_code=completed.returncode,
        )
        if completed.returncode == 0
        else failure_result(
            stderr or f"Command exited with code {completed.returncode}",
            output=stdout,
            stdout=stdout,
            stderr=stderr,
            return_code=completed.returncode,
        )
    )
    LOGGER.info(
        "run_command completed: command=%s return_code=%d stdout_bytes=%d stderr_bytes=%d",
        command,
        completed.returncode,
        len(stdout.encode()),
        len(stderr.encode()),
    )
    return result


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _validate_command(command: str) -> list[str]:
    if any(operator in command for operator in FORBIDDEN_OPERATORS):
        raise ValueError("shell operators are not allowed")

    try:
        args = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"invalid command syntax: {exc}") from exc
    if not args:
        raise ValueError("command must contain an executable")

    executable = args[0]
    executable_name = executable.rsplit("/", 1)[-1]
    is_python = executable_name in {"python", "python3"}
    allowed_python = is_python and (
        executable in {sys.executable, "python", "python3"}
        or executable_name in ALLOWED_COMMANDS
    )
    if executable_name not in ALLOWED_COMMANDS and not allowed_python:
        raise ValueError(f"command '{executable_name}' is not allowed")

    if executable_name == "git":
        git_subcommand = _git_subcommand(args[1:])
        if git_subcommand in FORBIDDEN_GIT_COMMANDS:
            raise ValueError(f"git command '{git_subcommand}' is not allowed")
        if git_subcommand not in SAFE_GIT_COMMANDS:
            raise ValueError(f"git command '{git_subcommand}' is not allowed")

    if allowed_python:
        if len(args) >= 2 and args[1] in {"-c", "-", "-m"}:
            if args[1] == "-m" and len(args) >= 3 and args[2] in {
                "pytest",
                "py_compile",
            }:
                return args
            raise ValueError("only python -m pytest or python -m py_compile is allowed")
        if len(args) >= 2:
            raise ValueError("running arbitrary Python scripts is not allowed")

    if executable_name == "find" and any(
        item in {"-exec", "-execdir"} for item in args[1:]
    ):
        raise ValueError("find -exec is not allowed")
    return args


def _git_subcommand(args: list[str]) -> str:
    """Find a git subcommand while skipping common global option values."""
    options_with_values = {"-C", "-c", "--git-dir", "--work-tree"}
    index = 0
    while index < len(args):
        item = args[index]
        if item in options_with_values:
            index += 2
            continue
        if item.startswith("-"):
            index += 1
            continue
        return item
    return ""


RUN_COMMAND_TOOL = ToolDefinition(
    name="run_command",
    description="Run a shell command with the workspace as its working directory.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    handler=run_command,
)

SHELL_TOOLS = (RUN_COMMAND_TOOL,)

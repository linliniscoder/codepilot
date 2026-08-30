from __future__ import annotations

import hashlib
import sys

from tools.file_tool import edit_file, read_file, write_file
from tools.shell_tool import _validate_command, run_command


def test_edit_file_replaces_one_exact_region(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    path = tmp_path / "sample.py"
    path.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    result = edit_file(
        "sample.py",
        "return a - b",
        "return a + b",
        expected_sha256=digest,
    )

    assert result["success"] is True
    assert "return a + b" in path.read_text(encoding="utf-8")
    assert "-    return a - b" in result["diff"]


def test_file_tools_reject_outside_workspace_and_stale_edits(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    (tmp_path / "sample.txt").write_text("old", encoding="utf-8")

    outside = read_file("../sample.txt")
    stale = edit_file("sample.txt", "old", "new", expected_sha256="wrong")

    assert outside["success"] is False
    assert "outside the workspace" in outside["error"]
    assert stale["success"] is False
    assert "File changed since it was read" in stale["error"]


def test_file_tools_protect_tests_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text("assert True\n", encoding="utf-8")

    result = edit_file("tests/test_sample.py", "assert True", "assert False")

    assert result["success"] is False
    assert "Test files are read-only" in result["error"]


def test_write_file_does_not_overwrite_existing_source_by_default(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    path = tmp_path / "sample.py"
    path.write_text("return_value = 1\n", encoding="utf-8")

    rejected = write_file("sample.py", "return_value = 2\n")

    assert rejected["success"] is False
    assert "must be modified with edit_file" in rejected["error"]
    assert path.read_text(encoding="utf-8") == "return_value = 1\n"


def test_write_file_can_create_new_source_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))

    result = write_file("new_module.py", "VALUE = 1\n")

    assert result["success"] is True
    assert (tmp_path / "new_module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_shell_tool_allows_workspace_command_without_shell_operators(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))

    allowed = run_command(f"{sys.executable} -m pytest --version")
    rejected = run_command("echo 123; rm -rf .")
    rejected_python = run_command("python -c 'print(123)'")

    assert allowed["success"] is True
    assert "pytest" in allowed["stdout"]
    assert rejected["success"] is False
    assert "shell operators are not allowed" in rejected["error"]
    assert rejected_python["success"] is False
    assert "only python -m pytest" in rejected_python["error"]


def test_shell_tool_allows_git_c_and_rejects_mutating_subcommands(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))

    allowed = _validate_command("git -C . status --short")
    rejected = run_command("git apply change.patch")

    assert allowed[:4] == ["git", "-C", ".", "status"]
    assert rejected["success"] is False
    assert "is not allowed" in rejected["error"]

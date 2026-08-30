from __future__ import annotations

from pathlib import Path

from agent.diagnostics import extract_python_paths, infer_imported_python_paths


def test_extract_python_paths_normalizes_absolute_and_relative_paths(
    tmp_path: Path,
) -> None:
    output = (
        f'File "{tmp_path}/package/module.py", line 8, in run\n'
        "FAILED tests/test_module.py:12\n"
    )

    assert extract_python_paths(output, workspace=tmp_path) == [
        "package/module.py",
        "tests/test_module.py",
    ]


def test_infer_imported_local_modules(tmp_path: Path) -> None:
    (tmp_path / "widget.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "helper.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    source = "from widget import value\nimport package.helper\n"

    assert infer_imported_python_paths(source, workspace=tmp_path) == [
        "widget.py",
        "package/helper.py",
    ]

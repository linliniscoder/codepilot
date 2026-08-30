from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.controller import CodingAgent
from agent.state import AgentState

from .metrics import summarize_results


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEMO_TASKS_PATH = Path(__file__).with_name("demo_tasks.json")
DEFAULT_RESULTS_PATH = Path(__file__).with_name("results.json")
TEST_TIMEOUT_SECONDS = 120

AgentFactory = Callable[[], Any]


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    issue: str
    repo: str
    public_test_command: str
    hidden_test_command: str | None = None
    hidden_test_source: str | None = None
    expected_files: tuple[str, ...] = ()
    forbidden_files: tuple[str, ...] = ()

    @property
    def test_command(self) -> str:
        """Backward-compatible alias for the public validation command."""
        return self.public_test_command

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkTask":
        missing = [key for key in ("id", "issue", "repo") if key not in data]
        if missing:
            raise ValueError(f"Benchmark task missing required fields: {missing}")

        public_test_command = data.get("public_test_command", data.get("test_command"))
        if not isinstance(public_test_command, str) or not public_test_command.strip():
            raise ValueError(
                "Benchmark task requires a non-empty public_test_command or test_command"
            )

        def string_tuple(key: str) -> tuple[str, ...]:
            value = data.get(key, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise ValueError(f"Benchmark task field '{key}' must be a list of strings")
            return tuple(item.strip() for item in value)

        hidden_test_command = data.get("hidden_test_command")
        if hidden_test_command is not None and not isinstance(hidden_test_command, str):
            raise ValueError("hidden_test_command must be a string or null")
        hidden_test_source = data.get("hidden_test_source")
        if hidden_test_source is not None and not isinstance(hidden_test_source, str):
            raise ValueError("hidden_test_source must be a string or null")

        return cls(
            id=str(data["id"]),
            issue=str(data["issue"]),
            repo=str(data["repo"]),
            public_test_command=public_test_command.strip(),
            hidden_test_command=(
                hidden_test_command.strip() if hidden_test_command else None
            ),
            hidden_test_source=(hidden_test_source.strip() if hidden_test_source else None),
            expected_files=string_tuple("expected_files"),
            forbidden_files=string_tuple("forbidden_files"),
        )


class BenchmarkRunner:
    """Run CodePilot on local JSON task sets and compute aggregate metrics."""

    def __init__(
        self,
        agent_factory: AgentFactory | None = None,
        results_path: str | Path = DEFAULT_RESULTS_PATH,
        test_timeout: int = TEST_TIMEOUT_SECONDS,
    ) -> None:
        self.agent_factory = agent_factory or self._default_agent_factory
        self.results_path = Path(results_path)
        self.test_timeout = test_timeout

    def run(
        self,
        tasks_path: str | Path = DEFAULT_DEMO_TASKS_PATH,
    ) -> dict[str, Any]:
        tasks = load_tasks(tasks_path)
        LOGGER.info("Benchmark started: tasks=%d", len(tasks))

        results = [self.run_task(task) for task in tasks]
        summary = summarize_results(results)
        payload = {
            "summary": summary,
            "results": results,
        }
        save_results(payload, self.results_path)

        LOGGER.info(
            "Benchmark completed: tasks=%d task_success_rate=%.3f test_pass_rate=%.3f",
            summary["num_tasks"],
            summary["task_success_rate"],
            summary["test_pass_rate"],
        )
        return payload

    def run_task(self, task: BenchmarkTask) -> dict[str, Any]:
        LOGGER.info("Benchmark task started: id=%s repo=%s", task.id, task.repo)
        started_at = time.perf_counter()
        state: AgentState | None = None
        agent_error: str | None = None
        source_repo = resolve_repo_path(task.repo)

        with tempfile.TemporaryDirectory(prefix="codepilot-benchmark-") as temp_dir:
            repo_path = Path(temp_dir) / "repo"
            prepare_repo_copy(source_repo, repo_path)
            before_agent_snapshot = snapshot_repo_files(repo_path)

            previous_workspace = os.environ.get("CODEPILOT_WORKSPACE")
            os.environ["CODEPILOT_WORKSPACE"] = str(repo_path)
            try:
                try:
                    agent = self.agent_factory()
                    state = agent.run(task.issue)
                except Exception as exc:
                    LOGGER.exception("Agent failed during benchmark task: id=%s", task.id)
                    agent_error = str(exc)

                after_agent_snapshot = snapshot_repo_files(repo_path)

                public_test_result = run_test_command(
                    command=task.public_test_command,
                    repo_path=repo_path,
                    timeout=self.test_timeout,
                )

                hidden_test_result: dict[str, Any] | None = None
                if task.hidden_test_source:
                    hidden_source = resolve_repo_path(task.hidden_test_source)
                    hidden_target = repo_path / "hidden_tests"
                    prepare_repo_copy(hidden_source, hidden_target)
                if task.hidden_test_command:
                    hidden_test_result = run_test_command(
                        command=task.hidden_test_command,
                        repo_path=repo_path,
                        timeout=self.test_timeout,
                    )
            finally:
                if previous_workspace is None:
                    os.environ.pop("CODEPILOT_WORKSPACE", None)
                else:
                    os.environ["CODEPILOT_WORKSPACE"] = previous_workspace

        validation_results = [public_test_result]
        if hidden_test_result is not None:
            validation_results.append(hidden_test_result)
        test_passed = all(result["return_code"] == 0 for result in validation_results)
        expected_files = set(task.expected_files)
        forbidden_files = set(task.forbidden_files)
        reported_changed_files = set(_state_changed_files(state))
        changed_files = set(
            changed_files_between(before_agent_snapshot, after_agent_snapshot)
        )
        expected_files_met = expected_files.issubset(changed_files)
        forbidden_files_touched = sorted(changed_files & forbidden_files)
        latency = time.perf_counter() - started_at
        task_success = (
            agent_error is None
            and test_passed
            and not _state_last_error(state)
            and expected_files_met
            and not forbidden_files_touched
        )
        result = {
            "id": task.id,
            "issue": task.issue,
            "repo": str(source_repo),
            "source_repo": str(source_repo),
            "test_command": task.public_test_command,
            "public_test_command": task.public_test_command,
            "hidden_test_command": task.hidden_test_command,
            "task_success": task_success,
            "test_passed": test_passed,
            "expected_files": list(task.expected_files),
            "expected_files_met": expected_files_met,
            "forbidden_files": list(task.forbidden_files),
            "forbidden_files_touched": forbidden_files_touched,
            "iterations": _state_iterations(state),
            "failure_count": _state_failure_count(state),
            "first_pass": test_passed and _state_failure_count(state) == 0,
            "recovered": test_passed and _state_failure_count(state) > 0,
            "token_cost": _state_token_cost(state),
            "latency": latency,
            "agent_error": agent_error,
            "last_error": _state_last_error(state),
            "changed_files": sorted(changed_files),
            "reported_changed_files": sorted(reported_changed_files),
            "test_result": public_test_result,
            "public_test_result": public_test_result,
            "hidden_test_result": hidden_test_result,
        }
        LOGGER.info(
            "Benchmark task completed: id=%s success=%s test_passed=%s latency=%.3fs",
            task.id,
            task_success,
            test_passed,
            latency,
        )
        return result

    @staticmethod
    def _default_agent_factory() -> CodingAgent:
        return CodingAgent()


def load_tasks(tasks_path: str | Path) -> list[BenchmarkTask]:
    path = Path(tasks_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("benchmark tasks file must contain a JSON list")
    return [BenchmarkTask.from_dict(item) for item in data]


def save_results(payload: dict[str, Any], results_path: str | Path) -> None:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_test_command(
    command: str,
    repo_path: str | Path,
    timeout: int = TEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if not command.strip():
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": "test_command cannot be empty",
        }

    repo = Path(repo_path)
    if not repo.is_dir():
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": f"repo does not exist: {repo}",
        }

    started_at = time.perf_counter()
    command_env = os.environ.copy()
    python_dir = str(Path(sys.executable).resolve().parent)
    command_env["PATH"] = os.pathsep.join(
        [python_dir, command_env.get("PATH", "")]
    )
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
            check=False,
            env=command_env,
        )
    except subprocess.TimeoutExpired as exc:
        latency = time.perf_counter() - started_at
        return {
            "success": False,
            "stdout": _as_text(exc.stdout),
            "stderr": _as_text(exc.stderr),
            "return_code": -1,
            "error": f"test command timed out after {timeout} seconds",
            "latency": latency,
        }
    except OSError as exc:
        latency = time.perf_counter() - started_at
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": str(exc),
            "latency": latency,
        }

    latency = time.perf_counter() - started_at
    return {
        "success": completed.returncode == 0,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "return_code": completed.returncode,
        "error": "" if completed.returncode == 0 else f"test command failed: {completed.returncode}",
        "latency": latency,
    }


def resolve_repo_path(repo: str | Path) -> Path:
    repo_path = Path(repo)
    if repo_path.is_absolute():
        return repo_path
    return (PROJECT_ROOT / repo_path).resolve()


def prepare_repo_copy(source: Path, target: Path) -> None:
    """Copy a benchmark repository while excluding large/runtime-only files."""
    if not source.is_dir():
        raise FileNotFoundError(f"repo does not exist: {source}")

    ignored_names = {
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        "__pycache__",
        "models",
        "evaluation/results.json",
    }

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            relative = (Path(directory) / name).relative_to(source).as_posix()
            if name in ignored_names or relative in ignored_names:
                ignored.add(name)
        return ignored

    shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=True)


def snapshot_repo_files(repo_path: Path) -> dict[str, str]:
    """Hash repository files so benchmark changes cannot be hidden in state."""
    ignored_parts = {
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        "__pycache__",
        "models",
        "hidden_tests",
    }
    snapshot: dict[str, str] = {}
    for path in repo_path.rglob("*"):
        if not path.is_file() or any(part in ignored_parts for part in path.parts):
            continue
        try:
            relative = path.relative_to(repo_path).as_posix()
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            LOGGER.debug("Skipping benchmark snapshot path: %s", path)
    return snapshot


def changed_files_between(
    before: dict[str, str],
    after: dict[str, str],
) -> list[str]:
    """Return added, removed, or modified repository-relative paths."""
    return sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )


def _state_iterations(state: AgentState | None) -> int:
    return int(getattr(state, "iterations", 0) or 0)


def _state_last_error(state: AgentState | None) -> str | None:
    value = getattr(state, "last_error", None)
    return str(value) if value else None


def _state_changed_files(state: AgentState | None) -> list[str]:
    return list(getattr(state, "changed_files", []) or [])


def _state_failure_count(state: AgentState | None) -> int:
    return int(getattr(state, "failure_count", 0) or 0)


def _state_token_cost(state: AgentState | None) -> int:
    if state is None:
        return 0

    total = sum(
        _usage_total_tokens(item)
        for item in getattr(state, "llm_usage", []) or []
    )
    for item in getattr(state, "history", []) or []:
        total += _usage_total_tokens(item.get("usage"))
    for item in getattr(state, "tool_results", []) or []:
        total += _usage_total_tokens(item.get("usage"))
        result = item.get("result", {})
        if isinstance(result, dict):
            total += _usage_total_tokens(result.get("usage"))
    return total


def _usage_total_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    value = usage.get("total_tokens")
    if value is None:
        return 0
    return int(value)


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_benchmark(
    tasks_path: str | Path = DEFAULT_DEMO_TASKS_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    agent_factory: AgentFactory | None = None,
) -> dict[str, Any]:
    return BenchmarkRunner(
        agent_factory=agent_factory,
        results_path=results_path,
    ).run(tasks_path=tasks_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CodePilot benchmark tasks")
    parser.add_argument("--tasks", default=str(DEFAULT_DEMO_TASKS_PATH))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--test-timeout", type=int, default=TEST_TIMEOUT_SECONDS)
    args = parser.parse_args()
    payload = BenchmarkRunner(
        results_path=args.results,
        test_timeout=args.test_timeout,
    ).run(tasks_path=args.tasks)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

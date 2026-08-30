from __future__ import annotations

import json
import os
from pathlib import Path

from agent.state import AgentState
from evaluation.benchmark import BenchmarkRunner
from evaluation.benchmark import load_tasks
from evaluation.metrics import summarize_results


def test_summarize_results() -> None:
    summary = summarize_results(
        [
            {
                "task_success": True,
                "test_passed": True,
                "iterations": 2,
                "token_cost": 10,
                "latency": 1.0,
                "first_pass": True,
            },
            {
                "task_success": False,
                "test_passed": False,
                "iterations": 4,
                "token_cost": 30,
                "latency": 3.0,
                "first_pass": False,
                "recovered": False,
            },
        ]
    )

    assert summary["task_success_rate"] == 0.5
    assert summary["test_pass_rate"] == 0.5
    assert summary["average_iteration_count"] == 3.0
    assert summary["total_token_cost"] == 40
    assert summary["average_latency"] == 2.0
    assert summary["recovery_opportunity_count"] == 1
    assert summary["conditional_recovery_rate"] == 0.0


def test_benchmark_runner_with_fake_agent(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    results_path = tmp_path / "results.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "id": "demo",
                    "issue": "demo issue",
                    "repo": ".",
                    "test_command": "python -m py_compile main.py",
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeAgent:
        def run(self, issue: str) -> AgentState:
            workspace = Path(os.environ["CODEPILOT_WORKSPACE"])
            (workspace / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
            state = AgentState(task=issue)
            state.update(iterations=2, changed_files=["demo.py"])
            state.append_history("assistant", "planned", usage={"total_tokens": 12})
            state.add_tool_result(
                "run_tests",
                {
                    "success": True,
                    "output": "ok",
                    "error": "",
                    "usage": {"total_tokens": 5},
                },
            )
            return state

    payload = BenchmarkRunner(
        agent_factory=lambda: FakeAgent(),
        results_path=results_path,
    ).run(tasks_path)

    assert results_path.is_file()
    assert payload["summary"]["num_tasks"] == 1
    assert payload["summary"]["task_success_rate"] == 1.0
    assert payload["summary"]["total_token_cost"] == 17
    assert payload["results"][0]["changed_files"] == ["demo.py"]
    assert payload["results"][0]["reported_changed_files"] == ["demo.py"]


def test_benchmark_detects_forbidden_shell_side_effects(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "id": "tamper",
                    "issue": "demo issue",
                    "repo": "evaluation/fixtures/math_average",
                    "public_test_command": "python -m pytest tests -q",
                    "forbidden_files": ["tests/test_calculator.py"],
                }
            ]
        ),
        encoding="utf-8",
    )

    class TamperingAgent:
        def run(self, issue: str) -> AgentState:
            workspace = Path(os.environ["CODEPILOT_WORKSPACE"])
            test_path = workspace / "tests" / "test_calculator.py"
            test_path.write_text("assert True\n", encoding="utf-8")
            state = AgentState(task=issue)
            state.update(tests_run=True, tests_passed=True)
            return state

    result = BenchmarkRunner(
        agent_factory=TamperingAgent,
        results_path=tmp_path / "results.json",
    ).run_task(load_tasks(tasks_path)[0])

    assert result["changed_files"] == ["tests/test_calculator.py"]
    assert result["forbidden_files_touched"] == ["tests/test_calculator.py"]
    assert result["task_success"] is False


def test_benchmark_runs_hidden_tests_on_a_clean_copy(tmp_path: Path) -> None:
    task_path = tmp_path / "tasks.json"
    task_path.write_text(
        json.dumps(
            [
                {
                    "id": "math",
                    "issue": "修复加法",
                    "repo": "evaluation/fixtures/math_average",
                    "public_test_command": "python -m pytest tests -q",
                    "hidden_test_command": "python -m pytest hidden_tests -q",
                    "hidden_test_source": "evaluation/hidden_tests/math_average",
                    "expected_files": ["calculator.py"],
                }
            ]
        ),
        encoding="utf-8",
    )

    class FixAgent:
        def run(self, issue: str) -> AgentState:
            workspace = Path(os.environ["CODEPILOT_WORKSPACE"])
            (workspace / "calculator.py").write_text(
                "def add(a, b):\n    return a + b\n\n\n"
                "def average(values):\n    return sum(values) / len(values) if values else 0\n",
                encoding="utf-8",
            )
            state = AgentState(task=issue)
            state.update(changed_files=["calculator.py"], tests_run=True, tests_passed=True)
            return state

    result = BenchmarkRunner(
        agent_factory=FixAgent,
        results_path=tmp_path / "results.json",
    ).run_task(load_tasks(task_path)[0])

    original = Path("evaluation/fixtures/math_average/calculator.py").read_text(
        encoding="utf-8"
    )
    assert result["task_success"] is True
    assert result["public_test_result"]["return_code"] == 0
    assert result["hidden_test_result"]["return_code"] == 0
    assert result["expected_files_met"] is True
    assert "return a - b" in original

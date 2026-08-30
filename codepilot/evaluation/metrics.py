from __future__ import annotations

from statistics import mean
from typing import Any


def task_success_rate(results: list[dict[str, Any]]) -> float:
    """Ratio of tasks marked successful."""
    if not results:
        return 0.0
    return _ratio(result.get("task_success", False) for result in results)


def test_pass_rate(results: list[dict[str, Any]]) -> float:
    """Ratio of tasks whose validation tests passed."""
    if not results:
        return 0.0
    return _ratio(result.get("test_passed", False) for result in results)


def average_iteration_count(results: list[dict[str, Any]]) -> float:
    """Average Agent iteration count."""
    return _average(result.get("iterations", 0) for result in results)


def total_token_cost(results: list[dict[str, Any]]) -> int:
    """Total token usage across benchmark results."""
    return sum(int(result.get("token_cost", 0) or 0) for result in results)


def average_token_cost(results: list[dict[str, Any]]) -> float:
    """Average token usage per task."""
    return _average(result.get("token_cost", 0) for result in results)


def average_latency(results: list[dict[str, Any]]) -> float:
    """Average end-to-end latency in seconds."""
    return _average(result.get("latency", 0.0) for result in results)


def first_pass_rate(results: list[dict[str, Any]]) -> float:
    """Ratio of tasks passing validation without a failed test attempt."""
    if not results:
        return 0.0
    return _ratio(result.get("first_pass", False) for result in results)


def recovery_rate(results: list[dict[str, Any]]) -> float:
    """Ratio of all tasks recovered after at least one failed test attempt."""
    if not results:
        return 0.0
    return _ratio(result.get("recovered", False) for result in results)


def recovery_opportunity_count(results: list[dict[str, Any]]) -> int:
    """Count tasks that needed at least one recovery attempt."""
    return sum(1 for result in results if not result.get("first_pass", False))


def conditional_recovery_rate(results: list[dict[str, Any]]) -> float:
    """Ratio of recovered tasks among tasks that needed recovery."""
    opportunities = [
        result for result in results if not result.get("first_pass", False)
    ]
    if not opportunities:
        return 0.0
    return _ratio(result.get("recovered", False) for result in opportunities)


def average_changed_file_count(results: list[dict[str, Any]]) -> float:
    """Average number of files changed per task."""
    return _average(len(result.get("changed_files", []) or []) for result in results)


def forbidden_file_rate(results: list[dict[str, Any]]) -> float:
    """Ratio of tasks that touched a forbidden file."""
    if not results:
        return 0.0
    return _ratio(bool(result.get("forbidden_files_touched")) for result in results)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the full benchmark metric summary."""
    return {
        "num_tasks": len(results),
        "task_success_rate": task_success_rate(results),
        "test_pass_rate": test_pass_rate(results),
        "average_iteration_count": average_iteration_count(results),
        "total_token_cost": total_token_cost(results),
        "average_token_cost": average_token_cost(results),
        "average_latency": average_latency(results),
        "first_pass_rate": first_pass_rate(results),
        "recovery_rate": recovery_rate(results),
        "recovery_opportunity_count": recovery_opportunity_count(results),
        "conditional_recovery_rate": conditional_recovery_rate(results),
        "average_changed_file_count": average_changed_file_count(results),
        "forbidden_file_rate": forbidden_file_rate(results),
    }


def _ratio(values: Any) -> float:
    items = [bool(value) for value in values]
    if not items:
        return 0.0
    return sum(1 for value in items if value) / len(items)


def _average(values: Any) -> float:
    items = [float(value or 0.0) for value in values]
    if not items:
        return 0.0
    return mean(items)

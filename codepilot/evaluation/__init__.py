from .metrics import (
    average_changed_file_count,
    average_iteration_count,
    average_latency,
    average_token_cost,
    conditional_recovery_rate,
    first_pass_rate,
    forbidden_file_rate,
    recovery_opportunity_count,
    recovery_rate,
    summarize_results,
    task_success_rate,
    test_pass_rate,
    total_token_cost,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkTask",
    "average_iteration_count",
    "average_latency",
    "average_token_cost",
    "average_changed_file_count",
    "first_pass_rate",
    "forbidden_file_rate",
    "conditional_recovery_rate",
    "recovery_opportunity_count",
    "recovery_rate",
    "load_tasks",
    "run_benchmark",
    "summarize_results",
    "task_success_rate",
    "test_pass_rate",
    "total_token_cost",
]


def __getattr__(name: str):
    """Load benchmark symbols lazily so `python -m evaluation.benchmark` stays quiet."""
    if name in {"BenchmarkRunner", "BenchmarkTask", "load_tasks", "run_benchmark"}:
        from .benchmark import BenchmarkRunner, BenchmarkTask, load_tasks, run_benchmark

        return {
            "BenchmarkRunner": BenchmarkRunner,
            "BenchmarkTask": BenchmarkTask,
            "load_tasks": load_tasks,
            "run_benchmark": run_benchmark,
        }[name]
    raise AttributeError(name)

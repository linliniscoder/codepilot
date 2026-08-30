from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import agent_variants, build_benchmark_40
from evaluation.agent_variants import DisabledDebugReflection
from evaluation.experiments import ExperimentRunner


def test_agent_variants_apply_expected_ablation_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class FakeCodingAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)
            self.kwargs = kwargs

    monkeypatch.setattr(agent_variants, "CodingAgent", FakeCodingAgent)

    full = agent_variants.build_agent("full", workspace="demo")
    no_debug = agent_variants.build_agent("no_debug_reflection", workspace="demo")
    no_recovery = agent_variants.build_agent("no_recovery", workspace="demo")
    one_shot = agent_variants.build_agent("one_shot", workspace="demo")
    factory = agent_variants.build_agent_factory("full", workspace="demo")
    factory_agent = factory()

    assert isinstance(full, FakeCodingAgent)
    assert isinstance(no_debug, FakeCodingAgent)
    assert isinstance(no_recovery, FakeCodingAgent)
    assert isinstance(one_shot, FakeCodingAgent)
    assert isinstance(factory_agent, FakeCodingAgent)
    assert isinstance(no_debug.kwargs["debug_reflection"], DisabledDebugReflection)
    assert no_recovery.kwargs["max_failures"] == 1
    assert one_shot.kwargs["max_iterations"] == 1
    assert one_shot.kwargs["max_failures"] == 1
    assert len(captured) == 5


def test_benchmark_40_generator_builds_consistent_tasks() -> None:
    specs = build_benchmark_40.build_suite_specs()
    tasks = build_benchmark_40.build_tasks(specs)
    hidden_tests = build_benchmark_40.build_hidden_test_content_map(specs)

    assert len(specs) == 5
    assert len(tasks) == 40
    assert len(hidden_tests) == 40
    assert {spec.name for spec in specs} == {
        "math_average",
        "text_slugify",
        "order_total",
        "missing_module",
        "word_count",
    }
    assert all(
        task["public_test_command"] == "python -m pytest tests -q"
        for task in tasks
    )
    assert all(
        task["hidden_test_command"].startswith("python -m pytest hidden_tests/")
        for task in tasks
    )
    assert {
        (task["hidden_test_source"], task["hidden_test_command"].split()[-2].removeprefix("hidden_tests/"))
        for task in tasks
    } == {
        (f"evaluation/hidden_tests/{spec.name}", variant.hidden_test_filename)
        for spec in specs
        for variant in spec.variants
    }


def test_experiment_runner_writes_variant_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "id": "demo",
                    "issue": "修复 demo",
                    "repo": ".",
                    "public_test_command": "python -m pytest -q",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    raw_dir = tmp_path / "raw"
    output_path = tmp_path / "summary.json"
    variant_summaries = {
        "full": {
            "num_tasks": 1,
            "task_success_rate": 0.5,
            "test_pass_rate": 0.5,
            "first_pass_rate": 0.5,
            "recovery_rate": 0.0,
            "conditional_recovery_rate": 0.0,
            "average_iteration_count": 2.0,
            "average_token_cost": 12.0,
            "average_latency": 1.5,
            "average_changed_file_count": 1.0,
            "forbidden_file_rate": 0.0,
        },
        "no_recovery": {
            "num_tasks": 1,
            "task_success_rate": 0.25,
            "test_pass_rate": 0.25,
            "first_pass_rate": 0.25,
            "recovery_rate": 0.0,
            "conditional_recovery_rate": 0.0,
            "average_iteration_count": 1.0,
            "average_token_cost": 8.0,
            "average_latency": 1.0,
            "average_changed_file_count": 1.0,
            "forbidden_file_rate": 0.0,
        },
    }
    calls: list[tuple[str, str]] = []

    class FakeBenchmarkRunner:
        def __init__(self, agent_factory, results_path: str | Path, **kwargs: object) -> None:
            self.agent_factory = agent_factory
            self.results_path = Path(results_path)
            self.kwargs = kwargs

        def run(self, tasks_path_arg: str | Path) -> dict[str, object]:
            calls.append((self.results_path.stem, str(tasks_path_arg)))
            payload = {
                "summary": variant_summaries[self.results_path.stem],
                "results": [{"variant": self.results_path.stem}],
            }
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return payload

    monkeypatch.setattr("evaluation.experiments.BenchmarkRunner", FakeBenchmarkRunner)

    payload = ExperimentRunner(
        tasks_path=tasks_path,
        output_path=output_path,
        raw_results_dir=raw_dir,
        variants=("full", "no_recovery"),
    ).run()

    assert output_path.is_file()
    assert (raw_dir / "full.json").is_file()
    assert (raw_dir / "no_recovery.json").is_file()
    assert calls == [("full", str(tasks_path)), ("no_recovery", str(tasks_path))]
    assert payload["comparison"]["baseline"] == "full"
    assert payload["comparison"]["metrics"]["full"]["task_success_rate"]["delta"] == 0.0
    assert payload["comparison"]["metrics"]["no_recovery"]["task_success_rate"]["delta"] == -0.25
    assert len(payload["variants"]) == 2

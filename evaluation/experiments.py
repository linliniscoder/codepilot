from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_variants import available_variants, build_agent_factory
from .benchmark import BenchmarkRunner, load_tasks, save_results


DEFAULT_TASKS_PATH = Path(__file__).with_name("benchmark_40.json")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("experiment_results.json")
DEFAULT_RAW_RESULTS_DIR = Path(__file__).with_name("experiment_results")
DEFAULT_VARIANT_NAMES = (
    "full",
    "no_debug_reflection",
    "no_recovery",
    "one_shot",
)

_SUMMARY_METRICS = (
    "task_success_rate",
    "test_pass_rate",
    "first_pass_rate",
    "recovery_rate",
    "conditional_recovery_rate",
    "average_iteration_count",
    "average_token_cost",
    "average_latency",
    "average_changed_file_count",
    "forbidden_file_rate",
)


@dataclass(frozen=True)
class ExperimentVariantResult:
    name: str
    description: str
    results_path: str
    summary: dict[str, Any]


class ExperimentRunner:
    """Run a benchmark suite across several agent variants."""

    def __init__(
        self,
        tasks_path: str | Path = DEFAULT_TASKS_PATH,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
        raw_results_dir: str | Path = DEFAULT_RAW_RESULTS_DIR,
        variants: tuple[str, ...] = DEFAULT_VARIANT_NAMES,
    ) -> None:
        self.tasks_path = Path(tasks_path)
        self.output_path = Path(output_path)
        self.raw_results_dir = Path(raw_results_dir)
        self.variants = tuple(variant.strip() for variant in variants if variant.strip())

    def run(self) -> dict[str, Any]:
        """Execute the benchmark for every configured variant."""
        load_tasks(self.tasks_path)  # validates early and keeps the error local
        self.raw_results_dir.mkdir(parents=True, exist_ok=True)

        variant_results = [self._run_variant(variant) for variant in self.variants]
        payload = {
            "tasks_path": str(self.tasks_path),
            "output_path": str(self.output_path),
            "variants": [result.__dict__ for result in variant_results],
            "comparison": self._build_comparison(variant_results),
        }
        save_results(payload, self.output_path)
        return payload

    def _run_variant(self, variant: str) -> ExperimentVariantResult:
        results_path = self.raw_results_dir / f"{variant}.json"
        runner = BenchmarkRunner(
            agent_factory=build_agent_factory(variant),
            results_path=results_path,
        )
        payload = runner.run(self.tasks_path)
        description = next(
            (
                item.description
                for item in available_variants()
                if item.name == variant
            ),
            variant,
        )
        return ExperimentVariantResult(
            name=variant,
            description=description,
            results_path=str(results_path),
            summary=payload["summary"],
        )

    @staticmethod
    def _build_comparison(
        variant_results: list[ExperimentVariantResult],
    ) -> dict[str, Any]:
        if not variant_results:
            return {}

        baseline = variant_results[0]
        baseline_summary = baseline.summary
        comparison: dict[str, Any] = {
            "baseline": baseline.name,
            "metrics": {},
        }
        for result in variant_results:
            metrics: dict[str, Any] = {}
            for metric in _SUMMARY_METRICS:
                current_value = result.summary.get(metric)
                baseline_value = baseline_summary.get(metric)
                if isinstance(current_value, (int, float)) and isinstance(
                    baseline_value, (int, float)
                ):
                    metrics[metric] = {
                        "value": current_value,
                        "delta": current_value - baseline_value,
                    }
                else:
                    metrics[metric] = {"value": current_value, "delta": None}
            comparison["metrics"][result.name] = metrics
        return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CodePilot experiment variants")
    parser.add_argument(
        "--tasks",
        default=str(DEFAULT_TASKS_PATH),
        help="Path to the benchmark task JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for the combined experiment summary JSON.",
    )
    parser.add_argument(
        "--raw-results-dir",
        default=str(DEFAULT_RAW_RESULTS_DIR),
        help="Directory that receives per-variant benchmark JSON files.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANT_NAMES),
        help="Agent variant names to run in order.",
    )
    return parser


def run_experiments(
    tasks_path: str | Path = DEFAULT_TASKS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    raw_results_dir: str | Path = DEFAULT_RAW_RESULTS_DIR,
    variants: tuple[str, ...] = DEFAULT_VARIANT_NAMES,
) -> dict[str, Any]:
    return ExperimentRunner(
        tasks_path=tasks_path,
        output_path=output_path,
        raw_results_dir=raw_results_dir,
        variants=variants,
    ).run()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_experiments(
        tasks_path=args.tasks,
        output_path=args.output,
        raw_results_dir=args.raw_results_dir,
        variants=tuple(args.variants),
    )
    print(json.dumps(payload["comparison"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

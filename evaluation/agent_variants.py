from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.controller import CodingAgent


@dataclass(frozen=True)
class AgentVariant:
    name: str
    description: str


class DisabledDebugReflection:
    """Lightweight debug-reflection stub for ablation runs."""

    last_usage: dict[str, Any]

    def __init__(self) -> None:
        self.last_usage = {}

    def analyze(
        self,
        previous_action: str | dict[str, Any],
        error_message: str,
        code_context: str,
    ) -> dict[str, str]:
        self.last_usage = {}
        return {
            "analysis": "Debug reflection disabled for this run.",
            "suggestion": "Continue from the raw tool output and the current source context.",
            "next_action": "Re-read the failing test and source files, then continue.",
        }


DEFAULT_VARIANTS: tuple[AgentVariant, ...] = (
    AgentVariant("full", "Default controller with debug reflection and recovery"),
    AgentVariant("no_debug_reflection", "Disable the debug-reflection step"),
    AgentVariant("no_recovery", "Stop after the first failed test"),
    AgentVariant("one_shot", "Single-step run with no iterative recovery"),
)


def available_variants() -> tuple[AgentVariant, ...]:
    """Return the supported benchmark variants in declaration order."""
    return DEFAULT_VARIANTS


def build_agent_factory(
    variant: str,
    **kwargs: Any,
) -> Callable[[], CodingAgent]:
    """Create a zero-argument factory for the selected experiment variant."""
    variant_name = variant.strip().lower()

    def factory() -> CodingAgent:
        return build_agent(variant_name, **kwargs)

    return factory


def build_agent(
    variant: str,
    **kwargs: Any,
) -> CodingAgent:
    """Create one CodingAgent configured for a specific evaluation variant."""
    variant_name = variant.strip().lower()

    if variant_name == "full":
        return CodingAgent(**kwargs)
    if variant_name == "no_debug_reflection":
        return CodingAgent(debug_reflection=DisabledDebugReflection(), **kwargs)
    if variant_name == "no_recovery":
        return CodingAgent(max_failures=1, **kwargs)
    if variant_name == "one_shot":
        return CodingAgent(max_iterations=1, max_failures=1, **kwargs)

    raise ValueError(f"Unknown agent variant: {variant}")

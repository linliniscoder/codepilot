from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Shared mutable state for future planner/controller modules."""

    task: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)
    llm_usage: list[dict[str, Any]] = Field(default_factory=list)
    current_plan: list[dict[str, Any]] = Field(default_factory=list)
    pending_plan_steps: list[dict[str, Any]] = Field(default_factory=list)
    completed_edit_goals: list[str] = Field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    failure_paths: list[str] = Field(default_factory=list)
    last_test_failure: dict[str, Any] = Field(default_factory=dict)
    last_debug_result: dict[str, str] = Field(default_factory=dict)
    last_tool_failure: dict[str, Any] = Field(default_factory=dict)
    last_test_signature: str = ""
    stagnation_count: int = 0
    iterations: int = 0
    changed_files: list[str] = Field(default_factory=list)
    last_error: str | None = None
    tests_run: bool = False
    tests_passed: bool = False
    failure_count: int = 0
    success: bool = False
    termination_reason: str | None = None

    def update(self, **kwargs: Any) -> "AgentState":
        """Update state fields in place and return self for chaining."""
        valid_fields = self._field_names()
        unknown_fields = set(kwargs) - valid_fields
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown AgentState field(s): {unknown}")

        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def append_history(self, role: str, content: str, **metadata: Any) -> "AgentState":
        """Append one conversation message to the state history."""
        message: dict[str, Any] = {
            "role": role,
            "content": content,
        }
        if metadata:
            message.update(metadata)
        self.history.append(message)
        return self

    def add_tool_result(
        self,
        name: str,
        result: dict[str, Any],
        **metadata: Any,
    ) -> "AgentState":
        """Append one tool execution result."""
        tool_result: dict[str, Any] = {
            "name": name,
            "result": result,
        }
        if metadata:
            tool_result.update(metadata)
        self.tool_results.append(tool_result)
        return self

    def to_json(self) -> str:
        """Serialize the state to JSON."""
        if hasattr(self, "model_dump_json"):
            return self.model_dump_json(indent=2)
        return self.json(indent=2)

    @classmethod
    def _field_names(cls) -> set[str]:
        if hasattr(cls, "model_fields"):
            return set(cls.model_fields)
        return set(cls.__fields__)

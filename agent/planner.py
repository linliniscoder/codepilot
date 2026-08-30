from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from llm.prompts import PLANNER_SYSTEM_PROMPT

from .json_utils import loads_with_repaired_escapes


LOGGER = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        ...


class AgentPlanner:
    """Planner that decomposes a user issue into structured tool-oriented steps."""

    def __init__(self, llm_client: LLMClientProtocol | None = None) -> None:
        self.llm_client = llm_client or self._default_llm_client()
        self.last_usage: dict[str, Any] = {}

    def create_plan(self, issue: str) -> dict[str, list[dict[str, Any]]]:
        """Create a structured plan for the given user issue."""
        LOGGER.info("Planner started: issue_length=%d", len(issue))
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(issue)},
        ]

        response = self.llm_client.chat(
            messages,
            temperature=0.0,
            max_tokens=1024,
        )
        usage = response.get("usage", {})
        self.last_usage = dict(usage) if isinstance(usage, dict) else {}
        content = response.get("content", "")

        try:
            plan = self._parse_plan(content)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            LOGGER.warning("Planner JSON parse failed, using fallback plan: %s", exc)
            plan = self._fallback_plan(issue, str(exc))

        LOGGER.info("Planner completed: steps=%d", len(plan["steps"]))
        return plan

    @staticmethod
    def _build_user_prompt(issue: str) -> str:
        return (
            "请为下面的用户issue生成执行计划。\n\n"
            f"用户issue:\n{issue}\n\n"
            "只输出JSON。"
        )

    @classmethod
    def _parse_plan(cls, content: str) -> dict[str, list[dict[str, Any]]]:
        raw_plan = loads_with_repaired_escapes(cls._extract_json(content))
        if not isinstance(raw_plan, dict):
            raise ValueError("planner response must be a JSON object")

        steps = raw_plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("planner response must contain non-empty steps")

        normalized_steps = [
            cls._normalize_step(index=index, step=step)
            for index, step in enumerate(steps, start=1)
        ]
        return {"steps": normalized_steps}

    @staticmethod
    def _extract_json(content: str) -> str:
        text = content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            return text[start : end + 1]
        return text

    @staticmethod
    def _normalize_step(index: int, step: Any) -> dict[str, Any]:
        if not isinstance(step, dict):
            raise ValueError(f"step {index} must be an object")

        description = step.get("description")
        tool = step.get("tool")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"step {index} description is required")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError(f"step {index} tool is required")

        return {
            "id": index,
            "description": description.strip(),
            "tool": tool.strip(),
        }

    @staticmethod
    def _fallback_plan(issue: str, error: str) -> dict[str, list[dict[str, Any]]]:
        LOGGER.warning(
            "Using fallback planner output: issue_length=%d error=%s",
            len(issue),
            error,
        )
        return {
            "steps": [
                {
                    "id": 1,
                    "description": "搜索与用户issue相关的代码文件",
                    "tool": "search_files",
                },
                {
                    "id": 2,
                    "description": "读取候选文件并定位需要修改的位置",
                    "tool": "read_file",
                },
                {
                    "id": 3,
                    "description": "使用精确文本替换修改已有源文件",
                    "tool": "edit_file",
                },
                {
                    "id": 4,
                    "description": "运行测试验证修复结果",
                    "tool": "run_tests",
                },
            ]
        }

    @staticmethod
    def _default_llm_client() -> LLMClientProtocol:
        from llm.client import VLLMClient

        return VLLMClient()


def plan_task(issue: str, llm_client: LLMClientProtocol | None = None) -> dict[str, Any]:
    """Convenience function for one-off planning."""
    return AgentPlanner(llm_client=llm_client).create_plan(issue)

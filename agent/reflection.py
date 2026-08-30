from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm.prompts import DEBUG_REFLECTION_SYSTEM_PROMPT

from .planner import LLMClientProtocol
from .json_utils import loads_with_repaired_escapes


LOGGER = logging.getLogger(__name__)


class DebugReflection:
    """LLM-backed analyzer for failed test output and code context."""

    def __init__(self, llm_client: LLMClientProtocol | None = None) -> None:
        self.llm_client = llm_client or self._default_llm_client()
        self.last_usage: dict[str, Any] = {}

    def analyze(
        self,
        previous_action: str | dict[str, Any],
        error_message: str,
        code_context: str,
    ) -> dict[str, str]:
        """Analyze a failed action without modifying code or executing tools."""
        LOGGER.info(
            "Debug reflection started: action_type=%s error_length=%d context_length=%d",
            type(previous_action).__name__,
            len(error_message),
            len(code_context),
        )
        messages = [
            {"role": "system", "content": DEBUG_REFLECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_prompt(
                    previous_action,
                    error_message,
                    code_context,
                ),
            },
        ]

        try:
            response = self.llm_client.chat(
                messages,
                temperature=0.0,
                max_tokens=768,
            )
            usage = response.get("usage", {})
            self.last_usage = dict(usage) if isinstance(usage, dict) else {}
            result = self._parse_response(response.get("content", ""))
        except Exception as exc:
            LOGGER.warning("Debug reflection failed, using fallback result: %s", exc)
            self.last_usage = {}
            result = self._fallback_result(error_message)

        LOGGER.info(
            "Debug reflection completed: analysis_length=%d suggestion_length=%d",
            len(result["analysis"]),
            len(result["suggestion"]),
        )
        return result

    @staticmethod
    def _build_prompt(
        previous_action: str | dict[str, Any],
        error_message: str,
        code_context: str,
    ) -> str:
        if isinstance(previous_action, dict):
            action_text = json.dumps(previous_action, ensure_ascii=False, indent=2)
        else:
            action_text = previous_action

        return (
            "请分析下面的pytest失败案例。\n\n"
            f"上一步动作:\n{action_text}\n\n"
            f"测试错误:\n{error_message[:2600]}\n\n"
            f"相关代码上下文:\n{code_context[:3600]}\n\n"
            "只输出JSON。"
        )

    @classmethod
    def _parse_response(cls, content: str) -> dict[str, str]:
        data = cls._parse_json_object(content)
        result = {
            "analysis": cls._required_string(data, "analysis"),
            "suggestion": cls._required_string(data, "suggestion"),
            "next_action": cls._required_string(data, "next_action"),
        }
        return result

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        text = content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                text = text[start : end + 1]

        data = loads_with_repaired_escapes(text)
        if not isinstance(data, dict):
            raise ValueError("reflection response must be a JSON object")
        return data

    @staticmethod
    def _required_string(data: dict[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"reflection field '{key}' is required")
        return value.strip()

    @staticmethod
    def _fallback_result(error_message: str) -> dict[str, str]:
        short_error = error_message.strip().splitlines()[-1] if error_message.strip() else ""
        return {
            "analysis": f"LLM未能返回有效JSON。当前可见错误: {short_error}",
            "suggestion": "先根据pytest错误定位失败断言、异常类型和相关代码路径，再生成修复方案。",
            "next_action": "读取失败测试和被测代码，补充上下文后重新分析。",
        }

    @staticmethod
    def _default_llm_client() -> LLMClientProtocol:
        from llm.client import VLLMClient

        return VLLMClient()


def analyze_failure(
    previous_action: str | dict[str, Any],
    error_message: str,
    code_context: str,
    llm_client: LLMClientProtocol | None = None,
) -> dict[str, str]:
    """Convenience function for one-off pytest failure analysis."""
    return DebugReflection(llm_client=llm_client).analyze(
        previous_action=previous_action,
        error_message=error_message,
        code_context=code_context,
    )

from __future__ import annotations

from agent.reflection import DebugReflection


def test_debug_reflection_analyzes_pytest_failure() -> None:
    class FakeLLMClient:
        def chat(self, messages, temperature=0.2, max_tokens=2048):
            assert temperature == 0.0
            assert max_tokens == 768
            assert "代码debug专家" in messages[0]["content"]
            assert "AssertionError" in messages[1]["content"]
            return {
                "content": """{
                  "analysis": "add函数当前执行减法，导致测试期望3时得到-1。",
                  "suggestion": "将add函数实现改为返回a + b。",
                  "next_action": "读取calculator.py并修改add函数，然后重新运行pytest。"
                }""",
                "usage": {},
                "latency": 0.0,
            }

    result = DebugReflection(FakeLLMClient()).analyze(
        previous_action={"tool": "run_tests", "arguments": {}},
        error_message="AssertionError: assert -1 == 3",
        code_context="def add(a, b):\n    return a - b\n",
    )

    assert "减法" in result["analysis"]
    assert result["suggestion"] == "将add函数实现改为返回a + b。"
    assert result["next_action"].startswith("读取calculator.py")


def test_debug_reflection_falls_back_on_invalid_json() -> None:
    class BadJSONLLMClient:
        def chat(self, messages, temperature=0.2, max_tokens=2048):
            return {"content": "需要检查断言失败。", "usage": {}, "latency": 0.0}

    result = DebugReflection(BadJSONLLMClient()).analyze(
        previous_action="run_tests",
        error_message="AssertionError: assert -1 == 3",
        code_context="def add(a, b): return a - b",
    )

    assert set(result) == {"analysis", "suggestion", "next_action"}
    assert "LLM未能返回有效JSON" in result["analysis"]

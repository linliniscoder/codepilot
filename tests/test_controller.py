from __future__ import annotations

import json
from typing import Any

from agent.controller import CodingAgent
from agent.planner import LLMClientProtocol
from agent.state import AgentState
from tools.file_tool import EDIT_FILE_TOOL, READ_FILE_TOOL, read_file
from tools.registry import ToolDefinition, ToolRegistry


class FakePlanner:
    def create_plan(self, issue: str) -> dict[str, list[dict[str, Any]]]:
        return {
            "steps": [
                {
                    "id": 1,
                    "description": "运行测试",
                    "tool": "run_tests",
                }
            ]
        }


class LoopLLM:
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        system = messages[0]["content"]
        user = messages[1]["content"]
        if "代码debug专家" in system:
            return {
                "content": json.dumps(
                    {
                        "analysis": "测试断言失败，需要根据错误重新定位实现。",
                        "suggestion": "检查被测函数的边界条件。",
                        "next_action": "重新读取实现并修改。",
                    },
                    ensure_ascii=False,
                ),
                "usage": {"total_tokens": 3},
            }
        if "Action模块" in system:
            return {
                "content": '{"tool":"run_tests","arguments":{}}',
                "usage": {"total_tokens": 2},
            }
        if "Reflection模块" in system:
            done = '"success": true' in user
            return {
                "content": json.dumps(
                    {
                        "done": done,
                        "reflection": "测试通过。" if done else "测试失败。",
                    },
                    ensure_ascii=False,
                ),
                "usage": {"total_tokens": 1},
            }
        raise AssertionError(f"Unexpected prompt: {system}")


def make_test_registry(handler) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="run_tests",
            description="Run tests",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=handler,
        )
    )
    return registry


def test_failed_tests_trigger_debug_reflection_and_replan() -> None:
    attempts = 0
    events: list[dict[str, Any]] = []

    def run_tests() -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {
                "success": False,
                "output": "AssertionError: expected 5, got 1",
                "error": "test failed",
            }
        return {"success": True, "output": "1 passed", "error": ""}

    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=make_test_registry(run_tests),
        max_iterations=4,
        event_handler=events.append,
    )

    state = agent.run("修复测试失败")

    assert state.success is True
    assert state.tests_passed is True
    assert state.failure_count == 1
    assert state.termination_reason == "validated"
    assert sum(item["total_tokens"] for item in state.llm_usage) == 8
    assert [event["type"] for event in events].count("debug_reflection") == 1
    assert attempts == 2


def test_repeated_test_failures_stop_with_failure_reason() -> None:
    def run_tests() -> dict[str, Any]:
        return {
            "success": False,
            "output": "AssertionError: expected 5, got 1",
            "error": "test failed",
        }

    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=make_test_registry(run_tests),
        max_iterations=10,
        max_failures=2,
    )

    state: AgentState = agent.run("修复测试失败")

    assert state.success is False
    assert state.failure_count == 2
    assert state.termination_reason == "too_many_test_failures"


def test_retrieval_context_is_saved_and_sent_to_planner() -> None:
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
            assert query == "定位登录函数"
            assert top_k == 5
            return [
                {
                    "file": "auth.py",
                    "symbol": "login",
                    "code": "def login(): pass",
                    "start_line": 1,
                    "end_line": 1,
                    "score": 0.9,
                }
            ]

    class CapturingPlanner(FakePlanner):
        def __init__(self) -> None:
            self.issue = ""

        def create_plan(self, issue: str) -> dict[str, list[dict[str, Any]]]:
            self.issue = issue
            return super().create_plan(issue)

    def run_tests() -> dict[str, Any]:
        return {"success": True, "output": "1 passed", "error": ""}

    planner = CapturingPlanner()
    events: list[dict[str, Any]] = []
    agent = CodingAgent(
        planner=planner,
        llm_client=LoopLLM(),
        tool_registry=make_test_registry(run_tests),
        retriever=FakeRetriever(),
        event_handler=events.append,
    )

    state = agent.run("定位登录函数")

    assert state.success is True
    assert state.retrieved_context[0]["file"] == "auth.py"
    assert "auth.py" in planner.issue
    assert any(event["type"] == "retrieval" for event in events)


def test_issue_context_prefetches_named_source_files(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    (tmp_path / "helpers.py").write_text(
        "def normalize_name(name):\n    return name.strip()\n",
        encoding="utf-8",
    )

    class CapturingPlanner(FakePlanner):
        def __init__(self) -> None:
            self.issue = ""

        def create_plan(self, issue: str) -> dict[str, list[dict[str, Any]]]:
            self.issue = issue
            return super().create_plan(issue)

    def run_tests() -> dict[str, Any]:
        return {"success": True, "output": "1 passed", "error": ""}

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    registry.register(
        ToolDefinition(
            name="run_tests",
            description="Run tests",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=run_tests,
        )
    )

    planner = CapturingPlanner()
    state = CodingAgent(
        planner=planner,
        llm_client=LoopLLM(),
        tool_registry=registry,
        max_iterations=2,
    ).run("修复 helpers.py：display_name 需要折叠多个空格。")

    assert state.success is True
    assert "helpers.py" in planner.issue
    assert "return name.strip()" in planner.issue
    assert any(
        item.get("phase") == "issue_context"
        and item.get("arguments", {}).get("path") == "helpers.py"
        for item in state.tool_results
    )


def test_failure_diagnostics_are_sent_to_replan_and_related_files_are_read(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_widget.py").write_text(
        "from widget import value\n\ndef test_value():\n    assert value() == 2\n",
        encoding="utf-8",
    )
    (tmp_path / "widget.py").write_text(
        "def value():\n    return 1\n",
        encoding="utf-8",
    )

    attempts = 0

    def run_tests() -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {
                "success": False,
                "output": (
                    "=========================== short test summary info ===========================\n"
                    "FAILED tests/test_widget.py:4\n"
                    "E       AssertionError: assert 1 == 2"
                ),
                "error": "test command failed: 1",
                "return_code": 1,
            }
        return {"success": True, "output": "1 passed", "error": ""}

    class CapturingPlanner(FakePlanner):
        def __init__(self) -> None:
            self.issues: list[str] = []

        def create_plan(self, issue: str) -> dict[str, list[dict[str, Any]]]:
            self.issues.append(issue)
            return super().create_plan(issue)

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    registry.register(
        ToolDefinition(
            name="run_tests",
            description="Run tests",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=run_tests,
        )
    )
    planner = CapturingPlanner()
    state = CodingAgent(
        planner=planner,
        llm_client=LoopLLM(),
        tool_registry=registry,
        max_iterations=4,
    ).run("修复 widget 测试失败")

    assert state.success is True
    assert state.failure_paths == ["tests/test_widget.py", "widget.py"]
    assert state.last_debug_result["suggestion"] == "检查被测函数的边界条件。"
    assert len(planner.issues) == 2
    assert "widget.py" in planner.issues[1]
    assert "检查被测函数的边界条件" in planner.issues[1]
    assert any(
        item.get("phase") == "failure_context"
        and item.get("arguments", {}).get("path") == "widget.py"
        for item in state.tool_results
    )


def test_failure_limit_reason_is_not_overwritten_by_iteration_limit() -> None:
    def run_tests() -> dict[str, Any]:
        return {
            "success": False,
            "output": "AssertionError: expected 5, got 1",
            "error": "test failed",
        }

    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=make_test_registry(run_tests),
        max_iterations=2,
        max_failures=2,
    )

    state = agent.run("修复测试失败")

    assert state.failure_count == 2
    assert state.termination_reason == "too_many_test_failures"


def test_edit_action_requires_read_and_injects_latest_file_hash(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    path = tmp_path / "sample.py"
    path.write_text("def value():\n    return 1\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(EDIT_FILE_TOOL)
    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=registry,
    )
    unread_state = AgentState(task="修改 sample")

    rejected = agent._execute_action(
        unread_state,
        "edit_file",
        {
            "path": "sample.py",
            "old_text": "return 1",
            "new_text": "return 2",
        },
    )

    assert rejected["success"] is False
    assert "must be read before edit_file" in rejected["error"]

    read_result = read_file("sample.py")
    state = AgentState(task="修改 sample")
    state.add_tool_result(
        "read_file",
        read_result,
        arguments={"path": "sample.py"},
    )
    action = agent._prepare_action(
        state,
        {"tool": "edit_file", "description": "修改 sample.py"},
        {
            "tool": "edit_file",
            "arguments": {
                "path": "sample.py",
                "old_text": "return 1",
                "new_text": "return 2",
            },
        },
    )

    assert action["arguments"]["expected_sha256"] == read_result["sha256"]
    edited = agent._execute_action(state, action["tool"], action["arguments"])
    assert edited["success"] is True
    assert "return 2" in path.read_text(encoding="utf-8")


def test_action_normalization_discards_stale_arguments_for_planned_tool(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))

    def run_tests() -> dict[str, Any]:
        return {"success": True, "output": "1 passed", "error": ""}

    registry = make_test_registry(run_tests)
    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=registry,
    )
    action = agent._prepare_action(
        AgentState(task="运行测试"),
        {"tool": "run_tests", "description": "运行测试"},
        {
            "tool": "edit_file",
            "arguments": {"path": "sample.py", "old_text": "a", "new_text": "b"},
        },
    )

    assert action == {"tool": "run_tests", "arguments": {}}


def test_action_normalization_uses_failure_path_for_stale_read_action(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=registry,
    )
    state = AgentState(task="读取失败文件", failure_paths=["calculator.py"])

    action = agent._prepare_action(
        state,
        {"tool": "read_file", "description": "读取失败文件"},
        {
            "tool": "edit_file",
            "arguments": {"path": "stale.py", "old_text": "a", "new_text": "b"},
        },
    )

    assert action == {"tool": "read_file", "arguments": {"path": "calculator.py"}}


def test_read_fallback_uses_latest_search_result(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=registry,
    )
    state = AgentState(task="读取搜索到的文件")
    state.add_tool_result(
        "search_files",
        {"success": True, "output": "tests/test_demo.py\ndemo.py", "error": ""},
        arguments={"keyword": "*.py"},
    )

    action = agent._prepare_action(
        state,
        {"tool": "read_file", "description": "读取第一个Python文件"},
        {"tool": "edit_file", "arguments": {}},
    )

    assert action == {"tool": "read_file", "arguments": {"path": "demo.py"}}


def test_plan_sanitizer_drops_overwrite_after_edit_and_adds_validation() -> None:
    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=make_test_registry(lambda: {"success": True}),
    )

    plan = agent._sanitize_plan(
        {
            "steps": [
                {"id": 1, "description": "读取文件", "tool": "read_file"},
                {"id": 2, "description": "编辑文件", "tool": "edit_file"},
                {"id": 3, "description": "覆盖保存文件", "tool": "write_file"},
            ]
        }
    )

    assert [step["tool"] for step in plan["steps"]] == [
        "read_file",
        "edit_file",
        "run_tests",
    ]
    assert [step["id"] for step in plan["steps"]] == [1, 2, 3]


def test_plan_sanitizer_validates_before_a_second_edit() -> None:
    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=make_test_registry(lambda: {"success": True}),
    )

    plan = agent._sanitize_plan(
        {
            "steps": [
                {"id": 1, "description": "读取文件", "tool": "read_file"},
                {"id": 2, "description": "第一次编辑", "tool": "edit_file"},
                {"id": 3, "description": "第二次编辑", "tool": "edit_file"},
            ]
        }
    )

    assert [step["tool"] for step in plan["steps"]] == [
        "read_file",
        "edit_file",
        "run_tests",
        "edit_file",
        "run_tests",
    ]


def test_completed_edit_goals_ignore_duplicate_work_but_keep_new_requirement() -> None:
    state = AgentState(
        current_plan=[
            {"id": 1, "description": "编辑 text_utils.py 的 slugify", "tool": "edit_file"},
            {"id": 2, "description": "运行pytest验证 slugify", "tool": "run_tests"},
            {"id": 3, "description": "再次修复 slugify", "tool": "edit_file"},
            {"id": 4, "description": "修复 calculator.py 的 average", "tool": "edit_file"},
        ],
        completed_edit_goals=["slugify", "text_utils"],
    )

    assert CodingAgent._has_pending_work(state, 1) is True
    state.current_plan = state.current_plan[:3]
    assert CodingAgent._has_pending_work(state, 1) is False


def test_failed_step_preserves_only_unrelated_pending_requirements() -> None:
    state = AgentState(
        current_plan=[
            {"id": 1, "description": "读取 calculator.py", "tool": "read_file"},
            {"id": 2, "description": "修复 add", "tool": "edit_file"},
            {"id": 3, "description": "运行测试", "tool": "run_tests"},
            {"id": 4, "description": "继续修复 add", "tool": "edit_file"},
            {"id": 5, "description": "修复 average 空列表", "tool": "edit_file"},
        ]
    )

    pending = CodingAgent._filter_pending_plan_steps(
        state,
        2,
        {
            "success": False,
            "output": "FAILED tests/test_calculator.py::test_add - assert add(2, 3) == 5",
            "error": "test failed",
        },
        {
            "analysis": "add 使用了错误的减法运算",
            "suggestion": "修复 add 的实现",
        },
    )

    assert [step["description"] for step in pending] == ["修复 average 空列表"]


def test_replan_merges_generated_steps_without_duplicate_goals() -> None:
    generated = [
        {"id": 1, "description": "读取 calculator.py", "tool": "read_file"},
        {"id": 2, "description": "修复 average", "tool": "edit_file"},
    ]
    pending = [
        {"id": 4, "description": "再次读取 calculator.py", "tool": "read_file"},
        {"id": 5, "description": "继续修复 average 空列表", "tool": "edit_file"},
        {"id": 6, "description": "运行测试", "tool": "run_tests"},
    ]

    merged = CodingAgent._merge_pending_plan_steps(generated, pending)

    assert merged == generated


def test_preferred_source_path_ignores_failure_test_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    state = AgentState(
        failure_paths=["tests/test_word_count.py", "word_count.py"],
    )

    assert CodingAgent._preferred_source_path(state) == "word_count.py"


def test_diagnostic_hints_cover_empty_word_count_boundary() -> None:
    state = AgentState(
        tool_results=[
            {
                "name": "read_file",
                "arguments": {"path": "word_count.py"},
                "result": {
                    "success": True,
                    "output": "return len(re.split(r' ', value.strip()))",
                },
            }
        ]
    )

    hints = CodingAgent._diagnostic_hints(
        "AssertionError: assert 1 == 0 in test_count_words_empty_text",
        state,
    )

    assert "必须先返回 0" in hints
    assert "显式处理" in hints


def test_diagnostic_hints_protect_slugify_internal_separators() -> None:
    hints = CodingAgent._diagnostic_hints(
        "test_slugify_basic_sentence: assert 'hello world' == 'hello-world'",
        AgentState(),
    )

    assert "保留单个内部分隔符" in hints
    assert "不能删除" in hints


def test_replan_context_preserves_original_unfinished_requirements() -> None:
    state = AgentState(
        task="修复 add 和 average 的两个问题",
        current_plan=[
            {"id": 1, "description": "修复 add", "tool": "edit_file"},
            {"id": 2, "description": "处理 average 空列表", "tool": "edit_file"},
        ],
        last_test_failure={"output": "AssertionError", "return_code": 1},
        last_debug_result={"analysis": "只修复了 add"},
    )

    task = CodingAgent._build_replan_task(state)

    assert "处理 average 空列表" in task
    assert "保留原issue中尚未验证的要求" in task


def test_action_parser_repairs_raw_regex_backslash_in_json() -> None:
    content = (
        '{"tool":"edit_file","arguments":{'
        '"path":"word_count.py","old_text":"r\'\\s+\'",'
        '"new_text":"fixed"}}'
    )

    action = CodingAgent._parse_action(
        content,
        {"tool": "edit_file", "description": "编辑"},
    )

    assert action["arguments"]["old_text"] == r"r'\s+'"


def test_edit_action_ignores_model_hash_after_previous_edit(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CODEPILOT_WORKSPACE", str(tmp_path))
    path = tmp_path / "sample.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    from tools.file_tool import EDIT_FILE_TOOL

    registry.register(EDIT_FILE_TOOL)
    agent = CodingAgent(
        planner=FakePlanner(),
        llm_client=LoopLLM(),
        tool_registry=registry,
    )
    state = AgentState(task="修改 sample")
    read_result = read_file("sample.py")
    state.add_tool_result(
        "read_file",
        read_result,
        arguments={"path": "sample.py"},
    )
    first = agent._prepare_action(
        state,
        {"tool": "edit_file", "description": "第一次修改"},
        {
            "tool": "edit_file",
            "arguments": {"path": "sample.py", "old_text": "1", "new_text": "2"},
        },
    )
    first_result = agent._execute_action(state, "edit_file", first["arguments"])
    state.add_tool_result("edit_file", first_result, arguments=first["arguments"])

    second = agent._prepare_action(
        state,
        {"tool": "edit_file", "description": "第二次修改"},
        {
            "tool": "edit_file",
            "arguments": {
                "path": "sample.py",
                "old_text": "2",
                "new_text": "3",
                "expected_sha256": first_result["new_sha256"],
            },
        },
    )

    assert second["arguments"]["expected_sha256"] == read_result["sha256"]
    assert agent._execute_action(state, "edit_file", second["arguments"])["success"] is False

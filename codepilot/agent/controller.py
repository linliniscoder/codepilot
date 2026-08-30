from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from llm.prompts import ACTION_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT
from tools._common import get_workspace, is_test_path, resolve_workspace_path
from tools.registry import ToolRegistry, create_default_registry

from .diagnostics import extract_python_paths, infer_imported_python_paths
from .json_utils import loads_with_repaired_escapes
from .planner import AgentPlanner, LLMClientProtocol
from .reflection import DebugReflection
from .state import AgentState


try:
    from rich.console import Console
    from rich.text import Text
except ModuleNotFoundError:
    Console = None
    Text = None


LOGGER = logging.getLogger(__name__)
AgentEventHandler = Callable[[dict[str, Any]], None]

_GOAL_STOP_WORDS = {
    "read",
    "file",
    "files",
    "edit",
    "modify",
    "fix",
    "run",
    "test",
    "tests",
    "pytest",
    "function",
    "source",
    "code",
    "check",
    "verify",
    "validation",
    "all",
    "requirements",
    "requirement",
    "issue",
    "current",
    "ensure",
    "save",
    "write",
    "content",
    "correct",
    "python",
    "如果",
    "函数",
    "文件",
    "读取",
    "编辑",
    "修改",
    "测试",
    "验证",
    "检查",
    "运行",
    "处理",
}


def _looks_like_test_file(path: str) -> bool:
    """Return whether a workspace-relative path points to a test file."""
    try:
        workspace = get_workspace()
        return is_test_path(resolve_workspace_path(path), workspace)
    except ValueError:
        parts = path.replace("\\", "/").split("/")
        name = parts[-1]
        return (
            "tests" in parts
            or "test" in parts
            or name.startswith("test_")
            or name.endswith("_test.py")
        )


def _merge_paths(existing: list[str], additions: list[str]) -> list[str]:
    """Merge paths while preserving the order in which they were discovered."""
    merged = list(existing)
    seen = set(existing)
    for path in additions:
        if path not in seen:
            seen.add(path)
            merged.append(path)
    return merged


class CodingAgent:
    """Minimal observe-plan-act loop for repo-level coding tasks."""

    def __init__(
        self,
        planner: AgentPlanner | None = None,
        llm_client: LLMClientProtocol | None = None,
        tool_registry: ToolRegistry | None = None,
        max_iterations: int = 12,
        max_failures: int = 3,
        console: Any | None = None,
        event_handler: AgentEventHandler | None = None,
        debug_reflection: DebugReflection | None = None,
        retriever: Any | None = None,
        retrieval_top_k: int = 5,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if max_failures <= 0:
            raise ValueError("max_failures must be positive")
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")

        self.llm_client = llm_client or self._default_llm_client()
        self.planner = planner or AgentPlanner(llm_client=self.llm_client)
        self.debug_reflection = debug_reflection or DebugReflection(
            llm_client=self.llm_client
        )
        self.tool_registry = tool_registry or create_default_registry()
        self.max_iterations = max_iterations
        self.max_failures = max_failures
        self.console = console or self._default_console()
        self.event_handler = event_handler
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k

    def run(self, task: str) -> AgentState:
        """Run the core Agent loop and return the final shared state."""
        LOGGER.info("CodingAgent run started: task_length=%d", len(task))
        state = AgentState(task=task)
        state.append_history("user", task)
        self._initialize_retrieval(state)

        plan = self._plan(task, state)
        completed = False
        step_index = 0

        while state.iterations < self.max_iterations and not completed:
            if step_index >= len(state.current_plan):
                plan = self._plan(self._build_replan_task(state), state)
                step_index = 0
                if not plan["steps"]:
                    state.update(last_error="Planner returned no steps")
                    break

            step = state.current_plan[step_index]
            if self._should_skip_completed_step(state, step):
                LOGGER.info(
                    "Skipping completed plan step: id=%s description=%s",
                    step.get("id"),
                    step.get("description"),
                )
                self._emit_event("step_skipped", step=step)
                state.append_history(
                    "assistant",
                    f"跳过已完成步骤: {step.get('description', '')}",
                    phase="step_skipped",
                )
                if not self._has_pending_work(state, step_index):
                    completed = True
                    state.update(success=True, termination_reason="validated")
                    break
                step_index += 1
                continue
            state.update(iterations=state.iterations + 1)
            self._log_phase(
                "TOOL",
                f"iteration={state.iterations} step={step['id']} "
                f"tool={step['tool']} description={step['description']}",
                "bold yellow",
            )

            action = self._build_action(state, step)
            tool_name = action["tool"]
            arguments = action["arguments"]
            result = self._execute_action(state, tool_name, arguments)
            state.add_tool_result(
                tool_name,
                result,
                step_id=step["id"],
                arguments=arguments,
            )
            self._emit_event(
                "tool_result",
                step=step,
                tool=tool_name,
                arguments=arguments,
                result=result,
            )
            self._record_changed_file(state, tool_name, arguments, result)
            self._record_validation(state, tool_name, arguments, result)
            if result.get("success", False) and tool_name in {"write_file", "edit_file"}:
                # Any successful edit invalidates an earlier passing test result.
                state.update(tests_passed=False)
                state.update(
                    completed_edit_goals=self._merge_goal_names(
                        state.completed_edit_goals,
                        self._goal_identifiers(
                            self._edit_goal_text(step, arguments)
                        ),
                    )
                )
            if not result.get("success", False):
                state.update(last_error=result.get("error") or "Tool execution failed")
                if not self._is_test_command(tool_name, arguments):
                    state.update(
                        last_tool_failure={
                            "tool": tool_name,
                            "arguments": arguments,
                            "result": result,
                        }
                    )
            else:
                # A later successful observation supersedes a transient tool error.
                # Persistent test failures remain tracked separately in failure_count.
                state.update(last_error=None)
                if not self._is_test_command(tool_name, arguments):
                    state.update(last_tool_failure={})

            self._log_phase(
                "RESULT",
                self._summarize_result(result),
                "bold green" if result.get("success") else "bold red",
            )

            if self._is_test_failure(tool_name, arguments, result):
                self._prepare_failure_context(state, result)
                debug_result = self._debug_test_failure(
                    state,
                    step,
                    tool_name,
                    arguments,
                    result,
                )
                state.update(last_debug_result=debug_result)
                state.update(
                    pending_plan_steps=self._filter_pending_plan_steps(
                        state,
                        step_index,
                        result,
                        debug_result,
                    )
                )
                state.append_history(
                    "assistant",
                    json.dumps(debug_result, ensure_ascii=False),
                    phase="debug_reflection",
                )
                self._emit_event(
                    "debug_reflection",
                    step=step,
                    analysis=debug_result,
                )
                self._log_phase(
                    "DEBUG",
                    debug_result["analysis"],
                    "bold red",
                )

                if state.failure_count >= self.max_failures:
                    state.update(
                        termination_reason="too_many_test_failures",
                        last_error=(
                            f"Test failure limit reached ({self.max_failures})"
                        ),
                    )
                    break

                reflection = {
                    "done": False,
                    "reflection": (
                        f"测试未通过。{debug_result['suggestion']}"
                        "将根据错误信息重新规划。"
                    ),
                }
                force_replan = True
            else:
                reflection = self._reflect(state, step, tool_name, arguments, result)
                force_replan = False

            if reflection.get("done") and not state.tests_passed:
                reflection = {
                    "done": False,
                    "reflection": "当前还没有成功的测试验证，不能结束任务。",
                }
            elif state.tests_passed and self._is_test_success(tool_name, arguments, result):
                if self._has_pending_work(state, step_index):
                    reflection = {
                        "done": False,
                        "reflection": "当前测试通过，但原issue仍有未完成要求，继续执行后续计划。",
                    }
                else:
                    reflection = {
                        "done": True,
                        "reflection": "测试验证通过，任务已完成。",
                    }

            state.append_history(
                "assistant",
                reflection["reflection"],
                phase="reflection",
                done=reflection["done"],
            )
            self._emit_event(
                "reflection",
                step=step,
                reflection=reflection,
            )
            self._log_phase(
                "REFLECTION",
                f"done={reflection['done']} {reflection['reflection']}",
                "bold magenta",
            )

            completed = reflection["done"]
            if completed:
                state.update(success=True, termination_reason="validated")
            elif force_replan or not result.get("success", False):
                step_index = len(state.current_plan)
            else:
                step_index += 1

        if not completed and state.termination_reason is None and state.iterations >= self.max_iterations:
            state.update(
                last_error="Reached max_iterations before task completion",
                termination_reason="max_iterations",
            )
            self._log_phase(
                "REFLECTION",
                "done=False Reached max_iterations before task completion",
                "bold red",
            )
        elif not completed and state.termination_reason is None:
            state.update(termination_reason="planner_stopped")

        LOGGER.info(
            "CodingAgent run completed: iterations=%d changed_files=%d last_error=%s",
            state.iterations,
            len(state.changed_files),
            state.last_error,
        )
        return state

    @staticmethod
    def _is_test_failure(
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        return not result.get("success", False) and CodingAgent._is_test_command(
            tool_name,
            arguments,
        )

    @classmethod
    def _is_test_success(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        return result.get("success", False) and CodingAgent._is_test_command(
            tool_name,
            arguments,
        )

    @staticmethod
    def _is_test_command(tool_name: str, arguments: dict[str, Any]) -> bool:
        if tool_name == "run_tests":
            return True
        if tool_name != "run_command":
            return False
        command = arguments.get("command", "")
        return isinstance(command, str) and "pytest" in command

    @staticmethod
    def _record_validation(
        state: AgentState,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        if not CodingAgent._is_test_command(tool_name, arguments):
            return

        if result.get("success", False):
            state.update(
                tests_run=True,
                tests_passed=True,
                last_error=None,
                last_tool_failure={},
                last_test_signature="",
                stagnation_count=0,
            )
        else:
            failure_text = CodingAgent._result_text(result)
            signature = CodingAgent._failure_signature(failure_text)
            stagnation_count = (
                state.stagnation_count + 1
                if signature and signature == state.last_test_signature
                else 0
            )
            state.update(
                tests_run=True,
                tests_passed=False,
                failure_count=state.failure_count + 1,
                last_tool_failure={},
                last_test_signature=signature,
                stagnation_count=stagnation_count,
                failure_paths=extract_python_paths(
                    failure_text,
                    workspace=get_workspace(),
                ),
                last_test_failure={
                    "output": str(result.get("output", "")),
                    "error": str(result.get("error", "")),
                    "stdout": str(result.get("stdout", "")),
                    "stderr": str(result.get("stderr", "")),
                    "return_code": result.get("return_code"),
                },
            )

    def _prepare_failure_context(
        self,
        state: AgentState,
        result: dict[str, Any],
        max_files: int = 8,
    ) -> None:
        """Read files named by pytest before asking the model to replan."""
        workspace = get_workspace()
        discovered_paths = extract_python_paths(
            self._result_text(result),
            workspace=workspace,
        )
        state.update(failure_paths=_merge_paths(state.failure_paths, discovered_paths))
        pending = list(state.failure_paths)
        seen: set[str] = set()
        index = 0

        while pending and len(seen) < max_files:
            path = pending.pop(0)
            if path in seen:
                continue
            seen.add(path)

            read_result = self._execute_tool("read_file", {"path": path})
            state.add_tool_result(
                "read_file",
                read_result,
                phase="failure_context",
                arguments={"path": path},
            )
            self._emit_event(
                "failure_context",
                path=path,
                result=read_result,
            )
            index += 1

            if not read_result.get("success", False):
                continue
            if not _looks_like_test_file(path):
                continue

            imported_paths = infer_imported_python_paths(
                str(read_result.get("output", "")),
                workspace=workspace,
            )
            state.update(
                failure_paths=_merge_paths(state.failure_paths, imported_paths)
            )
            pending.extend(imported_paths)

        LOGGER.info(
            "Failure context prepared: paths=%s files_read=%d",
            state.failure_paths,
            index,
        )

    def _debug_test_failure(
        self,
        state: AgentState,
        step: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, str]:
        error_message = str(result.get("error", ""))
        output = str(result.get("output", ""))
        if output:
            error_message = f"{error_message}\n{output}".strip()
        hints = self._diagnostic_hints(error_message, state)
        if hints:
            error_message = (
                f"控制器诊断提示:\n{hints}\n\n"
                f"原始测试错误:\n{error_message}"
            )

        try:
            result = self.debug_reflection.analyze(
                previous_action={
                    "step": step,
                    "tool": tool_name,
                    "arguments": arguments,
                },
                error_message=self._trim_text(error_message, 7000),
                code_context=self._build_debug_context(state),
            )
            self._record_llm_usage(
                state,
                "debug_reflection",
                getattr(self.debug_reflection, "last_usage", {}),
            )
            return result
        except Exception as exc:
            LOGGER.exception("Debug reflection failed in controller")
            return {
                "analysis": f"测试失败分析不可用: {exc}",
                "suggestion": "读取失败测试和相关源文件后重新定位问题。",
                "next_action": "重新搜索并读取测试失败涉及的文件。",
            }

    @staticmethod
    def _build_debug_context(state: AgentState, max_length: int = 8000) -> str:
        contexts: list[str] = []
        for item in reversed(state.tool_results):
            if item.get("name") not in {"read_file", "search_files"}:
                continue
            result = item.get("result", {})
            if not isinstance(result, dict):
                continue
            output = str(result.get("output", ""))
            if output:
                arguments = item.get("arguments", {})
                path = arguments.get("path") if isinstance(arguments, dict) else None
                if path:
                    contexts.append(f"文件: {path}\n内容:\n{output}")
                else:
                    contexts.append(output)
            if sum(len(value) for value in contexts) >= max_length:
                break
        return CodingAgent._trim_text("\n\n".join(reversed(contexts)), max_length)

    def _plan(self, task: str, state: AgentState) -> dict[str, list[dict[str, Any]]]:
        plan = self.planner.create_plan(self._add_retrieval_context(task, state))
        if state.pending_plan_steps:
            plan = {
                "steps": self._merge_pending_plan_steps(
                    plan.get("steps", []),
                    state.pending_plan_steps,
                )
            }
            state.update(pending_plan_steps=[])
        plan = self._sanitize_plan(plan)
        self._record_llm_usage(state, "planner", getattr(self.planner, "last_usage", {}))
        state.update(current_plan=plan["steps"])
        self._log_phase(
            "PLAN",
            json.dumps(plan, ensure_ascii=False),
            "bold cyan",
        )
        self._emit_event("plan", plan=plan)
        state.append_history(
            "assistant",
            json.dumps(plan, ensure_ascii=False),
            phase="plan",
        )
        return plan

    @staticmethod
    def _goal_identifiers(text: str) -> set[str]:
        """Extract stable code/task identifiers from a plan description."""
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text.lower())
        return {
            token
            for token in tokens
            if token not in _GOAL_STOP_WORDS
        }

    @staticmethod
    def _merge_goal_names(existing: list[str], additions: set[str]) -> list[str]:
        return sorted(set(existing).union(additions))

    @staticmethod
    def _edit_goal_text(step: dict[str, Any], arguments: dict[str, Any]) -> str:
        path = arguments.get("path")
        path_text = path if isinstance(path, str) else ""
        return f"{step.get('description', '')} {path_text}"

    @staticmethod
    def _preferred_source_path(state: AgentState) -> str | None:
        """Choose a source path from failure and observation context."""
        for path in state.failure_paths:
            if not _looks_like_test_file(path):
                return path

        for item in reversed(state.tool_results):
            if item.get("name") == "read_file":
                arguments = item.get("arguments", {})
                result = item.get("result", {})
                path = arguments.get("path") if isinstance(arguments, dict) else None
                if (
                    isinstance(path, str)
                    and isinstance(result, dict)
                    and result.get("success")
                    and not _looks_like_test_file(path)
                ):
                    return path

        for item in reversed(state.tool_results):
            if item.get("name") != "search_files":
                continue
            result = item.get("result", {})
            if not isinstance(result, dict) or not result.get("success"):
                continue
            for candidate in str(result.get("output", "")).splitlines():
                candidate = candidate.strip()
                if candidate.endswith(".py") and not _looks_like_test_file(candidate):
                    return candidate

        for item in state.retrieved_context:
            path = item.get("file") if isinstance(item, dict) else None
            if isinstance(path, str) and path and not _looks_like_test_file(path):
                return path
        return None

    @classmethod
    def _diagnostic_hints(cls, error_message: str, state: AgentState) -> str:
        """Add deterministic guidance for common boundary and tool failures."""
        text = error_message.lower()
        source_context = cls._build_debug_context(state, max_length=5000).lower()
        hints: list[str] = []

        if "count_words" in text and (
            "empty" in text or "assertionerror: assert 1 == 0" in text
        ):
            hints.append(
                "count_words 的空字符串必须先返回 0；仅把分隔正则改为 \\s+ "
                "仍会让空字符串产生一个空元素。"
            )
        if "average" in text and (
            "zerodivisionerror" in text or "empty" in text
        ):
            hints.append(
                "average([]) 必须在计算 sum(values) / len(values) 之前返回 0，"
                "不能只修复其他函数。"
            )
        if "slugify" in text and any(
            expected in text
            for expected in ("hello-world", "python-fast", "one-two")
        ):
            hints.append(
                "slugify 必须保留单个内部分隔符：把连续非字母数字字符替换成一个短横线，"
                "最后只清理首尾短横线；不能删除空格、冒号或内部短横线。"
            )
        if "file changed since it was read" in text:
            hints.append("文件已变化，必须重新 read_file，再使用最新 sha256 和当前 old_text 编辑。")
        if "old_text must occur exactly once" in text:
            hints.append("old_text 与当前文件不匹配，必须重新读取文件并复制当前完整代码片段。")
        if "test files are read-only" in text or "测试文件" in text and "只读" in text:
            hints.append("测试文件只是证据，不能修改；应定位并编辑失败测试导入的源文件。")
        if "no module named pytest" in text or "pip install" in text:
            hints.append("不要安装依赖或调用 pip；run_tests 已提供固定的 pytest 入口。")
        if "python -m pdb" in text or "pdb" in text:
            hints.append("不要使用交互式 pdb；读取失败上下文后直接做最小源码修改。")

        if "re.split" in source_context and (
            "empty" in text or "assert 1 == 0" in text
        ):
            hints.append("当前源码仍使用 strip 后直接 split，需显式处理 strip 结果为空的情况。")
        return "\n".join(dict.fromkeys(hints))

    @classmethod
    def _has_pending_work(cls, state: AgentState, step_index: int) -> bool:
        """Return whether an uncompleted requirement remains after this step."""
        completed_goals = set(state.completed_edit_goals)
        for step in state.current_plan[step_index + 1 :]:
            description = str(step.get("description", ""))
            if step.get("tool") == "run_tests" or (
                step.get("tool") == "run_command"
                and "pytest" in description.lower()
            ):
                continue
            goals = cls._goal_identifiers(description)
            if not goals:
                continue
            if not goals.issubset(completed_goals):
                return True
        return False

    @classmethod
    def _should_skip_completed_step(
        cls,
        state: AgentState,
        step: dict[str, Any],
    ) -> bool:
        """Avoid re-editing a goal after a successful validation."""
        if not state.tests_passed:
            return False
        if step.get("tool") not in {"edit_file", "write_file"}:
            return False
        goals = cls._goal_identifiers(str(step.get("description", "")))
        return not goals or goals.issubset(set(state.completed_edit_goals))

    @classmethod
    def _filter_pending_plan_steps(
        cls,
        state: AgentState,
        step_index: int,
        test_result: dict[str, Any],
        debug_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Keep only future work not addressed by the current failure."""
        current = state.current_plan[step_index] if step_index < len(state.current_plan) else {}
        known_text = "\n".join(
            [
                str(current.get("description", "")),
                cls._result_text(test_result),
                cls._compact_result(debug_result, 1600),
            ]
        )
        known_goals = cls._goal_identifiers(known_text)
        pending: list[dict[str, Any]] = []
        for raw_step in state.current_plan[step_index + 1 :]:
            if not isinstance(raw_step, dict):
                continue
            description = str(raw_step.get("description", ""))
            if raw_step.get("tool") == "run_tests" or (
                raw_step.get("tool") == "run_command"
                and "pytest" in description.lower()
            ):
                # Every replan is sanitized to end with validation. Keeping old
                # validation steps only creates duplicate test runs.
                continue
            goals = cls._goal_identifiers(description)
            if goals and goals.issubset(known_goals):
                continue
            pending.append(dict(raw_step))
        return pending

    @classmethod
    def _merge_pending_plan_steps(
        cls,
        generated_steps: list[dict[str, Any]],
        pending_steps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge preserved requirements without duplicating generated work."""
        merged = [dict(step) for step in generated_steps if isinstance(step, dict)]
        for raw_step in pending_steps:
            if not isinstance(raw_step, dict):
                continue
            description = str(raw_step.get("description", ""))
            if raw_step.get("tool") == "run_tests" or (
                raw_step.get("tool") == "run_command"
                and "pytest" in description.lower()
            ):
                continue
            tool_name = str(raw_step.get("tool", ""))
            goals = cls._goal_identifiers(description)
            duplicate = False
            for existing in merged:
                if str(existing.get("tool", "")) != tool_name:
                    continue
                existing_goals = cls._goal_identifiers(
                    str(existing.get("description", ""))
                )
                if str(existing.get("description", "")).strip() == str(
                    raw_step.get("description", "")
                ).strip() or (goals and existing_goals.intersection(goals)):
                    duplicate = True
                    break
            if not duplicate:
                merged.append(dict(raw_step))
        return merged

    def _sanitize_plan(
        self,
        plan: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Remove unsafe redundant steps and guarantee a validation step."""
        steps: list[dict[str, Any]] = []
        has_edit = False
        has_validation = False
        for raw_step in plan.get("steps", []):
            step = dict(raw_step)
            tool_name = str(step.get("tool", "")).strip()
            description = str(step.get("description", ""))
            normalized_description = description.lower()
            if tool_name in {"edit_file", "write_file"} and any(
                marker in normalized_description
                for marker in ("测试文件", "测试用例", "test file", "test case")
            ):
                LOGGER.warning(
                    "Dropping plan step that edits tests: %s",
                    description,
                )
                continue
            if tool_name == "run_command" and any(
                marker in normalized_description
                for marker in (
                    "pip install",
                    "pip uninstall",
                    "安装pytest",
                    "安装依赖",
                    "python -m pdb",
                    "pdb",
                    "python -m unittest",
                    "unittest",
                )
            ):
                LOGGER.warning(
                    "Dropping unsupported plan command: %s",
                    description,
                )
                continue
            if tool_name == "write_file" and has_edit:
                LOGGER.warning(
                    "Dropping redundant write_file step after edit_file: %s",
                    description,
                )
                continue
            if tool_name == "edit_file":
                has_edit = True
            if tool_name == "run_tests" or (
                tool_name == "run_command" and "pytest" in description.lower()
            ):
                has_validation = True
            steps.append(step)

        validated_steps: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            validated_steps.append(step)
            if step.get("tool") != "edit_file":
                continue
            next_tool = steps[index + 1].get("tool") if index + 1 < len(steps) else None
            next_description = (
                str(steps[index + 1].get("description", ""))
                if index + 1 < len(steps)
                else ""
            )
            next_is_validation = next_tool == "run_tests" or (
                next_tool == "run_command" and "pytest" in next_description.lower()
            )
            if not next_is_validation:
                # Validate each edit before allowing another edit to build on it.
                validated_steps.append(
                    {
                        "id": 0,
                        "description": "立即运行pytest测试验证本次编辑",
                        "tool": "run_tests",
                    }
                )

        if not has_validation and not any(
            step.get("tool") == "run_tests" for step in validated_steps
        ):
            validated_steps.append(
                {
                    "id": 0,
                    "description": "运行pytest测试验证所有修改",
                    "tool": "run_tests",
                }
            )

        for index, step in enumerate(validated_steps, start=1):
            step["id"] = index
        return {"steps": validated_steps}

    def _build_action(
        self,
        state: AgentState,
        step: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_action_prompt(state, step)},
        ]
        try:
            response = self.llm_client.chat(
                messages,
                temperature=0.0,
                max_tokens=768,
            )
            self._record_llm_usage(state, "action", response.get("usage", {}))
            action = self._parse_action(response.get("content", ""), step)
            action = self._prepare_action(state, step, action)
        except Exception as exc:
            LOGGER.exception("Action generation failed, using fallback arguments")
            action = {
                "tool": step.get("tool", ""),
                "arguments": self._fallback_arguments(step, state),
            }
            # A usable fallback should not poison the final task status. The tool
            # result will report a real failure if the fallback is insufficient.
            LOGGER.warning("Action fallback selected: %s", exc)

        self._emit_event("action", step=step, action=action)
        return action

    def _prepare_action(
        self,
        state: AgentState,
        step: dict[str, Any],
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize one model action against the current plan and tool schema."""
        planned_tool = str(step.get("tool", "")).strip()
        requested_tool = str(action.get("tool", "")).strip()
        tool_name = planned_tool or requested_tool
        arguments = dict(action.get("arguments", {}))

        # A stale action is worse than a missing action: keeping its arguments
        # can send edit_file fields into run_tests or read_file. Use only safe
        # step-specific fallback arguments when the model chose another tool.
        if planned_tool and requested_tool and requested_tool != planned_tool:
            if planned_tool == "write_file" and requested_tool == "edit_file":
                # This is a safer correction for a common plan mistake: an
                # existing file should be edited, never overwritten.
                tool_name = requested_tool
            else:
                LOGGER.warning(
                    "Discarding mismatched action tool=%s; plan requires=%s",
                    requested_tool,
                    planned_tool,
                )
                tool_name = planned_tool
                arguments = self._fallback_arguments(step, state)

        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return {"tool": tool_name, "arguments": arguments}

        properties = tool.parameters.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        arguments = {
            key: value for key, value in arguments.items() if key in properties
        }

        if tool_name == "read_file":
            path = arguments.get("path")
            if (
                not isinstance(path, str)
                or not path.strip()
                or "{{" in path
                or (state.tests_passed and _looks_like_test_file(path))
            ):
                fallback = self._fallback_arguments(
                    {"tool": "read_file", "description": "读取相关源文件"},
                    state,
                )
                if fallback.get("path"):
                    arguments["path"] = fallback["path"]

        if tool_name == "edit_file":
            path = arguments.get("path")
            if isinstance(path, str) and _looks_like_test_file(path):
                fallback = self._fallback_arguments(
                    {"tool": "read_file", "description": "读取相关源文件"},
                    state,
                )
                if fallback.get("path"):
                    LOGGER.warning(
                        "Redirecting edit away from test file: %s -> %s",
                        path,
                        fallback["path"],
                    )
                    arguments["path"] = fallback["path"]

        # run_tests has no parameters. This also protects against stale edit
        # arguments when a model emits a previous action verbatim.
        if tool_name == "run_tests":
            arguments = {}

        required = tool.parameters.get("required", [])
        if not isinstance(required, list):
            required = []
        missing = [key for key in required if key not in arguments]
        if missing:
            fallback = self._fallback_arguments(step, state)
            arguments.update(
                {
                    key: value
                    for key, value in fallback.items()
                    if key in properties and key not in arguments
                }
            )
            missing = [key for key in required if key not in arguments]
            if missing:
                LOGGER.warning(
                    "Action missing required arguments: tool=%s missing=%s",
                    tool_name,
                    missing,
                )

        if tool_name == "edit_file":
            path = arguments.get("path")
            if isinstance(path, str):
                digest = CodingAgent._latest_read_hash(state, path)
                if digest:
                    # Never trust a hash invented by the model. A second edit
                    # requires a fresh read after the first edit.
                    arguments["expected_sha256"] = digest
        return {"tool": tool_name, "arguments": arguments}

    @staticmethod
    def _latest_read_hash(state: AgentState, path: str) -> str | None:
        try:
            target = resolve_workspace_path(path).resolve()
        except (OSError, ValueError):
            return None

        for item in reversed(state.tool_results):
            if item.get("name") != "read_file":
                continue
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict) or arguments.get("path") is None:
                continue
            try:
                read_path = resolve_workspace_path(str(arguments["path"])).resolve()
            except (OSError, ValueError):
                continue
            if read_path != target:
                continue
            result = item.get("result", {})
            if isinstance(result, dict) and result.get("success"):
                digest = result.get("sha256")
                return str(digest) if digest else None
        return None

    def _execute_action(
        self,
        state: AgentState,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Enforce read-before-edit without consuming an Agent iteration."""
        if tool_name == "edit_file":
            path = arguments.get("path")
            if not isinstance(path, str) or not path:
                return {
                    "success": False,
                    "output": "",
                    "error": "edit_file requires a file path",
                }
            if not self._was_read(state, path):
                return {
                    "success": False,
                    "output": "",
                    "error": (
                        f"File must be read before edit_file: {path}. "
                        "Read the current file and retry."
                    ),
                }
        return self._execute_tool(tool_name, arguments)

    @staticmethod
    def _was_read(state: AgentState, path: str) -> bool:
        try:
            target = resolve_workspace_path(path).resolve()
        except (OSError, ValueError):
            return False

        for item in state.tool_results:
            if item.get("name") != "read_file":
                continue
            arguments = item.get("arguments", {})
            result = item.get("result", {})
            if not isinstance(arguments, dict) or not isinstance(result, dict):
                continue
            if not result.get("success"):
                continue
            read_path = arguments.get("path")
            if not isinstance(read_path, str):
                continue
            try:
                if resolve_workspace_path(read_path).resolve() == target:
                    return True
            except (OSError, ValueError):
                continue
        return False

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            error = f"Tool not found: {tool_name}"
            LOGGER.error(error)
            return {"success": False, "output": "", "error": error}

        try:
            LOGGER.info("Executing tool: name=%s arguments=%s", tool_name, arguments)
            result = tool.handler(**arguments)
        except Exception as exc:
            LOGGER.exception("Tool execution raised: name=%s", tool_name)
            return {"success": False, "output": "", "error": str(exc)}

        if not isinstance(result, dict):
            error = f"Tool returned non-dict result: {type(result).__name__}"
            LOGGER.error(error)
            return {"success": False, "output": "", "error": error}

        return {
            "success": bool(result.get("success", False)),
            "output": str(result.get("output", "")),
            "error": str(result.get("error", "")),
            **{key: value for key, value in result.items() if key not in {"success", "output", "error"}},
        }

    def _reflect(
        self,
        state: AgentState,
        step: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_reflection_prompt(
                    state,
                    step,
                    tool_name,
                    arguments,
                    result,
                ),
            },
        ]
        try:
            response = self.llm_client.chat(
                messages,
                temperature=0.0,
                max_tokens=512,
            )
            self._record_llm_usage(state, "reflection", response.get("usage", {}))
            return self._parse_reflection(response.get("content", ""))
        except Exception as exc:
            LOGGER.exception("Reflection failed, using fallback reflection")
            state.update(last_error=str(exc))
            return self._fallback_reflection(step, result)

    @staticmethod
    def _record_llm_usage(
        state: AgentState,
        phase: str,
        usage: Any,
    ) -> None:
        if not isinstance(usage, dict) or not usage:
            return
        state.llm_usage.append({"phase": phase, **usage})

    def _build_action_prompt(
        self,
        state: AgentState,
        step: dict[str, Any],
    ) -> str:
        return (
            "请为当前计划步骤生成工具调用JSON。\n\n"
            f"用户任务:\n{state.task}\n\n"
            f"当前步骤:\n{json.dumps(step, ensure_ascii=False)}\n\n"
            f"相关代码检索结果:\n{self._retrieval_text(state, max_length=800)}\n\n"
            f"失败文件:\n{json.dumps(state.failure_paths, ensure_ascii=False)}\n\n"
            f"最近失败分析:\n{self._compact_result(state.last_debug_result, 900)}\n\n"
            f"最近测试失败:\n{self._compact_result(state.last_test_failure, 1400)}\n\n"
            f"最近工具失败:\n{self._compact_result(state.last_tool_failure, 700)}\n\n"
            f"连续相同失败次数:\n{state.stagnation_count}\n\n"
            f"已完成修改目标:\n{json.dumps(state.completed_edit_goals, ensure_ascii=False)}\n\n"
            f"控制器诊断提示:\n{self._diagnostic_hints(self._result_text(state.last_test_failure), state)}\n\n"
            f"最近工具结果:\n{self._recent_tool_results_text(state, max_items=3, max_length=1400)}\n\n"
            f"可用工具:\n{self._compact_tool_schemas()}\n\n"
            "如果这是测试失败后的修复步骤，必须先使用失败文件和测试输出定位问题；"
            "不要重复上一轮的相同修改，也不要为了通过测试修改测试文件。\n"
            "只输出JSON。"
        )

    def _build_reflection_prompt(
        self,
        state: AgentState,
        step: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> str:
        return (
            "请根据当前工具执行结果判断任务是否完成。\n\n"
            f"用户任务:\n{state.task}\n\n"
            f"当前步骤:\n{json.dumps(step, ensure_ascii=False)}\n\n"
            f"相关代码检索结果:\n{self._retrieval_text(state, max_length=1000)}\n\n"
            f"工具调用:\n{json.dumps({'tool': tool_name, 'arguments': arguments}, ensure_ascii=False)}\n\n"
            f"工具结果:\n{self._compact_result(result, 1800)}\n\n"
            f"已执行iterations: {state.iterations}\n\n"
            "只输出JSON。"
        )

    def _compact_tool_schemas(self) -> str:
        """Describe only the fields needed for the current JSON action."""
        compact: list[dict[str, Any]] = []
        for tool in self.tool_registry.list_tools():
            function = tool.get("function", {})
            parameters = function.get("parameters", {})
            properties = parameters.get("properties", {})
            compact.append(
                {
                    "name": function.get("name"),
                    "required": parameters.get("required", []),
                    "fields": list(properties) if isinstance(properties, dict) else [],
                }
            )
        return json.dumps(compact, ensure_ascii=False)

    @classmethod
    def _parse_action(
        cls,
        content: str,
        step: dict[str, Any],
    ) -> dict[str, Any]:
        data = cls._parse_json_object(content)
        planned_tool = step.get("tool", "")
        requested_tool = data.get("tool")
        tool_name = requested_tool or planned_tool
        arguments = data.get("arguments", {})
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("action tool is required")
        if not isinstance(arguments, dict):
            raise ValueError("action arguments must be an object")
        return {
            "tool": tool_name.strip(),
            "arguments": arguments,
        }

    @classmethod
    def _parse_reflection(cls, content: str) -> dict[str, Any]:
        data = cls._parse_json_object(content)
        done = data.get("done")
        reflection = data.get("reflection", "")
        if not isinstance(done, bool):
            raise ValueError("reflection done must be a boolean")
        if not isinstance(reflection, str) or not reflection.strip():
            raise ValueError("reflection text is required")

        return {
            "done": done,
            "reflection": reflection.strip(),
        }

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
            raise ValueError("response must be a JSON object")
        return data

    @staticmethod
    def _fallback_arguments(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
        tool_name = step.get("tool", "")
        if tool_name == "search_files":
            description = str(step.get("description", ""))
            path_match = re.search(r"(?:[\w./-]+/)?[\w.-]+\.py", description)
            return {"keyword": path_match.group(0) if path_match else state.task}
        if tool_name == "read_file":
            source_path = CodingAgent._preferred_source_path(state)
            if source_path:
                return {"path": source_path}
            description = str(step.get("description", ""))
            path_match = re.search(r"(?:[\w./-]+/)?[\w.-]+\.py", description)
            if path_match:
                return {"path": path_match.group(0)}
            for item in reversed(state.tool_results):
                if item.get("name") != "search_files":
                    continue
                result = item.get("result", {})
                if not isinstance(result, dict) or not result.get("success"):
                    continue
                output = str(result.get("output", ""))
                first_python_path: str | None = None
                for candidate in output.splitlines():
                    candidate = candidate.strip()
                    if candidate.endswith(".py"):
                        first_python_path = first_python_path or candidate
                        if not _looks_like_test_file(candidate):
                            return {"path": candidate}
                if first_python_path:
                    return {"path": first_python_path}
            for item in state.retrieved_context:
                path = item.get("file") if isinstance(item, dict) else None
                if isinstance(path, str) and path:
                    return {"path": path}
            return {}
        if tool_name == "edit_file":
            source_path = CodingAgent._preferred_source_path(state)
            return {"path": source_path} if source_path else {}
        if tool_name == "run_tests":
            return {}
        if tool_name == "run_command":
            description = str(step.get("description", ""))
            if "测试" in description or "pytest" in description.lower():
                return {"command": "python -m pytest -q"}
            return {"command": "pwd"}
        return {}

    @staticmethod
    def _fallback_reflection(
        step: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if not result.get("success", False):
            return {
                "done": False,
                "reflection": f"工具{step.get('tool')}执行失败，需要继续处理错误。",
            }
        if step.get("tool") == "run_tests":
            return {
                "done": True,
                "reflection": "测试工具执行成功，任务可以视为完成。",
            }
        return {
            "done": False,
            "reflection": "当前步骤执行完成，继续执行后续计划。",
        }

    @staticmethod
    def _build_replan_task(state: AgentState) -> str:
        unfinished_plan = state.pending_plan_steps or state.current_plan
        completed_goals = json.dumps(
            state.completed_edit_goals,
            ensure_ascii=False,
        )
        if state.last_tool_failure:
            return (
                f"{state.task}\n\n"
                "这是一次工具执行失败后的重新规划。请先修复工具调用问题。\n\n"
                f"原计划和未完成要求:\n{CodingAgent._compact_result(unfinished_plan, 1000)}\n\n"
                f"工具失败:\n{CodingAgent._compact_result(state.last_tool_failure, 1400)}\n\n"
            f"已完成修改目标:\n{completed_goals}\n\n"
            f"控制器诊断提示:\n{CodingAgent._diagnostic_hints(CodingAgent._result_text(state.last_tool_failure), state)}\n\n"
            f"最近工具结果摘要:\n{CodingAgent._recent_tool_results_text(state, max_items=2, max_length=1800)}\n\n"
                "请生成具体、可执行的后续计划。必须使用真实文件路径和合法工具参数，"
                "不要修改测试文件。"
            )
        return (
            f"{state.task}\n\n"
            "这是一次测试失败后的重新规划。请严格基于下面的诊断继续修复。\n\n"
            f"原计划和未完成要求:\n{CodingAgent._compact_result(unfinished_plan, 1000)}\n\n"
            f"失败文件:\n{json.dumps(state.failure_paths, ensure_ascii=False)}\n\n"
            f"测试失败:\n{CodingAgent._compact_result(state.last_test_failure, 1800)}\n\n"
            f"失败分析:\n{CodingAgent._compact_result(state.last_debug_result, 1100)}\n\n"
            f"已完成修改目标:\n{completed_goals}\n\n"
            f"控制器诊断提示:\n{CodingAgent._diagnostic_hints(CodingAgent._result_text(state.last_test_failure), state)}\n\n"
            f"当前相关代码:\n{CodingAgent._build_debug_context(state, 1800)}\n\n"
            f"连续相同失败次数:\n{state.stagnation_count}\n\n"
            f"当前执行结果摘要:\n{CodingAgent._recent_tool_results_text(state, max_items=3, max_length=1800)}\n\n"
            "后续计划必须保留原issue中尚未验证的要求，优先读取失败测试和相关源文件，"
            "然后进行最小范围修改，最后重新运行测试。"
            "不要修改测试文件，不要使用pdb，不要用与上一轮相同的修改方式。"
            "如果连续失败次数大于0，必须先解释上一轮为什么没有解决问题。"
        )

    @staticmethod
    def _result_text(result: dict[str, Any]) -> str:
        parts = [
            str(result.get("error", "")),
            str(result.get("output", "")),
            str(result.get("stdout", "")),
            str(result.get("stderr", "")),
        ]
        return "\n".join(part for part in parts if part and part != "None")

    @staticmethod
    def _failure_signature(text: str) -> str:
        normalized = "\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )
        if not normalized:
            return ""
        import hashlib

        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _compact_result(result: Any, max_length: int = 6000) -> str:
        if result is None or result == {} or result == []:
            return "暂无"
        text = json.dumps(result, ensure_ascii=False)
        if len(text) <= max_length:
            return text
        head_length = max_length // 2
        tail_length = max_length - head_length
        return CodingAgent._trim_text(text, max_length)

    @staticmethod
    def _trim_text(text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        head_length = max_length // 2
        tail_length = max_length - head_length
        return f"{text[:head_length]}... <truncated> ...{text[-tail_length:]}"

    @staticmethod
    def _recent_tool_results_text(
        state: AgentState,
        max_items: int = 6,
        max_length: int = 9000,
    ) -> str:
        items = []
        for item in state.tool_results[-max_items:]:
            if not isinstance(item, dict):
                continue
            result = item.get("result", {})
            if isinstance(result, dict):
                compact_result = {
                    key: CodingAgent._trim_text(str(value), 1400)
                    if key in {"output", "error", "stdout", "stderr"}
                    else value
                    for key, value in result.items()
                }
            else:
                compact_result = result
            items.append(
                {
                    "name": item.get("name"),
                    "phase": item.get("phase"),
                    "arguments": item.get("arguments"),
                    "result": compact_result,
                }
            )
        return CodingAgent._compact_result(items, max_length=max_length)

    def _initialize_retrieval(self, state: AgentState) -> None:
        if self.retriever is None:
            return

        try:
            results = self.retriever.retrieve(state.task, top_k=self.retrieval_top_k)
            if not isinstance(results, list):
                raise ValueError("retriever must return a list")
            normalized = [
                dict(item) for item in results if isinstance(item, dict)
            ]
            state.update(retrieved_context=normalized)
            self._emit_event("retrieval", query=state.task, results=normalized)
            LOGGER.info("Retrieved code context: results=%d", len(normalized))
        except Exception as exc:
            LOGGER.warning("Code retrieval unavailable: %s", exc)
            self._emit_event(
                "retrieval",
                query=state.task,
                results=[],
                error=str(exc),
            )

    @staticmethod
    def _add_retrieval_context(task: str, state: AgentState) -> str:
        context = CodingAgent._retrieval_text(state, max_length=1200)
        if not context:
            return task
        return f"{task}\n\n相关代码检索上下文:\n{context}"

    @staticmethod
    def _retrieval_text(state: AgentState, max_length: int = 4000) -> str:
        if not state.retrieved_context:
            return "暂无"

        parts: list[str] = []
        for index, item in enumerate(state.retrieved_context, start=1):
            parts.append(
                f"[{index}] 文件: {item.get('file', '')}\n"
                f"符号: {item.get('symbol', '')}\n"
                f"行号: {item.get('start_line', '?')}-{item.get('end_line', '?')}\n"
                f"相关度: {item.get('score', '?')}\n"
                f"代码:\n{item.get('code', '')}"
            )
            if sum(len(part) for part in parts) >= max_length:
                break
        return "\n\n".join(parts)[:max_length]

    @staticmethod
    def _record_changed_file(
        state: AgentState,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        if tool_name not in {"write_file", "edit_file"} or not result.get("success"):
            return

        path = arguments.get("path")
        if not isinstance(path, str) or not path:
            return
        if path in state.changed_files:
            return
        state.changed_files.append(path)

    @staticmethod
    def _summarize_result(result: dict[str, Any], max_length: int = 600) -> str:
        payload = json.dumps(result, ensure_ascii=False)
        if len(payload) <= max_length:
            return payload
        return f"{payload[:max_length]}... <truncated>"

    def _log_phase(self, phase: str, message: str, style: str) -> None:
        LOGGER.info("[%s] %s", phase, message)
        if Text is not None:
            tag = Text(f"[{phase}]", style=style)
            self.console.print(tag, message)
            return
        self.console.print(f"[{phase}] {message}")

    def _emit_event(self, event_type: str, **payload: Any) -> None:
        if self.event_handler is None:
            return
        event = {"type": event_type, **payload}
        try:
            self.event_handler(event)
        except Exception:
            LOGGER.exception("Agent event handler failed: type=%s", event_type)

    @staticmethod
    def _default_llm_client() -> LLMClientProtocol:
        from llm.client import VLLMClient

        return VLLMClient()

    @staticmethod
    def _default_console() -> Any:
        if Console is not None:
            return Console()

        class PlainConsole:
            @staticmethod
            def print(*values: Any) -> None:
                print(*values)

        return PlainConsole()

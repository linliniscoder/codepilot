from __future__ import annotations

import html
import json
import logging
import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from main import (
    CONFIG_PATH,
    collect_diff,
    extract_test_result,
    get_console,
    initialize_llm_client,
    initialize_logging,
    initialize_retriever,
    initialize_tools,
    load_config,
    resolve_workspace,
    snapshot_workspace,
)
from agent.controller import CodingAgent
from llm.client import LLMTimeoutError, ServerUnavailableError, VLLMClientError


LOGGER = logging.getLogger("codepilot.web")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


class CodePilotWebApp:
    def __init__(self, with_retriever: bool = False) -> None:
        self.task = ""
        self.last_state: Any | None = None
        self.last_diff = ""
        self.last_test_result: dict[str, Any] | None = None
        self.last_error = ""
        self.events: list[dict[str, Any]] = []
        self.running = False
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.config = load_config(CONFIG_PATH)
        self.workspace = resolve_workspace(self.config)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.console = get_console()
        self.tool_registry = initialize_tools()
        self.llm_client = initialize_llm_client(self.config)
        self.retriever = (
            initialize_retriever(self.config, self.console) if with_retriever else None
        )

    def run_task(self, task: str, max_iterations: int = 12) -> None:
        self.events = []
        self.last_error = ""
        self.last_state = None
        self.last_diff = ""
        self.last_test_result = None
        self.running = True
        self.started_at = time.perf_counter()
        self.finished_at = None
        before_snapshot = snapshot_workspace(self.workspace)
        agent = CodingAgent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            max_iterations=max_iterations,
            console=self.console,
            event_handler=self.events.append,
            retriever=self.retriever,
        )
        try:
            state = agent.run(task)
            self.last_state = state
            self.last_diff = collect_diff(
                self.workspace,
                state.changed_files,
                before_snapshot,
            )
            self.last_test_result = extract_test_result(state)
            self.last_error = state.last_error or ""
        finally:
            self.running = False
            self.finished_at = time.perf_counter()


def render_page(app: CodePilotWebApp) -> str:
    state = app.last_state
    changed_files = getattr(state, "changed_files", []) if state else []
    tool_results = getattr(state, "tool_results", []) if state else []
    history = getattr(state, "history", []) if state else []
    plan = getattr(state, "current_plan", []) if state else []
    llm_usage = getattr(state, "llm_usage", []) if state else []
    failure_count = getattr(state, "failure_count", 0) if state else 0
    stagnation_count = getattr(state, "stagnation_count", 0) if state else 0
    failure_paths = getattr(state, "failure_paths", []) if state else []
    termination_reason = getattr(state, "termination_reason", None) if state else None
    token_cost = sum(
        int(item.get("total_tokens", 0) or 0)
        for item in llm_usage
        if isinstance(item, dict)
    )
    task = html.escape(app.task)
    status = build_status(app)
    events_html = render_events(app.events)
    test_summary = render_test_summary(app.last_test_result)
    changed_html = render_changed_files(changed_files)
    plan_html = render_plan(plan)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CodePilot</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #16202a;
      --muted: #667085;
      --line: #d7dde6;
      --accent: #0f766e;
      --accent-2: #1d4ed8;
      --danger: #b42318;
      --warn: #b54708;
      --code: #101828;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .app {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }}
    .form {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 120px;
      gap: 12px;
      margin-bottom: 20px;
    }}
    textarea {{
      width: 100%;
      min-height: 92px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      resize: vertical;
      font: inherit;
      background: var(--panel);
    }}
    button {{
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      background: var(--accent-2);
      color: white;
      font-weight: 600;
      cursor: pointer;
      min-width: 120px;
      height: 92px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 15px;
    }}
    .stack {{
      display: grid;
      gap: 16px;
    }}
    .meta {{
      display: grid;
      gap: 10px;
      font-size: 13px;
    }}
    .label {{
      color: var(--muted);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      padding: 2px 10px;
      background: #e6f4f1;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }}
    .badge.error {{
      background: #fee4e2;
      color: var(--danger);
    }}
    .steps {{
      display: grid;
      gap: 10px;
    }}
    .step {{
      border-left: 3px solid var(--line);
      padding-left: 10px;
    }}
    .step-title {{
      font-weight: 700;
      font-size: 13px;
    }}
    .step-meta {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }}
    .timeline {{
      display: grid;
      gap: 12px;
    }}
    .event {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 12px;
    }}
    .event-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      font-size: 13px;
      font-weight: 700;
    }}
    .event-kind {{
      color: var(--accent-2);
    }}
    .event-tool {{
      color: var(--muted);
      font-weight: 600;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.5;
      color: var(--code);
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .error {{
      color: var(--danger);
    }}
    .warn {{
      color: var(--warn);
    }}
    .ok {{
      color: var(--accent);
    }}
    @media (max-width: 960px) {{
      .grid, .form {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <div>
        <h1>CodePilot</h1>
        <div class="subtitle">工作区：{html.escape(str(app.workspace))}</div>
      </div>
      <div>{status}</div>
    </div>
    <form class="form" method="post" action="/run">
      <textarea name="task" placeholder="输入任务，例如：修复 workspace 里的 bug">{task}</textarea>
      <button type="submit">运行</button>
    </form>
    <div class="grid">
      <div class="stack">
        <div class="panel">
          <h2>运行概览</h2>
          <div class="meta">
            <div><span class="label">当前任务</span><br />{task or "还没有运行任务"}</div>
            <div><span class="label">修改文件</span><br />{changed_html}</div>
            <div><span class="label">最近错误</span><br /><span class="error">{html.escape(app.last_error or "无")}</span></div>
            <div><span class="label">失败文件</span><br />{render_changed_files(failure_paths)}</div>
            <div><span class="label">测试结果</span><br />{test_summary}</div>
          </div>
        </div>
        <div class="panel">
          <h2>计划步骤</h2>
          {plan_html}
        </div>
        <div class="panel">
          <h2>统计</h2>
          <div class="meta">
            <div><span class="label">计划步骤数</span><br />{len(plan)}</div>
            <div><span class="label">工具调用数</span><br />{len(tool_results)}</div>
            <div><span class="label">历史消息数</span><br />{len(history)}</div>
            <div><span class="label">测试失败次数</span><br />{failure_count}</div>
            <div><span class="label">连续相同失败</span><br />{stagnation_count}</div>
            <div><span class="label">Token 消耗</span><br />{token_cost}</div>
            <div><span class="label">终止原因</span><br />{html.escape(str(termination_reason or "暂无"))}</div>
          </div>
        </div>
      </div>
      <div class="stack">
        <div class="panel">
          <h2>执行过程</h2>
          <div class="timeline">{events_html}</div>
        </div>
        <div class="panel">
          <h2>代码差异</h2>
          <pre>{html.escape(app.last_diff or "暂无 diff")}</pre>
        </div>
        <div class="panel">
          <h2>完整测试输出</h2>
          <pre>{html.escape(json.dumps(app.last_test_result, ensure_ascii=False, indent=2) if app.last_test_result else "暂无测试结果")}</pre>
        </div>
      </div>
    </div>
  </div>
</body>
</html>"""


def build_status(app: CodePilotWebApp) -> str:
    if app.running:
        return '<span class="badge">运行中</span>'
    if app.last_error:
        return '<span class="badge error">有错误</span>'
    if app.last_state:
        return '<span class="badge">已完成</span>'
    return '<span class="badge">待运行</span>'


def render_changed_files(changed_files: list[str]) -> str:
    if not changed_files:
        return "无"
    return "<br />".join(html.escape(str(path)) for path in changed_files)


def render_test_summary(test_result: dict[str, Any] | None) -> str:
    if not test_result:
        return "暂无"
    if test_result.get("success"):
        return '<span class="ok">通过</span>'
    return f'<span class="error">失败：{html.escape(str(test_result.get("error") or "测试未通过"))}</span>'


def render_plan(plan: list[dict[str, Any]]) -> str:
    if not plan:
        return '<div class="label">暂无计划</div>'
    items = []
    for step in plan:
        items.append(
            '<div class="step">'
            f'<div class="step-title">#{html.escape(str(step.get("id", "")))} '
            f'{html.escape(str(step.get("description", "")))}</div>'
            f'<div class="step-meta">工具：{html.escape(str(step.get("tool", "")))}</div>'
            '</div>'
        )
    return f'<div class="steps">{"".join(items)}</div>'


def render_events(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<div class="label">还没有执行记录。提交任务后，这里会显示计划、工具调用、结果和反思。</div>'
    return "".join(render_event(event, index + 1) for index, event in enumerate(events))


def render_event(event: dict[str, Any], index: int) -> str:
    event_type = event.get("type", "")
    title_map = {
        "plan": "生成计划",
        "retrieval": "代码检索",
        "action": "准备工具调用",
        "tool_result": "工具执行结果",
        "failure_context": "自动读取失败上下文",
        "reflection": "反思判断",
        "debug_reflection": "失败分析",
    }
    title = title_map.get(str(event_type), str(event_type))
    step = event.get("step") if isinstance(event.get("step"), dict) else {}
    step_label = (
        f'步骤 #{html.escape(str(step.get("id")))}'
        if step
        else ""
    )

    if event_type == "plan":
        body = render_plan(event.get("plan", {}).get("steps", []))
    elif event_type == "retrieval":
        results = event.get("results", [])
        body = (
            f'<div class="label">查询</div><pre>{html.escape(str(event.get("query", "")))}</pre>'
            f'<div class="label">结果数</div><pre>{len(results)}</pre>'
            f'<div class="label">结果</div><pre>{format_json(results)}</pre>'
        )
    elif event_type == "debug_reflection":
        analysis = event.get("analysis", {})
        body = (
            f'<div class="label">原因分析</div><pre>{html.escape(str(analysis.get("analysis", "")))}</pre>'
            f'<div class="label">修复建议</div><pre>{html.escape(str(analysis.get("suggestion", "")))}</pre>'
            f'<div class="label">下一步</div><pre>{html.escape(str(analysis.get("next_action", "")))}</pre>'
        )
    elif event_type == "action":
        action = event.get("action", {})
        body = (
            f'<div class="label">工具</div><pre>{html.escape(str(action.get("tool", "")))}</pre>'
            f'<div class="label">参数</div><pre>{format_json(action.get("arguments", {}))}</pre>'
        )
    elif event_type == "tool_result":
        result = event.get("result", {})
        ok = "成功" if result.get("success") else "失败"
        body = (
            f'<div class="label">工具</div><pre>{html.escape(str(event.get("tool", "")))}</pre>'
            f'<div class="label">参数</div><pre>{format_json(event.get("arguments", {}))}</pre>'
            f'<div class="label">状态</div><pre>{html.escape(ok)}</pre>'
            f'<div class="label">输出</div><pre>{html.escape(shorten(str(result.get("output", ""))))}</pre>'
            f'<div class="label">错误</div><pre>{html.escape(shorten(str(result.get("error", ""))))}</pre>'
        )
    elif event_type == "failure_context":
        result = event.get("result", {})
        ok = "成功" if result.get("success") else "失败"
        body = (
            f'<div class="label">文件</div><pre>{html.escape(str(event.get("path", "")))}</pre>'
            f'<div class="label">状态</div><pre>{html.escape(ok)}</pre>'
            f'<div class="label">内容</div><pre>{html.escape(shorten(str(result.get("output", ""))))}</pre>'
            f'<div class="label">错误</div><pre>{html.escape(shorten(str(result.get("error", ""))))}</pre>'
        )
    elif event_type == "reflection":
        reflection = event.get("reflection", {})
        body = (
            f'<div class="label">是否完成</div><pre>{html.escape(str(reflection.get("done", False)))}</pre>'
            f'<div class="label">说明</div><pre>{html.escape(str(reflection.get("reflection", "")))}</pre>'
        )
    else:
        body = f"<pre>{format_json(event)}</pre>"

    return (
        '<div class="event">'
        '<div class="event-head">'
        f'<span class="event-kind">{index}. {html.escape(title)}</span>'
        f'<span class="event-tool">{step_label}</span>'
        '</div>'
        f'{body}'
        '</div>'
    )


def format_json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def shorten(value: str, limit: int = 1600) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n... 已截断 ..."


class Handler(BaseHTTPRequestHandler):
    server_version = "CodePilotWeb/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        app: CodePilotWebApp = self.server.app  # type: ignore[attr-defined]
        page = render_page(app)
        self._send_html(page)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/run":
            self.send_error(404)
            return

        app: CodePilotWebApp = self.server.app  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(body)
        task = form.get("task", [""])[0].strip()
        if task:
            app.task = task
            try:
                app.run_task(task)
            except Exception as exc:
                LOGGER.exception("Task run failed")
                app.last_error = friendly_error(exc)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        LOGGER.info(format, *args)

    def _send_html(self, html_text: str) -> None:
        payload = html_text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, with_retriever: bool = False) -> None:
    initialize_logging()
    app = CodePilotWebApp(with_retriever=with_retriever)
    server = ThreadingHTTPServer((host, port), Handler)
    server.app = app  # type: ignore[attr-defined]
    LOGGER.info("Serving CodePilot UI on http://%s:%s", host, port)
    print(f"CodePilot UI running at http://{host}:{port}")
    server.serve_forever()


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, ServerUnavailableError):
        return (
            "vLLM 服务暂时不可用。请检查 vLLM 终端日志；如果看到 502，通常表示模型服务端请求失败、"
            "模型还在忙、显存/worker 异常，或服务刚启动还没准备好。"
        )
    if isinstance(exc, LLMTimeoutError):
        return "LLM 请求超时。可以稍后重试，或减少任务复杂度/max_tokens。"
    if isinstance(exc, VLLMClientError):
        return f"LLM 调用失败：{exc}"
    return str(exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodePilot local web UI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--with-retriever",
        action="store_true",
        help="Initialize the retrieval module on startup.",
    )
    args = parser.parse_args()
    serve(host=args.host, port=args.port, with_retriever=args.with_retriever)

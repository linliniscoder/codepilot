from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from agent.controller import CodingAgent
from tools.registry import create_default_registry


LOGGER = logging.getLogger("codepilot")
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SNAPSHOT_MAX_BYTES = 1_000_000


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the project configuration from YAML."""
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_yaml(config_path)

    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def initialize_logging() -> None:
    """Initialize the default application logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodePilot repo-level coding agent")
    parser.add_argument("--task", help="User task or issue for CodePilot to solve.")
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=12,
        help="Maximum Agent loop iterations.",
    )
    parser.add_argument(
        "--with-retriever",
        action="store_true",
        help="Initialize the retrieval module before running the agent.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    initialize_logging()
    args = build_argument_parser().parse_args(argv)
    console = get_console()
    config_path = Path(args.config)
    config = load_config(config_path)
    workspace = resolve_workspace(config)
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["CODEPILOT_WORKSPACE"] = str(workspace)

    console.print("[bold cyan]CodePilot[/bold cyan] starting...")
    LOGGER.info("Configuration loaded from %s", config_path)
    LOGGER.info("Workspace: %s", workspace)

    if not args.task:
        console.print("No task provided. Use --task \"修复xxx bug\" to run the agent.")
        return 0

    llm_client = initialize_llm_client(config)
    tool_registry = initialize_tools()
    retriever = None
    if args.with_retriever:
        retriever = initialize_retriever(config, console)
    else:
        LOGGER.info("Retriever initialization skipped. Use --with-retriever to enable it.")

    before_snapshot = snapshot_workspace(workspace)
    agent = CodingAgent(
        llm_client=llm_client,
        tool_registry=tool_registry,
        max_iterations=args.max_iterations,
        console=console,
        retriever=retriever,
    )
    state = agent.run(args.task)

    diff = collect_diff(workspace, state.changed_files, before_snapshot)
    test_result = extract_test_result(state)
    render_final_output(console, state.changed_files, diff, test_result)
    return 0


def initialize_llm_client(config: dict[str, Any]) -> Any:
    """Configure the vLLM OpenAI-compatible client."""
    try:
        from llm.client import DEFAULT_BASE_URL, DEFAULT_MODEL, VLLMClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The OpenAI SDK is required to connect to vLLM. "
            "Install requirements.txt before running CodePilot."
        ) from exc

    base_url = config.get("vllm", {}).get("server_url", DEFAULT_BASE_URL)
    model = config.get("model", {}).get("name", DEFAULT_MODEL)
    context_window = config.get("vllm", {}).get("context_window", 4096)
    LOGGER.info("vLLM client configured: base_url=%s model=%s", base_url, model)
    return VLLMClient(
        base_url=base_url,
        model=model,
        context_window=int(context_window),
    )


def initialize_tools() -> Any:
    """Create the built-in tool registry."""
    registry = create_default_registry()
    LOGGER.info("Tools initialized: count=%d", len(registry.list_tools()))
    return registry


def initialize_retriever(config: dict[str, Any], console: Any) -> Any | None:
    """Initialize the code retriever when retrieval dependencies are available."""
    try:
        from retrieval import CodeEmbedder, CodeRetriever
        from retrieval.embedder import DEFAULT_EMBEDDING_MODEL
    except ModuleNotFoundError as exc:
        LOGGER.warning("Retriever disabled: %s", exc)
        console.print(f"[yellow]Retriever disabled:[/yellow] {exc}")
        return None

    paths = config.get("paths", {})
    retrieval_config = config.get("retrieval", {})
    embedding_model = resolve_embedding_model(
        paths.get("embedding_model"),
        DEFAULT_EMBEDDING_MODEL,
    )
    index_path = resolve_project_path(paths.get("faiss_index", "retrieval/index.faiss"))
    device = retrieval_config.get("device", "cpu")

    try:
        embedder = CodeEmbedder(model_name=embedding_model, device=device)
        if index_path.is_file():
            retriever = CodeRetriever.load(index_path, embedder=embedder)
            LOGGER.info("Retriever initialized from index: %s", index_path)
        else:
            retriever = CodeRetriever(embedder=embedder)
            chunks = retriever.build(resolve_workspace(config))
            retriever.save(index_path)
            LOGGER.info(
                "Retriever index built: path=%s chunks=%d",
                index_path,
                len(chunks),
            )
        return retriever
    except Exception as exc:
        LOGGER.warning("Retriever initialization failed: %s", exc)
        console.print(f"[yellow]Retriever unavailable:[/yellow] {exc}")
        return None


def resolve_workspace(config: dict[str, Any]) -> Path:
    workspace_value = config.get("paths", {}).get("workspace", "workspace")
    return resolve_project_path(workspace_value)


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_embedding_model(value: str | None, default_model: str) -> str:
    if not value:
        return default_model

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.exists():
        return str(candidate.resolve())
    if not str(value).startswith(".") and "/" in str(value):
        return str(value)
    return default_model


def snapshot_workspace(workspace: Path) -> dict[str, str]:
    """Read a lightweight text snapshot for post-run unified diffs."""
    snapshot: dict[str, str] = {}
    if not workspace.is_dir():
        return snapshot

    for path in workspace.rglob("*"):
        if not path.is_file() or should_skip_snapshot(path):
            continue
        try:
            if path.stat().st_size > SNAPSHOT_MAX_BYTES:
                continue
            relative = path.relative_to(workspace).as_posix()
            snapshot[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            LOGGER.debug("Skipping file snapshot: %s", path)
    return snapshot


def should_skip_snapshot(path: Path) -> bool:
    skipped_parts = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    return any(part in skipped_parts for part in path.parts)


def collect_diff(
    workspace: Path,
    changed_files: list[str],
    before_snapshot: dict[str, str],
) -> str:
    git_diff = collect_git_diff(workspace)
    if git_diff.strip():
        return git_diff
    return collect_snapshot_diff(workspace, changed_files, before_snapshot)


def collect_git_diff(workspace: Path) -> str:
    try:
        probe = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            return ""
        diff = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return diff.stdout if diff.returncode == 0 else ""


def collect_snapshot_diff(
    workspace: Path,
    changed_files: list[str],
    before_snapshot: dict[str, str],
) -> str:
    diff_parts: list[str] = []
    for relative in changed_files:
        try:
            path = (workspace / relative).resolve()
            path.relative_to(workspace)
        except ValueError:
            LOGGER.warning("Skipping diff for path outside workspace: %s", relative)
            continue

        before = before_snapshot.get(relative, "")
        try:
            after = path.read_text(encoding="utf-8") if path.is_file() else ""
        except (OSError, UnicodeError):
            LOGGER.warning("Skipping non-text diff for file: %s", relative)
            continue

        diff_parts.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(diff_parts)


def extract_test_result(state: Any) -> dict[str, Any] | None:
    for item in reversed(getattr(state, "tool_results", []) or []):
        name = item.get("name")
        result = item.get("result")
        arguments = item.get("arguments", {})
        if name == "run_tests" and isinstance(result, dict):
            return result
        if (
            name == "run_command"
            and isinstance(result, dict)
            and "pytest" in str(arguments.get("command", ""))
        ):
            return result
    return None


def render_final_output(
    console: Any,
    changed_files: list[str],
    diff: str,
    test_result: dict[str, Any] | None,
) -> None:
    console.rule("[bold cyan]CodePilot Result[/bold cyan]")
    console.print("[bold]Final modified files[/bold]")
    if changed_files:
        for path in changed_files:
            console.print(f"- {path}")
    else:
        console.print("(none recorded)")

    console.print("\n[bold]Diff[/bold]")
    if diff.strip():
        print_syntax(console, diff, "diff")
    else:
        console.print("No diff available.")

    console.print("\n[bold]Test result[/bold]")
    if test_result is None:
        console.print("No test result recorded.")
    else:
        console.print(json.dumps(test_result, ensure_ascii=False, indent=2))


def print_syntax(console: Any, content: str, lexer: str) -> None:
    try:
        from rich.syntax import Syntax
    except ModuleNotFoundError:
        console.print(content)
        return
    console.print(Syntax(content, lexer, word_wrap=True))


def get_console() -> Any:
    try:
        from rich.console import Console
    except ModuleNotFoundError:
        return PlainConsole()
    return Console()


class PlainConsole:
    def print(self, *values: Any) -> None:
        print(*(strip_rich_markup(str(value)) for value in values))

    def rule(self, title: str) -> None:
        print(f"\n{strip_rich_markup(title)}")


def strip_rich_markup(value: str) -> str:
    return re.sub(r"\[/?[^\]]+\]", "", value)


def _load_simple_yaml(config_path: Path) -> dict[str, Any]:
    """Tiny fallback parser for the project's simple two-level config.yaml."""
    data: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not raw_line.startswith(" ") and line.endswith(":"):
            section = line[:-1]
            current_section = {}
            data[section] = current_section
            continue
        if current_section is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_section[key.strip()] = value.strip().strip("\"'")
    return data


if __name__ == "__main__":
    raise SystemExit(main())

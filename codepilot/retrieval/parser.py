from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PYTHON_LANGUAGE = "python"
SYMBOL_NODE_TYPES = {"function_definition", "class_definition"}


@dataclass(frozen=True)
class CodeSymbol:
    """A parsed code symbol extracted from a source file."""

    file: str
    symbol: str
    kind: str
    code: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CodeParser:
    """Tree-sitter based parser for Python source files."""

    def __init__(self, language: str = PYTHON_LANGUAGE) -> None:
        if language != PYTHON_LANGUAGE:
            raise ValueError(f"Unsupported language: {language}")

        self.language = language
        self._parser = self._create_python_parser()

    def parse_file(
        self,
        path: str | Path,
        root: str | Path | None = None,
    ) -> list[CodeSymbol]:
        """Parse one Python file and return file/function/class symbols."""
        file_path = Path(path)
        source = file_path.read_text(encoding="utf-8")
        display_path = self._display_path(file_path, root)
        return self.parse_source(source, display_path)

    def parse_source(self, source: str, file_path: str = "<memory>") -> list[CodeSymbol]:
        """Parse Python source text and return file/function/class symbols."""
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root_node = tree.root_node

        symbols = [
            CodeSymbol(
                file=file_path,
                symbol="__file__",
                kind="file",
                code=source,
                start_line=1,
                end_line=max(1, source.count("\n") + 1),
            )
        ]
        symbols.extend(
            self._walk_symbols(
                node=root_node,
                source_bytes=source_bytes,
                file_path=file_path,
                symbol_stack=(),
            )
        )
        return symbols

    @classmethod
    def parse_repository(
        cls,
        root: str | Path,
        pattern: str = "*.py",
    ) -> list[CodeSymbol]:
        """Parse all Python files under a repository root."""
        parser = cls()
        root_path = Path(root)
        symbols: list[CodeSymbol] = []
        for file_path in sorted(root_path.rglob(pattern)):
            if file_path.is_file():
                symbols.extend(parser.parse_file(file_path, root=root_path))
        return symbols

    def _walk_symbols(
        self,
        node: object,
        source_bytes: bytes,
        file_path: str,
        symbol_stack: tuple[str, ...],
    ) -> Iterable[CodeSymbol]:
        node_type = getattr(node, "type", "")
        next_stack = symbol_stack

        if node_type in SYMBOL_NODE_TYPES:
            name = self._node_name(node, source_bytes)
            if name:
                kind = "function" if node_type == "function_definition" else "class"
                qualified_name = ".".join((*symbol_stack, name))
                next_stack = (*symbol_stack, name)
                yield CodeSymbol(
                    file=file_path,
                    symbol=qualified_name,
                    kind=kind,
                    code=self._node_text(node, source_bytes),
                    start_line=self._line_number(getattr(node, "start_point")),
                    end_line=self._line_number(getattr(node, "end_point")),
                )

        for child in getattr(node, "children", []):
            yield from self._walk_symbols(
                node=child,
                source_bytes=source_bytes,
                file_path=file_path,
                symbol_stack=next_stack,
            )

    @staticmethod
    def _node_name(node: object, source_bytes: bytes) -> str | None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        return CodeParser._node_text(name_node, source_bytes).strip()

    @staticmethod
    def _node_text(node: object, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8",
            errors="replace",
        )

    @staticmethod
    def _line_number(point: object) -> int:
        try:
            row = point[0]
        except (TypeError, KeyError):
            row = getattr(point, "row")
        return int(row) + 1

    @staticmethod
    def _display_path(path: Path, root: str | Path | None) -> str:
        if root is None:
            return path.as_posix()
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _create_python_parser() -> object:
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_python
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Python parsing requires tree-sitter and tree-sitter-python. "
                "Install project requirements before using retrieval.parser."
            ) from exc

        try:
            language = Language(tree_sitter_python.language())
        except TypeError:
            language = Language(tree_sitter_python.language(), "python")

        try:
            parser = Parser(language)
        except TypeError:
            parser = Parser()
            if hasattr(parser, "set_language"):
                parser.set_language(language)
            else:
                parser.language = language
        return parser

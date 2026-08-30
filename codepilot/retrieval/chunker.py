from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .parser import CodeParser, CodeSymbol


CHUNK_KINDS = {"function", "class"}


@dataclass(frozen=True)
class CodeChunk:
    """A code chunk prepared for embedding and indexing."""

    file: str
    symbol: str
    code: str
    kind: str = "symbol"
    start_line: int | None = None
    end_line: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CodeChunker:
    """Create code chunks from parsed Python symbols."""

    def __init__(self, parser: CodeParser | None = None) -> None:
        self.parser = parser

    def chunk_file(
        self,
        path: str | Path,
        root: str | Path | None = None,
    ) -> list[dict[str, object]]:
        parser = self.parser or CodeParser()
        return self.chunk_symbols(parser.parse_file(path, root=root))

    def chunk_repository(
        self,
        root: str | Path,
        pattern: str = "*.py",
    ) -> list[dict[str, object]]:
        parser = self.parser or CodeParser()
        root_path = Path(root)
        chunks: list[dict[str, object]] = []
        for file_path in sorted(root_path.rglob(pattern)):
            if file_path.is_file():
                chunks.extend(self.chunk_symbols(parser.parse_file(file_path, root=root_path)))
        return chunks

    def chunk_symbols(
        self,
        symbols: Iterable[CodeSymbol | dict[str, object]],
    ) -> list[dict[str, object]]:
        normalized = [self._as_symbol_dict(symbol) for symbol in symbols]
        chunks = [
            CodeChunk(
                file=str(symbol["file"]),
                symbol=str(symbol["symbol"]),
                code=str(symbol["code"]),
                kind=str(symbol["kind"]),
                start_line=self._optional_int(symbol.get("start_line")),
                end_line=self._optional_int(symbol.get("end_line")),
            ).to_dict()
            for symbol in normalized
            if symbol.get("kind") in CHUNK_KINDS
        ]

        if chunks:
            return chunks

        return [
            CodeChunk(
                file=str(symbol["file"]),
                symbol=str(symbol["symbol"]),
                code=str(symbol["code"]),
                kind=str(symbol["kind"]),
                start_line=self._optional_int(symbol.get("start_line")),
                end_line=self._optional_int(symbol.get("end_line")),
            ).to_dict()
            for symbol in normalized
            if symbol.get("kind") == "file"
        ]

    @staticmethod
    def _as_symbol_dict(symbol: CodeSymbol | dict[str, object]) -> dict[str, object]:
        if isinstance(symbol, CodeSymbol):
            return symbol.to_dict()
        return dict(symbol)

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        return int(value)

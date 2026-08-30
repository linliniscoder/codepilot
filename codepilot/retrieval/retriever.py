from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunker import CodeChunker
from .embedder import CodeEmbedder
from .index import CodeIndex


class CodeRetriever:
    """High-level code retrieval pipeline for Python repositories."""

    def __init__(
        self,
        embedder: CodeEmbedder | None = None,
        index: CodeIndex | None = None,
        chunker: CodeChunker | None = None,
    ) -> None:
        self.embedder = embedder or CodeEmbedder()
        self.index = index or CodeIndex()
        self.chunker = chunker or CodeChunker()

    def build(
        self,
        root: str | Path,
        pattern: str = "*.py",
    ) -> list[dict[str, Any]]:
        """Parse, chunk, embed, and index a Python codebase."""
        chunks = self.chunker.chunk_repository(root, pattern=pattern)
        embeddings = self.embedder.embed_chunks(chunks)
        self.index.build(chunks, embeddings)
        return chunks

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k code chunks for a natural-language query."""
        query_embedding = self.embedder.embed_query(query)
        return self.index.search(query_embedding, top_k=top_k)

    def save(
        self,
        index_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.index.save(index_path, metadata_path=metadata_path)

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_path: str | Path | None = None,
        embedder: CodeEmbedder | None = None,
    ) -> "CodeRetriever":
        return cls(
            embedder=embedder,
            index=CodeIndex.load(index_path, metadata_path=metadata_path),
        )

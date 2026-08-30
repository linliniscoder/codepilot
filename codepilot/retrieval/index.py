from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class CodeIndex:
    """FAISS IndexFlatIP wrapper with JSON chunk metadata."""

    def __init__(
        self,
        index: Any | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        self.index = index
        self.chunks = chunks or []

    def build(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> "CodeIndex":
        """Build a FAISS IndexFlatIP from chunk embeddings."""
        faiss = self._import_faiss()
        vectors = self._as_float32_matrix(embeddings)
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunks length {len(chunks)} does not match embeddings rows {vectors.shape[0]}"
            )
        if vectors.shape[0] == 0:
            raise ValueError("cannot build an index with zero chunks")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = list(chunks)
        return self

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the index and return top-k chunks with scores."""
        if self.index is None:
            raise ValueError("index has not been built or loaded")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = self._as_float32_matrix(query_embedding)
        scores, indices = self.index.search(query, min(top_k, len(self.chunks)))
        results: list[dict[str, Any]] = []
        for score, index_id in zip(scores[0], indices[0], strict=False):
            if index_id < 0:
                continue
            chunk = dict(self.chunks[int(index_id)])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def save(
        self,
        index_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> None:
        """Save FAISS index and chunk metadata."""
        if self.index is None:
            raise ValueError("cannot save an empty index")

        faiss = self._import_faiss()
        index_file = Path(index_path)
        metadata_file = Path(metadata_path) if metadata_path else self._metadata_path(index_file)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_file))
        metadata_file.write_text(
            json.dumps({"chunks": self.chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> "CodeIndex":
        """Load FAISS index and chunk metadata."""
        faiss = cls._import_faiss()
        index_file = Path(index_path)
        metadata_file = Path(metadata_path) if metadata_path else cls._metadata_path(index_file)

        index = faiss.read_index(str(index_file))
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        chunks = metadata.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError("index metadata must contain a chunks list")
        return cls(index=index, chunks=chunks)

    @staticmethod
    def _as_float32_matrix(array: np.ndarray) -> np.ndarray:
        matrix = np.asarray(array, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")
        return np.ascontiguousarray(matrix)

    @staticmethod
    def _metadata_path(index_path: Path) -> Path:
        return index_path.with_suffix(index_path.suffix + ".json")

    @staticmethod
    def _import_faiss() -> Any:
        try:
            import faiss
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CodeIndex requires faiss-cpu. Install project requirements "
                "before building or loading indexes."
            ) from exc
        return faiss

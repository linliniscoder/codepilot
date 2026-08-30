from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class CodeEmbedder:
    """Sentence-transformers based embedder for code chunks and queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CodeEmbedder requires sentence-transformers. "
                "Install project requirements before building embeddings."
            ) from exc

        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: str | Sequence[str]) -> np.ndarray:
        """Encode one or many texts into float32 embeddings."""
        inputs = [texts] if isinstance(texts, str) else list(texts)
        embeddings = self.model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if self.normalize:
            array = self._normalize(array)
        return array

    def embed_query(self, query: str) -> np.ndarray:
        return self.encode(query)

    def embed_chunks(self, chunks: Sequence[dict[str, Any]]) -> np.ndarray:
        texts = [self._chunk_text(chunk) for chunk in chunks]
        return self.encode(texts)

    @staticmethod
    def _chunk_text(chunk: dict[str, Any]) -> str:
        return (
            f"file: {chunk.get('file', '')}\n"
            f"symbol: {chunk.get('symbol', '')}\n"
            f"code:\n{chunk.get('code', '')}"
        )

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return array / norms

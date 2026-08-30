from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from retrieval.chunker import CodeChunker


def test_chunker_uses_function_and_class_symbols() -> None:
    symbols = [
        {
            "file": "app/auth.py",
            "symbol": "__file__",
            "kind": "file",
            "code": "class AuthService: pass\n",
            "start_line": 1,
            "end_line": 1,
        },
        {
            "file": "app/auth.py",
            "symbol": "AuthService",
            "kind": "class",
            "code": "class AuthService: pass",
            "start_line": 1,
            "end_line": 1,
        },
        {
            "file": "app/auth.py",
            "symbol": "login",
            "kind": "function",
            "code": "def login(): pass",
            "start_line": 3,
            "end_line": 3,
        },
    ]

    chunks = CodeChunker().chunk_symbols(symbols)

    assert [chunk["symbol"] for chunk in chunks] == ["AuthService", "login"]
    assert chunks[0]["file"] == "app/auth.py"
    assert chunks[1]["code"] == "def login(): pass"


def test_code_index_build_search_save_load_with_fake_faiss(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeIndexFlatIP:
        def __init__(self, dim: int) -> None:
            self.dim = dim
            self.vectors = np.empty((0, dim), dtype=np.float32)

        def add(self, vectors: np.ndarray) -> None:
            self.vectors = np.asarray(vectors, dtype=np.float32)

        def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
            scores = query @ self.vectors.T
            order = np.argsort(-scores, axis=1)[:, :top_k]
            ordered_scores = np.take_along_axis(scores, order, axis=1)
            return ordered_scores.astype(np.float32), order.astype(np.int64)

    stored_indexes: dict[str, FakeIndexFlatIP] = {}
    fake_faiss = types.ModuleType("faiss")
    fake_faiss.IndexFlatIP = FakeIndexFlatIP

    def write_index(index: FakeIndexFlatIP, path: str) -> None:
        stored_indexes[path] = index
        Path(path).write_text("fake", encoding="utf-8")

    fake_faiss.write_index = write_index
    fake_faiss.read_index = lambda path: stored_indexes[path]
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    from retrieval.index import CodeIndex

    chunks = [
        {"file": "a.py", "symbol": "alpha", "code": "def alpha(): pass"},
        {"file": "b.py", "symbol": "beta", "code": "def beta(): pass"},
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    index = CodeIndex().build(chunks, embeddings)
    results = index.search(np.asarray([[0.9, 0.1]], dtype=np.float32), top_k=1)

    assert results[0]["symbol"] == "alpha"
    assert "score" in results[0]

    index_path = tmp_path / "code.faiss"
    index.save(index_path)
    loaded = CodeIndex.load(index_path)

    assert loaded.chunks[1]["symbol"] == "beta"

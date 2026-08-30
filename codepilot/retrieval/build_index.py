from __future__ import annotations

import argparse
from pathlib import Path

from .embedder import DEFAULT_EMBEDDING_MODEL, CodeEmbedder
from .retriever import CodeRetriever


def build_index(
    root: str | Path,
    index_path: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> int:
    """Build and save a searchable Python code index."""
    embedder = CodeEmbedder(model_name=embedding_model, device=device)
    retriever = CodeRetriever(embedder=embedder)
    chunks = retriever.build(root)
    retriever.save(index_path)
    print(f"Built index: {index_path} ({len(chunks)} chunks)")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CodePilot Python code index")
    parser.add_argument("--root", default="workspace")
    parser.add_argument("--index", default="retrieval/index.faiss")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    build_index(
        root=args.root,
        index_path=args.index,
        embedding_model=args.embedding_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()

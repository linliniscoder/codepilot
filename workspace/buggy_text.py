from __future__ import annotations


def slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "-") + "-"


def count_words(text: str) -> int:
    return len(text.split(" "))

from __future__ import annotations


def add(a: int, b: int) -> int:
    return a + b


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def split_evenly(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, remainder = divmod(total, parts)
    result = [base] * parts
    if remainder > 0:
        result[-1] += 1
    return result

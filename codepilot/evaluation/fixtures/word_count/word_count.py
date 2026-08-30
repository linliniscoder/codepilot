import re


def count_words(value: str) -> int:
    return len(re.split(r" ", value.strip()))

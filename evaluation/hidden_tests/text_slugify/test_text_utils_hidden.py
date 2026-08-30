from text_utils import slugify


def test_slugify_collapses_repeated_separators() -> None:
    assert slugify("one---two") == "one-two"


def test_slugify_empty_text() -> None:
    assert slugify(" !!! ") == ""

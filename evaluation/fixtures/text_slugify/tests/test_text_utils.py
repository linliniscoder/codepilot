from text_utils import slugify


def test_slugify_basic_sentence() -> None:
    assert slugify("Hello World") == "hello-world"


def test_slugify_trims_outer_separators() -> None:
    assert slugify("  Python: Fast!  ") == "python-fast"

from text_utils import slugify

def test_trim_outer_separators() -> None:
    assert slugify("  Python: Fast!  ") == "python-fast"

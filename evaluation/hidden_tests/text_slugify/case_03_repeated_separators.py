from text_utils import slugify

def test_collapse_repeated_separators() -> None:
    assert slugify("one---two") == "one-two"

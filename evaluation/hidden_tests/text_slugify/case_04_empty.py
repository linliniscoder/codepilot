from text_utils import slugify

def test_empty_text() -> None:
    assert slugify(" !!! ") == ""

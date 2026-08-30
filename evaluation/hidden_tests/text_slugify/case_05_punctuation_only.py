from text_utils import slugify

def test_punctuation_only() -> None:
    assert slugify("...") == ""

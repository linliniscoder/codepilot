from text_utils import slugify

def test_trim_and_punctuation() -> None:
    assert slugify("  ---trim---  ") == "trim"

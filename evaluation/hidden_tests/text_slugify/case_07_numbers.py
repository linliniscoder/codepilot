from text_utils import slugify

def test_numbers_are_preserved() -> None:
    assert slugify("v2.0 release") == "v2-0-release"

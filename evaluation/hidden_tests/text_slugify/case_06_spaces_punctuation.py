from text_utils import slugify

def test_spaces_and_punctuation() -> None:
    assert slugify("Hello, world!!!") == "hello-world"

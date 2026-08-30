from text_utils import slugify

def test_basic_sentence() -> None:
    assert slugify("Hello World") == "hello-world"

from buggy_text import count_words, slugify


def test_slugify():
    assert slugify("Hello CodePilot") == "hello-codepilot"


def test_count_words():
    assert count_words("  hello   world  ") == 2

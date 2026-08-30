from word_count import count_words

def test_leading_and_trailing_whitespace() -> None:
    assert count_words("  one two  ") == 2

from word_count import count_words

def test_repeated_spaces() -> None:
    assert count_words("one   two") == 2

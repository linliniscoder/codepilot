from word_count import count_words

def test_newline_separated_words() -> None:
    assert count_words("one\n\n two") == 2

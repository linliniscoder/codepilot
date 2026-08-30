from word_count import count_words


def test_count_words_single_spaces() -> None:
    assert count_words("one two three") == 3


def test_count_words_empty_text() -> None:
    assert count_words("") == 0

from word_count import count_words


def test_count_words_repeated_spaces() -> None:
    assert count_words("one   two") == 2


def test_count_words_newlines_and_tabs() -> None:
    assert count_words("one\ttwo\nthree") == 3

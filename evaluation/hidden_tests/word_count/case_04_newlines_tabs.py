from word_count import count_words

def test_newlines_and_tabs() -> None:
    assert count_words("one\ttwo\nthree") == 3

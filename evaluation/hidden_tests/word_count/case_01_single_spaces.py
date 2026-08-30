from word_count import count_words

def test_single_spaces() -> None:
    assert count_words("one two three") == 3

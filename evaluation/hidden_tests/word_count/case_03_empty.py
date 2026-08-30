from word_count import count_words

def test_empty_string() -> None:
    assert count_words("") == 0

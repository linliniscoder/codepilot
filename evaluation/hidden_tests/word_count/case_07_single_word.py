from word_count import count_words

def test_single_word() -> None:
    assert count_words("word") == 1

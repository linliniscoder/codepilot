from word_count import count_words

def test_mixed_whitespace() -> None:
    assert count_words("one \n two \t three") == 3

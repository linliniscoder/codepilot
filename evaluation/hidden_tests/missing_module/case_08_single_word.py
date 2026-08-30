from app import display_name

def test_single_word() -> None:
    assert display_name("Ada") == "Ada"

from app import display_name

def test_multiple_internal_spaces() -> None:
    assert display_name("Jean    Luc   Picard") == "Jean Luc Picard"

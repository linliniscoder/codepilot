from app import display_name

def test_blank_input() -> None:
    assert display_name("   ") == ""

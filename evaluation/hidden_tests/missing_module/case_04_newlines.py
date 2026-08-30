from app import display_name

def test_collapse_newlines() -> None:
    assert display_name("  Alan\nTuring  ") == "Alan Turing"

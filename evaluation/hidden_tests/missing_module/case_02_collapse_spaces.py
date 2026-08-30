from app import display_name

def test_collapse_multiple_spaces() -> None:
    assert display_name("Grace    Hopper") == "Grace Hopper"

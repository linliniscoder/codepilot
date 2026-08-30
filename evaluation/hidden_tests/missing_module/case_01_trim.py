from app import display_name

def test_normalize_inner_spaces() -> None:
    assert display_name("Ada   Lovelace") == "Ada Lovelace"

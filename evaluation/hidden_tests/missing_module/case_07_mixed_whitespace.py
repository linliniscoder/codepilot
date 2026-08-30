from app import display_name

def test_mixed_whitespace() -> None:
    assert display_name("  Ada\n\tLovelace  ") == "Ada Lovelace"

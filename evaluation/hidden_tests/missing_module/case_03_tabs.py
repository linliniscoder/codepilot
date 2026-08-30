from app import display_name

def test_collapse_tabs() -> None:
    assert display_name("\tAda\tLovelace\t") == "Ada Lovelace"

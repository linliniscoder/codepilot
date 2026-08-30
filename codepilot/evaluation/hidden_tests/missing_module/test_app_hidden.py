from app import display_name


def test_display_name_normalizes_inner_spaces() -> None:
    assert display_name("Ada   Lovelace") == "Ada Lovelace"

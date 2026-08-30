from app import display_name


def test_display_name_removes_outer_spaces() -> None:
    assert display_name("  Ada  ") == "Ada"

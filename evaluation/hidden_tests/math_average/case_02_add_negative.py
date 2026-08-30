from calculator import add, average

def test_add_negative_numbers() -> None:
    assert add(-7, -4) == -11

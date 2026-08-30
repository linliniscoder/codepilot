from calculator import add, average


def test_add_negative_numbers() -> None:
    assert add(-2, -3) == -5


def test_average_empty_values_returns_zero() -> None:
    assert average([]) == 0

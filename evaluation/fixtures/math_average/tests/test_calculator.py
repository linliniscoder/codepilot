from calculator import add, average


def test_add_positive_numbers() -> None:
    assert add(2, 3) == 5


def test_average_values() -> None:
    assert average([2, 4, 6]) == 4

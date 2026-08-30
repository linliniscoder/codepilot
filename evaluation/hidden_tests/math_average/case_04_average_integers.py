from calculator import add, average

def test_average_integer_values() -> None:
    assert average([2, 4, 6]) == 4

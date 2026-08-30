from calculator import add, average

def test_average_empty_values_returns_zero() -> None:
    assert average([]) == 0

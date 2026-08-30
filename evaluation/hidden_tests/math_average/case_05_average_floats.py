from calculator import add, average

def test_average_float_values() -> None:
    assert average([1.5, 2.5, 3.5]) == 2.5

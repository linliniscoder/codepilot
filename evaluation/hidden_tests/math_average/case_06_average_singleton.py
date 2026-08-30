from calculator import add, average

def test_average_single_value() -> None:
    assert average([7]) == 7

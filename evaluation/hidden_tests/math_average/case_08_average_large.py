from calculator import add, average

def test_average_large_numbers() -> None:
    assert average([1000000, 1000002]) == 1000001

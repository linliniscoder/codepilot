from calculator import add, average

def test_average_mixed_signs() -> None:
    assert average([-2, 4, -6, 8]) == 1

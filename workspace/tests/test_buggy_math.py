from buggy_math import add, average, split_evenly


def test_add():
    assert add(2, 3) == 5


def test_average():
    assert average([2.0, 4.0, 6.0]) == 4.0


def test_split_evenly():
    assert split_evenly(10, 3) == [4, 3, 3]

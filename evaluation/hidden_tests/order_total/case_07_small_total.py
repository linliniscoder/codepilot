from decimal import Decimal

from order import total_after_discount

def test_small_total() -> None:
    assert total_after_discount(Decimal("0.99"), Decimal("10")) == Decimal("0.891")

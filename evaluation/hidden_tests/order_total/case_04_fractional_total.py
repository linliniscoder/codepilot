from decimal import Decimal

from order import total_after_discount

def test_fractional_total() -> None:
    assert total_after_discount(Decimal("19.99"), Decimal("15")) == Decimal("16.9915")

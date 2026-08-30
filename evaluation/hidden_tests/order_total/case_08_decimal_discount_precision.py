from decimal import Decimal

from order import total_after_discount

def test_decimal_discount_precision() -> None:
    assert total_after_discount(Decimal("10"), Decimal("7.5")) == Decimal("9.25")

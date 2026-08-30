from decimal import Decimal

from order import total_after_discount

def test_decimal_discount() -> None:
    assert total_after_discount(Decimal("200"), Decimal("12.5")) == Decimal("175.0")

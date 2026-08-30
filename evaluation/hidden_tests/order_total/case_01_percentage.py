from decimal import Decimal

from order import total_after_discount

def test_percentage_discount() -> None:
    assert total_after_discount(Decimal("100"), Decimal("20")) == Decimal("80")

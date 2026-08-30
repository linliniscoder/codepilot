from decimal import Decimal

from order import total_after_discount

def test_full_discount() -> None:
    assert total_after_discount(Decimal("80"), Decimal("100")) == Decimal("0")

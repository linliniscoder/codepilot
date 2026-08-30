from decimal import Decimal

from order import total_after_discount

def test_precision_is_preserved() -> None:
    assert total_after_discount(Decimal("19.99"), Decimal("5")) == Decimal("18.9905")

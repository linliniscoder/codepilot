from decimal import Decimal

from order import total_after_discount


def test_discount_is_a_percentage() -> None:
    assert total_after_discount(Decimal("100"), Decimal("20")) == Decimal("80")


def test_zero_discount() -> None:
    assert total_after_discount(Decimal("45"), Decimal("0")) == Decimal("45")

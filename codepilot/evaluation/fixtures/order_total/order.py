from decimal import Decimal


def total_after_discount(total: Decimal, discount_percent: Decimal) -> Decimal:
    discount = total * discount_percent
    return total - discount

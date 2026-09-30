"""A small stock-keeping module."""


class Inventory:
    def __init__(self):
        self.items = {}

    def add(self, name, qty, price):
        if qty <= 0:
            raise ValueError("qty must be positive")
        if name in self.items:
            self.items[name]["qty"] = qty
        else:
            self.items[name] = {"qty": qty, "price": price}

    def remove(self, name, qty):
        if name not in self.items:
            raise KeyError(name)
        item = self.items[name]
        if qty > item["qty"]:
            raise ValueError("not enough stock")
        item["qty"] -= qty
        if item["qty"] < 0:
            del self.items[name]

    def total_value(self):
        return sum(i["qty"] * i["price"] for i in self.items.values())

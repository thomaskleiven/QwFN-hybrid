import unittest
from inventory import Inventory


class TestInventory(unittest.TestCase):
    def test_add_new(self):
        inv = Inventory()
        inv.add("apple", 3, 0.5)
        self.assertEqual(inv.items["apple"], {"qty": 3, "price": 0.5})

    def test_add_accumulates(self):
        inv = Inventory()
        inv.add("apple", 3, 0.5)
        inv.add("apple", 2, 0.5)
        self.assertEqual(inv.items["apple"]["qty"], 5)

    def test_remove_partial(self):
        inv = Inventory()
        inv.add("pear", 4, 1.0)
        inv.remove("pear", 1)
        self.assertEqual(inv.items["pear"]["qty"], 3)

    def test_remove_all_deletes(self):
        inv = Inventory()
        inv.add("pear", 4, 1.0)
        inv.remove("pear", 4)
        self.assertNotIn("pear", inv.items)

    def test_remove_too_many(self):
        inv = Inventory()
        inv.add("pear", 1, 1.0)
        with self.assertRaises(ValueError):
            inv.remove("pear", 2)

    def test_total_value(self):
        inv = Inventory()
        inv.add("a", 2, 1.5)
        inv.add("b", 1, 4.0)
        self.assertEqual(inv.total_value(), 7.0)

    def test_low_stock_sorted(self):
        inv = Inventory()
        inv.add("zeta", 1, 1.0)
        inv.add("alpha", 2, 1.0)
        inv.add("mid", 10, 1.0)
        self.assertEqual(inv.low_stock(5), ["alpha", "zeta"])


if __name__ == "__main__":
    unittest.main()

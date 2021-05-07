import unittest
import arena


class ArenaTestCase(unittest.TestCase):
    def test_order_schema(self):
        rl = arena.Arena()
        rl.order_schema([[0, 0], [0, 0]])


if __name__ == '__main__':
    unittest.main()

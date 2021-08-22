import unittest
from example import sum


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.a = 2
        self.b = 3

    def test_something(self):
        result = sum(self.a, self.b)
        self.assertEqual(result, 5)


if __name__ == '__main__':
    unittest.main()

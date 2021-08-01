import unittest
from example import sum

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.a = 3
        self.b = 2

    def test_sum(self):
        result = sum(self.a, self.b)
        self.assertEqual(result, 5)


if __name__ == '__main__':
    unittest.main()

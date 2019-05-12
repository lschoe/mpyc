import unittest
from mpyc import gmpy


class Arithmetic(unittest.TestCase):

    def test_fpp(self):
        fpp = gmpy.factor_prime_power
        for i in range(1, 10):
            self.assertEqual(fpp(2**i), (2, i))
            self.assertEqual(fpp(3**i), (3, i))
            self.assertEqual(fpp(5**i), (5, i))
        self.assertEqual(fpp(101**7), (101, 7))
        self.assertEqual(fpp((2**31 - 1)**3), (2**31 - 1, 3))  # 8th Mersenne prime

        with self.assertRaises(ValueError):
            fpp(2*3)
        with self.assertRaises(ValueError):
            fpp(2**6 * 3)
        with self.assertRaises(ValueError):
            fpp(2**6 * 3**7)

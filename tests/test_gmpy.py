import unittest
from mpyc import gmpy


class Arithmetic(unittest.TestCase):

    def test_basic(self):
        self.assertFalse(gmpy.is_prime(1))
        self.assertTrue(gmpy.is_prime(2))
        self.assertTrue(gmpy.is_prime(101))
        self.assertFalse(gmpy.is_prime(561))
        self.assertTrue(gmpy.is_prime(2**16+1))
        self.assertFalse(gmpy.is_prime(41041))

        self.assertEqual(gmpy.next_prime(1), 2)
        self.assertEqual(gmpy.next_prime(2), 3)
        self.assertEqual(gmpy.next_prime(256), 257)

        self.assertEqual(gmpy.powmod(3, 256, 257), 1)

        self.assertEqual(gmpy.invert(3, 257), 86)
        self.assertRaises(ZeroDivisionError, gmpy.invert, 2, 4)

        self.assertEqual(gmpy.legendre(0, 101), 0)
        self.assertEqual(gmpy.legendre(42, 101), -1)
        self.assertEqual(gmpy.legendre(54, 101), 1)

        self.assertTrue(gmpy.is_square(625))
        self.assertFalse(gmpy.is_square(652))

        self.assertEqual(gmpy.isqrt(0), 0)
        self.assertEqual(gmpy.isqrt(1225), 35)

        self.assertTrue(gmpy.iroot(0, 10)[1])
        self.assertFalse(gmpy.iroot(1226, 2)[1])
        self.assertEqual(gmpy.iroot(1226, 2)[0], 35)
        self.assertEqual(gmpy.iroot(3**10 + 42, 10)[0], 3)

    def test_fpp(self):
        fpp = gmpy.factor_prime_power
        for i in range(1, 10):
            self.assertEqual(fpp(2**i), (2, i))
            self.assertEqual(fpp(3**i), (3, i))
            self.assertEqual(fpp(5**i), (5, i))
        self.assertEqual(fpp(101**7), (101, 7))
        self.assertEqual(fpp((2**31 - 1)**3), (2**31 - 1, 3))  # 8th Mersenne prime

        self.assertRaises(ValueError, fpp, 1)
        self.assertRaises(ValueError, fpp, 2*3)
        self.assertRaises(ValueError, fpp, 2**6 * 3)
        self.assertRaises(ValueError, fpp, 2**6 * 3**7)
        self.assertRaises(ValueError, fpp, (1031*1033)**2)


if __name__ == "__main__":
    unittest.main()

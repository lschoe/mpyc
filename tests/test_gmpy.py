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

        self.assertEqual(gmpy.gcdext(3, 257), (1, 86, -1))
        self.assertEqual(gmpy.gcdext(1234, 257), (1, -126, 605))
        self.assertEqual(gmpy.gcdext(-1234*3, -257*3), (3, 126, -605))

        def te_s_t(a, b):
            g, s, t = gmpy.gcdext(a, b)
            if abs(a) == abs(b):
                test_s = s == 0
            elif b == 0 or abs(b) == 2*g:
                test_s = s == bool(a > 0) - bool(a < 0)  # sign of a
            else:
                test_s = abs(s) < abs(b)/(2*g)
            if abs(a) == abs(b) or a == 0 or abs(a) == 2*g:
                test_t = t == bool(b > 0) - bool(b < 0)  # sign of b
            else:
                test_t = abs(t) < abs(a)/(2*g)
            return g == a*s + b*t and test_s and test_t

        self.assertTrue(all((te_s_t(0, 0), te_s_t(0, -1), te_s_t(1, 0), te_s_t(-1, 1))))
        self.assertTrue(te_s_t(-1234, 257))
        self.assertTrue(te_s_t(-12537, -257))
        self.assertTrue(te_s_t(-11*1234, -11*2567))
        self.assertTrue(te_s_t(1234, -2*1234))
        self.assertTrue(te_s_t(-2*12364, 12364))

        # self.assertEqual(gmpy.invert(3, -1), 0)  # pending gmpy2 issue if modulus is 1 or -1
        self.assertEqual(gmpy.invert(3, 257), 86)
        self.assertRaises(ZeroDivisionError, gmpy.invert, 2, 0)
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

    def test_ratrec(self):
        ratrec = gmpy.ratrec
        self.assertEqual(ratrec(0, 1), (0, 1))
        self.assertEqual(ratrec(0, 2), (0, 1))
        self.assertRaises(ValueError, ratrec, 0, 2, N=1, D=1)
        self.assertRaises(ValueError, ratrec, 1, 2)
        self.assertEqual(ratrec(2, 12), (2, 1))
        self.assertRaises(ValueError, ratrec, 5, 12)

        self.assertEqual(ratrec(6, 19), (-1, 3))
        self.assertRaises(ValueError, ratrec, 6, 19, N=4)
        self.assertEqual(ratrec(6, 19, D=4), (-1, 3))
        self.assertEqual(ratrec(11, 19), (3, 2))
        self.assertEqual(ratrec(11, 19, N=4), (3, 2))
        self.assertRaises(ValueError, ratrec, 11, 19, D=4)


if __name__ == "__main__":
    unittest.main()

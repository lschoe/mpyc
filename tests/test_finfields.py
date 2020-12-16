import operator
import unittest
from mpyc import gfpx
from mpyc import finfields


class Arithmetic(unittest.TestCase):

    def setUp(self):
        self.f2 = finfields.GF(gfpx.GFpX(2)(2))
        self.f256 = finfields.GF(gfpx.GFpX(2)(283))  # AES polynomial (283)_2 = X^8+X^4+X^3+X+1

        self.f2p = finfields.GF(2)
        self.f19 = finfields.GF(19)    # 19 % 4 = 3
        self.f101 = finfields.GF(101)  # 101 % 4 = 1
        self.f101.is_signed = False

        self.f27 = finfields.GF(gfpx.GFpX(3)(46))  # irreducible polynomial X^3 + 2X^2 + 1
        self.f81 = finfields.GF(gfpx.GFpX(3)(115))  # irreducible polynomial X^4 + X^3 + 2X + 1

    def test_field_caching(self):
        self.assertNotEqual(self.f2(1), self.f2p(1))
        f2_cached = finfields.GF(gfpx.GFpX(2)(2))
        self.assertEqual(self.f2(1), f2_cached(1))
        self.assertEqual(self.f2(1) * f2_cached(1), self.f2(1))
        f256_cached = finfields.GF(gfpx.GFpX(2)(283))
        self.assertEqual(self.f256(3), f256_cached(3))
        self.assertEqual(self.f256(3) * f256_cached(3), self.f256(5))
        self.assertEqual(self.f256(48) * f256_cached(16), self.f256(45))

        f2_cached = finfields.GF(2)
        self.assertEqual(self.f2p(1), f2_cached(1))
        self.assertEqual(self.f2p(1) * f2_cached(1), 1)
        f19_cached = finfields.GF(19)
        self.assertEqual(self.f19(3), f19_cached(3))
        self.assertEqual(self.f19(3) * f19_cached(3), 9)
        f101_cached = finfields.GF(101)
        self.assertEqual(self.f101(3), f101_cached(3))
        self.assertEqual(self.f101(3) * f101_cached(23), 69)

    def test_to_from_bytes(self):
        for F in [self.f2, self.f256, self.f2p, self.f19, self.f101]:
            self.assertEqual(F.from_bytes(F.to_bytes([])), [])
            self.assertEqual(F.from_bytes(F.to_bytes([0, 1])), [0, 1])
            self.assertEqual(F.from_bytes(F.to_bytes([F.order - 1])), [F.order - 1])

    def test_find_prime_root(self):
        f = finfields.find_prime_root
        pnw = f(2, False)
        self.assertEqual(pnw, (2, 1, 1))
        pnw = f(2)
        self.assertEqual(pnw, (3, 2, 3-1))
        pnw = f(5, n=1)
        self.assertEqual(pnw, (19, 1, 1))
        pnw = f(5, n=2)
        self.assertEqual(pnw, (19, 2, 19-1))
        p, n, w = f(5, n=3)
        self.assertEqual((w**3) % p, 1)
        p, n, w = f(10, n=4)
        self.assertEqual((w**n) % p, 1)

    def test_f2(self):
        f2 = self.f2
        self.assertFalse(f2(0))
        self.assertTrue(f2(1))
        self.assertEqual(f2(1) + f2(0), f2(0) + f2(1))
        self.assertEqual(1 + f2(0), 0 + f2(1))
        self.assertEqual(1 + f2(1), 0)
        self.assertEqual(1 - f2(1), 0)
        self.assertEqual(f2(1) / f2(1), f2(1))
        self.assertEqual(bool(f2(0)), False)
        self.assertEqual(bool(f2(1)), True)

        a = f2(1)
        b = f2(1)
        a += b
        self.assertEqual(a, f2(0))
        a -= b
        self.assertEqual(a, f2(1))
        a *= b
        self.assertEqual(a, f2(1))
        a /= b
        self.assertEqual(a, f2(1))

    def test_f256(self):
        f256 = self.f256
        self.assertFalse(f256(0))
        self.assertTrue(f256(1))
        self.assertEqual(f256(1) + 0, f256(0) + f256(1))
        self.assertEqual(f256(1) + 1, f256(0))
        self.assertEqual(f256(3) * 0, f256(0))
        self.assertEqual(f256(3) * 1, f256(3))
        self.assertEqual(f256(16) * f256(16), f256(27))
        self.assertEqual(f256(32) * f256(16), f256(54))
        self.assertEqual(f256(57) * f256(67), f256(137))
        self.assertEqual(f256(67) * f256(57), f256(137))
        self.assertEqual(f256(137) / f256(57), f256(67))
        self.assertEqual(f256(137) / f256(67), f256(57))

        a = f256(0)
        b = f256(1)
        a += b
        self.assertEqual(a, f256(1))
        a += 1
        self.assertEqual(a, f256(0))
        a -= b
        self.assertEqual(a, f256(1))
        a *= b
        self.assertEqual(a, f256(1))
        a *= 1
        self.assertEqual(a, f256(1))
        a /= 1
        self.assertEqual(a, f256(1))
        a <<= 0
        a = a >> 0
        self.assertEqual(a, f256(1))
        a <<= 2
        self.assertEqual(a, f256(4))
        a >>= 2
        self.assertEqual(a, f256(1))

        a = f256(3)  # generator X + 1
        s = [int((a**i).value) for i in range(255)]
        self.assertListEqual(sorted(s), list(range(1, 256)))
        s = [int((a**i).value) for i in range(-255, 0)]
        self.assertListEqual(sorted(s), list(range(1, 256)))

        f256 = finfields.GF(gfpx.GFpX(2)(391))  # primitive polynomial X^8 + X^7 + X^2 + X + 1
        a = f256(2)  # generator X
        s = [int((a**i).value) for i in range(255)]
        self.assertListEqual(sorted(s), list(range(1, 256)))

        a = f256(177)
        self.assertTrue(a.is_sqr())
        self.assertEqual(a.sqrt()**2, a)
        a = f256(255)
        self.assertEqual(a.sqrt()**2, a)

    def test_f2p(self):
        f2 = self.f2p
        self.assertEqual(f2.nth, 1)
        self.assertEqual(f2.root, 1)
        self.assertEqual(f2.root ** f2.nth, 1)
        self.assertFalse(f2(0))
        self.assertTrue(f2(1))
        self.assertEqual(f2(1) + f2(0), f2(0) + f2(1))
        self.assertEqual(1 + f2(0), 0 + f2(1))
        self.assertEqual(1 + f2(1), 0)
        self.assertEqual(1 - f2(1), 0)
        self.assertEqual(f2(1) / f2(1), 1)
        self.assertEqual(f2(1).sqrt(), 1)
        self.assertEqual(bool(f2(0)), False)
        self.assertEqual(bool(f2(1)), True)
        self.assertTrue(f2(1).is_sqr())

        a = f2(1)
        b = f2(1)
        a += b
        self.assertEqual(a, 0)
        a -= b
        self.assertEqual(a, 1)
        a *= b
        self.assertEqual(a, 1)
        a /= b
        self.assertEqual(a, 1)

    def test_f19(self):
        f19 = self.f19
        self.assertEqual(f19.nth, 2)
        self.assertEqual(f19.root, 19 - 1)
        self.assertEqual(f19(f19.root) ** f19.nth, 1)
        self.assertEqual(bool(f19(0)), False)
        self.assertEqual(bool(f19(1)), True)
        self.assertEqual(bool(f19(-1)), True)
        self.assertEqual(int(f19(-1)), -1)
        self.assertEqual(abs(f19(-1)), 1)

        a = f19(12)
        b = f19(11)
        c = a + b
        self.assertEqual(c, (a.value + b.value) % 19)
        c = c - b
        self.assertEqual(c, a)
        c = c - a
        self.assertEqual(c, 0)
        self.assertEqual(a / a, 1)
        self.assertEqual(1 / a, 8)
        self.assertEqual((f19(1).sqrt())**2, 1)
        self.assertEqual(((a**2).sqrt())**2, a**2)
        self.assertNotEqual(((a**2).sqrt())**2, -a**2)
        self.assertEqual(a**f19.modulus, a)
        b = -a
        self.assertEqual(-b, a)

        a = f19(12)
        b = f19(11)
        a += b
        self.assertEqual(a, 4)
        a -= b
        self.assertEqual(a, 12)
        a *= b
        self.assertEqual(a, 18)
        a <<= 2
        self.assertEqual(a, 15)
        a <<= 0
        self.assertEqual(a, 15)
        a >>= 2
        self.assertEqual(a, 18)
        a >>= 0
        self.assertEqual(a, 18)

    def test_f101(self):
        f101 = self.f101
        self.assertEqual(f101.nth, 2)
        self.assertEqual(f101.root, 101 - 1)
        self.assertEqual(f101(f101.root) ** f101.nth, 1)

        a = f101(12)
        b = f101(11)
        c = a + b
        self.assertEqual(c, (a.value + b.value) % 101)
        c = c - b
        self.assertEqual(c, a)
        c = c - a
        self.assertEqual(c, 0)
        self.assertEqual(a / a, 1)
        self.assertEqual((f101(1).sqrt())**2, 1)
        self.assertEqual((f101(4).sqrt())**2, 4)
        self.assertEqual(((a**2).sqrt())**2, a**2)
        self.assertNotEqual(((a**2).sqrt())**2, -a**2)
        self.assertEqual(a**f101.modulus, a)
        b = -a
        self.assertEqual(-b, a)

        a = f101(120)
        b = f101(110)
        a += b
        self.assertEqual(a, 28)
        a -= b
        self.assertEqual(a, 19)
        a *= b
        self.assertEqual(a, 70)
        a /= b
        self.assertEqual(a, 19)

    def test_f27(self):
        f27 = self.f27  # 27 == 3 (mod 4)
        a = f27(10)
        self.assertTrue((a**2).is_sqr())
        self.assertFalse((-a**2).is_sqr())
        b = (a**2).sqrt()
        self.assertEqual(b**2, a**2)
        b = (a**2).sqrt(INV=True)
        self.assertEqual((a * b)**2, 1)

    def test_f81(self):
        f81 = self.f81  # 81 == 1 (mod 4)
        a = f81(21)
        self.assertTrue((a**2).is_sqr())
        self.assertTrue((-a**2).is_sqr())
        b = (a**2).sqrt()
        self.assertEqual(b**2, a**2)
        b = (a**2).sqrt(INV=True)
        self.assertEqual((a * b)**2, 1)

    def test_errors(self):
        self.assertRaises(ValueError, finfields.GF, 4)
        self.assertRaises(ValueError, finfields.GF, gfpx.GFpX(2)(4))
        f2 = self.f2
        f2p = self.f2p
        f256 = self.f256
        f19 = self.f19
        self.assertRaises(TypeError, operator.add, f2(1), f2p(2))
        self.assertRaises(TypeError, operator.iadd, f2(1), f2p(2))
        self.assertRaises(TypeError, operator.sub, f2(1), f256(2))
        self.assertRaises(TypeError, operator.isub, f2(1), f256(2))
        self.assertRaises(TypeError, operator.mul, f2(1), f19(2))
        self.assertRaises(TypeError, operator.imul, f2(1), f19(2))
        self.assertRaises(TypeError, operator.truediv, f256(1), f19(2))
        self.assertRaises(TypeError, operator.itruediv, f256(1), f19(2))
        self.assertRaises(TypeError, operator.truediv, 3.14, f19(2))
        self.assertRaises(TypeError, operator.lshift, f2(1), f2(1))
        self.assertRaises(TypeError, operator.ilshift, f2(1), f2(1))
        self.assertRaises(TypeError, operator.lshift, 1, f2(1))
        self.assertRaises(TypeError, operator.rshift, f19(1), f19(1))
        self.assertRaises(TypeError, operator.irshift, f19(1), f19(1))
        self.assertRaises(TypeError, operator.irshift, f256(1), f256(1))
        self.assertRaises(TypeError, operator.pow, f2(1), f19(2))
        self.assertRaises(TypeError, operator.pow, f19(1), 3.14)


if __name__ == "__main__":
    unittest.main()

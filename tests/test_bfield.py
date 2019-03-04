import unittest
from mpyc import bfield


class Arithmetic(unittest.TestCase):

    def setUp(self):
        self.f2 = bfield.GF(2)
        self.f256 = bfield.GF(283)  # AES polynomial (283)_2 = x^8 + x^4 + x^3 + x + 1

    def test_field_caching(self):
        f2_cached = bfield.GF(2)
        self.assertEqual(self.f2(1), f2_cached(1))
        self.assertEqual(self.f2(1) * f2_cached(1), self.f2(1))
        f256_cached = bfield.GF(283)
        self.assertEqual(self.f256(3), f256_cached(3))
        self.assertEqual(self.f256(3) * f256_cached(3), self.f256(5))
        self.assertEqual(self.f256(48) * f256_cached(16), self.f256(45))

    def test_f2_vs_f256(self):
        f2 = self.f2
        f256 = self.f256
        with self.assertRaises(TypeError):
            f2(1) + f256(2)
        with self.assertRaises(TypeError):
            f2(1) - f256(2)
        with self.assertRaises(TypeError):
            f2(1) * f256(2)
        with self.assertRaises(TypeError):
            f2(1) / f256(2)

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
        a >>= 0
        self.assertEqual(a, f256(1))
        a <<= 2
        self.assertEqual(a, f256(4))
        a >>= 2
        self.assertEqual(a, f256(1))

        a = f256(3)  # generator x + 1
        s = [int((a**i).value) for i in range(255)]
        self.assertListEqual(sorted(s), list(range(1, 256)))
        s = [int((a**i).value) for i in range(-255, 0)]
        self.assertListEqual(sorted(s), list(range(1, 256)))

        f256 = bfield.GF(391)  # primitive polynomial x^8 + x^7 + x^2 + x + 1
        a = f256(2)  # generator x
        s = [int((a**i).value) for i in range(255)]
        self.assertListEqual(sorted(s), list(range(1, 256)))

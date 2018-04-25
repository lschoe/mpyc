import unittest
from mpyc import pfield

class Arithmetic(unittest.TestCase):

    def setUp(self):
        self.f2 = pfield.GF(2)
        self.f19 = pfield.GF(19)   # 19 % 4 = 3
        self.f101 = pfield.GF(101) # 101 % 4 = 1

    def test_field_caching(self):
        f2_cached = pfield.GF(2)
        self.assertEqual(self.f2(1), f2_cached(1))
        self.assertEqual(self.f2(1) * f2_cached(1), 1)
        f19_cached = pfield.GF(19)
        self.assertEqual(self.f19(3), f19_cached(3))
        self.assertEqual(self.f19(3) * f19_cached(3), 9)
        f101_cached = pfield.GF(101)
        self.assertEqual(self.f101(3), f101_cached(3))
        self.assertEqual(self.f101(3) * f101_cached(23), 69)

    def test_f2(self):
        f2 = self.f2
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

        a = f19(12)
        b = f19(11)
        c = a + b
        self.assertEqual(c, (a.value + b.value) % 19)
        c = c - b
        self.assertEqual(c, a)
        c = c - a
        self.assertEqual(c, 0)
        self.assertEqual(a / a, 1)
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

    def test_f2_vs_f19(self):
        f2 = self.f2
        f19 = self.f19
        with self.assertRaises(TypeError):
            f2(1) + f19(2)
        with self.assertRaises(TypeError):
            f2(1) - f19(2)
        with self.assertRaises(TypeError):
            f2(1) * f19(2)
        with self.assertRaises(TypeError):
            f2(1) / f19(2)

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

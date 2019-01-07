import unittest
from mpyc import gf2x

class Arithmetic(unittest.TestCase):

    def setUp(self):
        global poly
        poly = gf2x.Polynomial

    def test_terms(self):
        self.assertEqual(gf2x.to_terms(poly(0)), '0')
        self.assertEqual(gf2x.to_terms(poly(1)), '1')
        self.assertEqual(gf2x.to_terms(poly(2)), 'x')
        self.assertEqual(gf2x.to_terms(poly(3)), 'x+1')
        self.assertEqual(gf2x.to_terms(poly(7)), 'x^2+x+1')

    def test_degree(self):
        self.assertEqual(gf2x.degree(poly(0)), -1)
        self.assertEqual(gf2x.degree(poly(1)), 0)
        self.assertEqual(gf2x.degree(poly(2)), 1)
        self.assertEqual(gf2x.degree(poly(3)), 1)
        self.assertEqual(gf2x.degree(poly(7)), 2)

    def test_arithmetic(self):
        self.assertFalse(poly(0))
        self.assertTrue(poly(1))
        self.assertIsNot(0, poly(0))
        self.assertEqual(0, poly(0))
        self.assertEqual(poly(1) + poly(0), poly(0) + poly(1))
        self.assertEqual(1 + poly(0), 0 + poly(1))
        self.assertEqual(1 + poly(1), 0)
        self.assertEqual(1 - poly(1), 0)
        self.assertEqual(poly(15) << 2, poly(60))
        self.assertEqual(poly(32) // poly(8), poly(4))
        self.assertEqual(poly(32) % poly(8), 0)
        self.assertEqual(poly(5) // poly(3), poly(3))
        self.assertEqual(poly(5) % poly(3), 0)
        self.assertEqual(poly(3) ** 16, 2**16 + 1)
        with self.assertRaises(ValueError):
            poly(3) ** -16
        self.assertEqual(bool(poly(0)), False)
        self.assertEqual(bool(poly(1)), True)

        a = poly(1)
        b = poly(1)
        a += b
        self.assertEqual(a, poly(0))
        a -= b
        self.assertEqual(a, poly(1))
        a *= b
        self.assertEqual(a, poly(1))
        a //= b
        self.assertEqual(a, poly(1))
        a <<= 0
        self.assertEqual(a, poly(1))
        a <<= 1
        self.assertEqual(a, poly(2))

        self.assertEqual(gf2x.gcd(poly(2), poly(5)), poly(1))
        self.assertEqual(gf2x.gcd(poly(3), poly(5)), poly(3))

        # Example 2.223 from HAC:
        self.assertEqual(gf2x.gcd(poly(1905), poly(621)), poly(11))
        d, s, t = gf2x.gcdext(poly(1905), poly(621))
        self.assertEqual(d, poly(11))
        self.assertEqual(poly(1905) * s + poly(621) * t, poly(11))
        d, s, t = gf2x.gcdext(poly(621), poly(1905))
        self.assertEqual(d, poly(11))
        self.assertEqual(poly(621) * s + poly(1905) * t, poly(11))

        self.assertEqual((gf2x.invert(poly(11), poly(283)) * poly(11)) % poly(283), 1)
        self.assertEqual((gf2x.invert(poly(11) + poly(283), poly(283)) * poly(11)) % poly(283), 1)
        with self.assertRaises(ZeroDivisionError):
            gf2x.invert(poly(283), poly(283))

    def test_irreducible(self):
        self.assertFalse(gf2x.is_irreducible(0))
        self.assertFalse(gf2x.is_irreducible(1))
        self.assertTrue(gf2x.is_irreducible(2))
        self.assertTrue(gf2x.is_irreducible(3))
        self.assertFalse(gf2x.is_irreducible(4))
        self.assertFalse(gf2x.is_irreducible(5))
        self.assertFalse(gf2x.is_irreducible(6))
        self.assertTrue(gf2x.is_irreducible(7))
        self.assertFalse(gf2x.is_irreducible(17))
        self.assertTrue(gf2x.is_irreducible(283))
        self.assertTrue(gf2x.is_irreducible(391))
        self.assertFalse(gf2x.is_irreducible(621))
        self.assertFalse(gf2x.is_irreducible(1905))

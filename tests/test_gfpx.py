import unittest
from mpyc import gfpx

X = gfpx.X


class Arithmetic(unittest.TestCase):

    def setUp(self):
        global mod_2, mod_all
        gf2x = gfpx.GFpX(2)
        gf2x_ = gfpx.BinaryPolynomial
        gf3x = gfpx.GFpX(3)
        gf101x = gfpx.GFpX(101)
        mod_2 = (gf2x, gf2x_)
        mod_all = mod_2 + (gf3x, gf101x)

    def test_modall(self):
        for poly in mod_all:
            self._test_modall(poly)

    def _test_modall(self, poly):
        self.assertEqual(poly.from_terms('0'), 0)
        self.assertEqual(poly.from_terms('1'), 1)
        self.assertEqual(poly.from_terms(f'{X}'), poly.p)
        self.assertEqual(poly.to_terms(poly(0)), '0')
        self.assertEqual(poly.to_terms(poly(1)), '1')
        self.assertEqual(poly.to_terms(poly(poly.p)), f'{X}')
        self.assertEqual(poly.deg(poly(0)), -1)
        self.assertEqual(poly.deg(poly(1)), 0)
        self.assertEqual(poly.deg(poly(poly.p)), 1)
        self.assertFalse(poly(0))
        self.assertTrue(poly(1))
        self.assertIsNot(0, poly(0))
        self.assertEqual(0, poly(0))
        self.assertEqual(poly(1) + poly(0), poly(0) + poly(1))
        self.assertEqual(1 + poly(0), 0 + poly(1))
        self.assertEqual(1 - poly(1), 0)
        self.assertEqual(1 * poly(0), 0 * poly(1))
        self.assertEqual(poly(0) + [], 0)

    def test_mod2(self):
        for poly in mod_2:
            self._test_mod2(poly)

    def _test_mod2(self, poly):
        self.assertEqual(poly.from_terms(f'{X}+{X}'), 0)
        self.assertEqual(poly.to_terms(poly(2)), f'{X}')
        self.assertEqual(poly.to_terms(poly(3)), f'{X}+1')
        self.assertEqual(poly.to_terms(poly(7)), f'{X}^2+{X}+1')
        self.assertEqual(poly.deg(poly(2)), 1)
        self.assertEqual(poly.deg(poly(3)), 1)
        self.assertEqual(poly.deg(poly(7)), 2)

        self.assertEqual(1 + poly(1), 0)
        self.assertEqual(poly(3) + poly(4), 7)
        self.assertEqual(poly.add(poly(3), poly(4)), 7)
        self.assertEqual(poly.sub(poly(3), poly(4)), poly(7))
        self.assertEqual(poly(2) * poly(3), 6)
        self.assertEqual(poly(15) << 2, poly(60))
        self.assertEqual(poly(15) >> 2, poly(3))
        self.assertEqual(poly(32) // poly(8), poly(4))
        self.assertEqual(poly(32) % poly(8), 0)
        self.assertEqual(divmod(poly(32), poly(8)), (poly(4), 0))
        self.assertEqual(5 // poly(3), poly(3))
        self.assertEqual(5 % poly(3), 0)
        self.assertEqual(divmod(5, poly(3)), (poly(3), 0))
        self.assertEqual(poly(31) // 5, poly(6))
        self.assertEqual(poly(31) % 5, 1)
        self.assertEqual(divmod(poly(31), 5), (poly(6), 1))
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

        self.assertEqual(poly.gcd(poly(2), poly(5)), poly(1))
        self.assertEqual(poly.gcd(poly(3), poly(5)), poly(3))

        # Example 2.223 from HAC:
        self.assertEqual(poly.gcd(poly(1905), poly(621)), poly(11))
        d, s, t = poly.gcdext(poly(1905), poly(621))
        self.assertEqual(d, poly(11))
        self.assertEqual(poly(1905) * s + poly(621) * t, poly(11))
        d, s, t = poly.gcdext(poly(621), poly(1905))
        self.assertEqual(d, poly(11))
        self.assertEqual(poly(621) * s + poly(1905) * t, poly(11))

        self.assertEqual((poly.invert(poly(11), poly(283)) * poly(11)) % poly(283), 1)
        self.assertEqual((poly.invert(poly(11) + poly(283), poly(283)) * poly(11)) % poly(283), 1)
        with self.assertRaises(ZeroDivisionError):
            poly.invert(poly(283), poly(283))

        self.assertFalse(poly.is_irreducible(0))
        self.assertFalse(poly.is_irreducible(1))
        self.assertTrue(poly.is_irreducible(2))
        self.assertTrue(poly.is_irreducible(3))
        self.assertFalse(poly.is_irreducible(4))
        self.assertFalse(poly.is_irreducible(5))
        self.assertFalse(poly.is_irreducible(6))
        self.assertTrue(poly.is_irreducible(7))
        self.assertFalse(poly.is_irreducible(17))
        self.assertTrue(poly.is_irreducible(283))
        self.assertTrue(poly.is_irreducible(391))
        self.assertFalse(poly.is_irreducible(621))
        self.assertFalse(poly.is_irreducible(1905))

        self.assertTrue(poly(7) > poly(5))

    def test_mod3(self):
        poly = gfpx.GFpX(3)
        self.assertEqual(poly(1) + poly(1), 2)
        self.assertEqual(poly(1) * poly(2), 2)
        self.assertEqual(poly(4) + poly(2), 3)
        self.assertEqual(poly(4) * poly(2), 8)
        self.assertEqual(poly(5) << 2, poly(45))
        self.assertEqual(poly(45) >> 2, poly(5))
        self.assertEqual(poly(82) % poly(46), 15)
        self.assertEqual(divmod(poly(81), poly(9)), (poly(9), 0))
        self.assertEqual(divmod(poly(83), poly(9)), (poly(9), poly(2)))

        self.assertEqual(poly.gcd(poly(2), poly(5)), poly(1))
        self.assertEqual(poly.gcd(poly(3), poly(5)), poly(1))
        self.assertEqual(poly.gcd(poly(455), poly(410)), poly(5))
        with self.assertRaises(ZeroDivisionError):
            poly.invert(poly(283), poly(283))

        self.assertTrue(poly.is_irreducible(3))
        self.assertFalse(poly.is_irreducible(99985))
        self.assertTrue(poly.is_irreducible(99986))
        self.assertEqual(poly.next_irreducible(53), 86)

        self.assertTrue(poly(7) > poly(5))

    def test_mod11(self):
        poly = gfpx.GFpX(11)
        self.assertEqual(poly.to_terms(poly(122)), f'{X}^2+1')
        self.assertEqual(poly.add(poly(9), poly(4)), 2)
        self.assertEqual(poly(1) + poly(1), 2)
        self.assertEqual(poly(1) * poly(2), 2)
        self.assertEqual(poly(4) + poly(2), 6)
        self.assertEqual(poly(4) * poly(2), 8)
        self.assertEqual(poly(5) << 2, poly(605))
        self.assertEqual(poly(605) >> 2, poly(5))
        self.assertEqual(divmod(poly(14641), poly(121)), (poly(121), 0))
        self.assertEqual(divmod(poly(14643), poly(121)), (poly(121), poly(2)))

        self.assertEqual(poly.gcd(poly(2), poly(5)), poly(1))
        self.assertEqual(poly.gcd(poly(3), poly(5)), poly(1))
        self.assertEqual(poly.gcd(poly(455), poly(420)), poly(19))
        with self.assertRaises(ZeroDivisionError):
            poly.invert(poly(283), poly(283))

        self.assertTrue(poly.is_irreducible(11))
        self.assertFalse(poly.is_irreducible(19487))
        self.assertTrue(poly.is_irreducible(19488))
        self.assertEqual(poly.next_irreducible(53), 122)

        self.assertTrue(poly(17) > poly(15))

import operator
import unittest
from mpyc.numpy import np
from mpyc.gfpx import GFpX
from mpyc.secpols import secpoly
from mpyc.runtime import mpc


class Arithmetic(unittest.TestCase):

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_errors(self):
        self.assertRaises(TypeError, secpoly, [])
        f = secpoly(np.array([]), sectype=mpc.SecFld(101))
        g = secpoly(np.array([]), sectype=mpc.SecFld(257))
        self.assertRaises(TypeError, operator.add, f, g)
        self.assertRaises(TypeError, operator.sub, f, g)
        self.assertRaises(TypeError, operator.mul, f, g)
        self.assertRaises(ValueError, f.reverse, -2)
        self.assertRaises(IndexError, f.__getitem__, 1.0)
        self.assertRaises(IndexError, f.__getitem__, -1)

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_secfld(self):
        p = 31
        secfld = mpc.SecFld(p)
        poly = GFpX(p)
        s = [[0, 0, 1, -1, 2, 3], [30, 13, 17, 13], [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        t = [[3], [2, 2, 2], [11, 0, 12, 0, 13]]
        s = [[a % p for a in x] for x in s]
        t = [[a % p for a in x] for x in t]
        for a, b in zip(s + t, t + s):
            a = poly(a)
            b = poly(b)
            f = secpoly(a, sectype=secfld)
            g = secpoly(b, sectype=secfld)
            f = f.copy()  # dummy copy operation
            z = f * g - g * f
            self.assertEqual(mpc.run(mpc.output(f(3))), a(3))
            self.assertEqual(mpc.run(mpc.output(f(secfld(-2)))), a(-2))
            self.assertEqual(mpc.run(mpc.output(f.degree())), a.degree())
            self.assertEqual(mpc.run(mpc.output((f*f*f).degree())), (a*a*a).degree())
            self.assertEqual(mpc.run(mpc.output(z.degree())), -1)
            self.assertEqual(mpc.run(mpc.output((z + f).degree())), a.degree())
            self.assertEqual(mpc.run(mpc.output(f.reverse(-1))), a.reverse(-1))
            self.assertEqual(mpc.run(mpc.output(f.reverse(10))), a.reverse(10))
            self.assertEqual(mpc.run(mpc.output(f.reverse(mpc.SecInt()(-1)))), a.reverse(-1))
            if len(f.share) >= 3:
                self.assertEqual(mpc.run(mpc.output(f.reverse(secfld(2)))), a.reverse(2))
            self.assertEqual(mpc.run(mpc.output((f + z).reverse(2))), a.reverse(2))
            self.assertEqual(mpc.run(mpc.output((f*f*f).reverse())), (a*a*a).reverse())
            self.assertEqual(mpc.run(mpc.output(f.truncate(2))), a.truncate(2))
            self.assertEqual(mpc.run(mpc.output(f[2])), a[2])
            self.assertEqual(mpc.run(mpc.output(f)), a)
            self.assertEqual(mpc.run(mpc.output(-f)), -a)
            self.assertEqual(mpc.run(mpc.output(+f)), +a)
            self.assertEqual(mpc.run(mpc.output(g)), b)
            self.assertEqual(mpc.run(mpc.output(-g)), -b)
            self.assertEqual(mpc.run(mpc.output(+g)), +b)
            self.assertEqual(mpc.run(mpc.output(f + (-f))), 0)
            self.assertEqual(mpc.run(mpc.output(f + g)), a + b)
            self.assertEqual(mpc.run(mpc.output(z + f + g)), a + b)
            self.assertEqual(mpc.run(mpc.output(f + b)), a + b)
            self.assertEqual(mpc.run(mpc.output(a + g)), a + b)
            self.assertEqual(mpc.run(mpc.output(f - g)), a - b)
            self.assertEqual(mpc.run(mpc.output(z + f - g)), a - b)
            self.assertEqual(mpc.run(mpc.output(f - b)), a - b)
            self.assertEqual(mpc.run(mpc.output(a - g)), a - b)
            self.assertEqual(mpc.run(mpc.output(f * g)), a * b)
            self.assertEqual(mpc.run(mpc.output((z + f) * g)), a * b)
            self.assertEqual(mpc.run(mpc.output(f * b)), a * b)
            self.assertEqual(mpc.run(mpc.output(a * g)), a * b)
            self.assertEqual(mpc.run(mpc.output(f // g)), a // b)
            self.assertEqual(mpc.run(mpc.output(f // (z + g))), a // b)
            self.assertEqual(mpc.run(mpc.output((z + f) // g)), a // b)
            self.assertEqual(mpc.run(mpc.output(a // g)), a // b)
            self.assertEqual(mpc.run(mpc.output(f // b)), a // b)
            self.assertEqual(mpc.run(mpc.output(f // f)), a // a)
            self.assertEqual(mpc.run(mpc.output((z + f) // f)), a // a)
            self.assertEqual(mpc.run(mpc.output(g // f)), b // a)
            self.assertEqual(mpc.run(mpc.output((f // (f + z)) // ((z + f) // f))), 1)
            self.assertEqual(mpc.run(mpc.output(g % g)), 0)
            self.assertEqual(mpc.run(mpc.output(secpoly.invert(f, g))), poly.invert(a, b))
            self.assertEqual(mpc.run(mpc.output(secpoly.powmod(f, 0, g))), poly.powmod(a, 0, b))
            self.assertEqual(mpc.run(mpc.output(secpoly.powmod(f, -3, g))), poly.powmod(a, -3, b))
            self.assertEqual(mpc.run(mpc.output(f % g)), a % b)
            self.assertEqual(mpc.run(mpc.output(a % g)), a % b)
            self.assertEqual(mpc.run(mpc.output(f % b)), a % b)
            self.assertTrue(mpc.run(mpc.output(f == divmod(f, g)[0] * g + divmod(a, g)[1])))
            self.assertEqual(mpc.run(mpc.output(f << 3)), a << 3)
            self.assertEqual(mpc.run(mpc.output(f >> 2)), a >> 2)
            self.assertEqual(mpc.run(mpc.output(f < g)), a < b)
            self.assertEqual(mpc.run(mpc.output(f <= g)), a <= b)
            self.assertEqual(mpc.run(mpc.output(f > g)), a > b)
            self.assertEqual(mpc.run(mpc.output(f >= g)), a >= b)
            self.assertTrue(mpc.run(mpc.output(f == f)))
            self.assertTrue(mpc.run(mpc.output(f == f + z)))
            self.assertEqual(mpc.run(mpc.output(f == g)), a == b)
            self.assertEqual(mpc.run(mpc.output(f != g)), a != b)
            self.assertEqual(mpc.run(mpc.output(secpoly.gcd(f, g))), poly.gcd(a, b))
            self.assertEqual(mpc.run(mpc.output(secpoly.gcd(f + g, g))), poly.gcd(a + b, b))
            self.assertEqual(mpc.run(mpc.output(secpoly.gcd(z + f, z + g))), poly.gcd(a, b))
            self.assertEqual(mpc.run(mpc.output(list(secpoly.gcdext(f, g)))),
                             list(poly.gcdext(a, b)))
            self.assertEqual(mpc.run(mpc.output(list(secpoly.gcdext(f + g, g)))),
                             list(poly.gcdext(a + b, b)))
            self.assertEqual(mpc.run(mpc.output(list(secpoly.gcdext(f * g + poly(1), f)))),
                             list(poly.gcdext(a * b + 1, a)))
            self.assertEqual(mpc.run(mpc.output(secpoly.is_irreducible(f))),
                             poly.is_irreducible(a))


if __name__ == "__main__":
    unittest.main()

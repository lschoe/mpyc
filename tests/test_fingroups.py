import operator
import unittest
from mpyc import fingroups as fg


class Arithmetic(unittest.TestCase):

    def setUp(self):
        self.S3 = fg.SymmetricGroup(3)
        self.QR11 = fg.QuadraticResidues(11)
        self.SG11 = fg.SchnorrGroup(11, 5, 4)
        self.Cl23 = fg.ClassGroup(Delta=-23)

    def test_group_caching(self):
        S3_cached = fg.SymmetricGroup(3)
        self.assertEqual(self.S3, S3_cached)
        self.assertEqual(self.S3(), S3_cached())
        QR11_cached = fg.QuadraticResidues(l=4)
        self.assertEqual(self.QR11, QR11_cached)
        Cl23_cached = fg.ClassGroup(Delta=-23)
        self.assertEqual(self.Cl23, Cl23_cached)
        self.assertEqual(self.Cl23(), Cl23_cached())
        Cl23_cached = fg.ClassGroup(l=5)
        self.assertEqual(self.Cl23, Cl23_cached)

    def test_Sn(self):
        S0 = fg.SymmetricGroup(0)
        self.assertEqual(S0.degree, 0)
        self.assertEqual(S0.order, 1)
        p = S0.identity
        self.assertEqual(p^2, p)
        self.assertRaises(TypeError, operator.add, p, p)
        self.assertRaises(TypeError, operator.mul, p, p)
        self.assertRaises(TypeError, operator.mul, 1, p)

        S1 = fg.SymmetricGroup(1)
        self.assertEqual(S1.degree, 1)
        self.assertEqual(S1.order, 1)
        p = S1.identity
        self.assertEqual(p^2, p^-1)

        S3 = fg.SymmetricGroup(3)
        self.assertEqual(S3.degree, 3)
        self.assertEqual(S3.order, 6)
        self.assertEqual(S3.identity, S3([0, 1, 2]))
        p = S3([1, 2, 0])
        self.assertEqual(p^3, S3.identity)
        q = S3([1, 0, 2])
        self.assertEqual(q @ q, S3.identity)
        self.assertEqual(q, ~q)
        self.assertEqual(p @ q, S3([0, 2, 1]))
        self.assertEqual(q @ p, S3([2, 1, 0]))

        self.assertEqual({p, q, q}, {p, p, q})

        self.assertRaises(ValueError, S3, [1, 2])
        self.assertRaises(ValueError, S3, [1, 2, 3])

    def test_QR_SG(self):
        for group in self.QR11, self.SG11:
            self.assertEqual(group.order, 5)
            self.assertTrue(group.is_cyclic)
            self.assertEqual(group.identity, group(1))
            a = group(3)
            self.assertEqual(a^5, group.identity)
            b = group(4)
            self.assertEqual(b^5, group.identity)
            self.assertEqual(a * b, group.identity)
            self.assertEqual(1/a, b)
            self.assertEqual(a / b, a**2)
            self.assertRaises(TypeError, operator.truediv, 2, a)
            self.assertRaises(TypeError, operator.add, a, b)
            self.assertRaises(TypeError, operator.neg, a)
            self.assertRaises(TypeError, operator.sub, a, b)
            self.assertRaises(TypeError, group, 1.0)
            self.assertRaises(ValueError, group, 0)
            self.assertRaises(ValueError, group, 2)

            self.assertEqual({a, b, b}, {a, a, b})

        self.assertRaises(ValueError, fg.QuadraticResidues, 2)
        QR_768 = fg.QuadraticResidues(l=768)
        self.assertEqual(QR_768.decode(*QR_768.encode(42)), 42)
        SG = fg.SchnorrGroup(q=2**31 - 1)
        self.assertEqual(SG.decode(*SG.encode(42)), 42)

    def test_EC(self):
        curves = (fg.EllipticCurve('Ed25519'),  # affine coordinates
                  fg.EllipticCurve('Ed25519', coordinates='projective'),
                  fg.EllipticCurve('Ed25519', coordinates='extended'),
                  fg.EllipticCurve('Ed448'),  # affine coordinates
                  fg.EllipticCurve('Ed448', coordinates='projective'),
                  fg.EllipticCurve('secp256k1'),  # affine coordinates
                  fg.EllipticCurve('secp256k1', coordinates='projective'),
                  fg.EllipticCurve('secp256k1', coordinates='jacobian'),
                  fg.EllipticCurve('BN256'),  # affine coordinates
                  fg.EllipticCurve('BN256', coordinates='projective'),
                  fg.EllipticCurve('BN256', coordinates='jacobian'),
                  fg.EllipticCurve('BN256_twist', coordinates='projective'))
        for group in curves:
            self.assertEqual(5*group.identity, group.identity^-1)
            g = group.generator
            self.assertEqual(group(g.value), g)
            self.assertEqual((g^12) - 13*g, -g)
            self.assertEqual(~-g, g)
            self.assertEqual(g - g, group.identity)
            self.assertEqual(group.order*g, group.identity)

            self.assertEqual({g, -g, -g}, {g, g, -g})

            if group.curvename != 'BN256_twist':
                self.assertEqual(group.decode(*group.encode(42)), 42)

            self.assertRaises(TypeError, operator.mul, g, 13)
            self.assertRaises(TypeError, operator.truediv, g, g)
            self.assertRaises(TypeError, operator.truediv, 1, g)
            self.assertRaises(TypeError, operator.pow, g, 1)
            self.assertRaises(ValueError, group, (0, 0))

        self.assertRaises(ValueError, fg.EllipticCurve, 'Ed25519', 'jacobian')
        self.assertRaises(ValueError, fg.EllipticCurve, 'BN256', 'extended')

    def test_Cl(self):
        Cl3 = fg.ClassGroup()  # trivial class group D=-3 with 1 elt
        g = Cl3((1, 1, 1))
        self.assertEqual(g * (1 / g) @ g^2, Cl3.identity)
        self.assertEqual({g, 1/g, 1/g}, {g, g, 1/g})

        Cl23 = self.Cl23  # discriminant -23
        self.assertEqual(Cl23.order, 3)
        self.assertTrue(Cl23.is_multiplicative)
        g = Cl23.generator
        self.assertEqual(g, Cl23((2, 1, 3)))
        self.assertEqual(g * g, Cl23((2, -1)))
        self.assertEqual((g^2) @ g, Cl23.identity)
        self.assertEqual(g @ g.inverse(), Cl23.identity)

        Cl227 = fg.ClassGroup(Delta=-227)  # Example 9.6.2 from Buchman&Vollmer
        self.assertEqual(Cl227.order, 5)
        self.assertEqual(Cl227((1, 1, 57)), Cl227.identity)
        g = Cl227((3, 1, 19))
        self.assertEqual(g^5, Cl227.identity)

        Cl1123 = fg.ClassGroup(Delta=-1123)  # Example 9.7.5 from Buchman&Vollmer
        self.assertEqual(Cl1123((1, 1, 281)), Cl1123.identity)
        g = Cl1123((7, 5, 41))
        self.assertEqual(g^5, Cl1123.identity)
        self.assertEqual(g^3, Cl1123((17, 13, 19)))
        self.assertRaises(ValueError, Cl23, (1, 1, 2))
        self.assertRaises(ValueError, Cl23, (2, 2, 2))

        Cl_2_16 = fg.ClassGroup(l=16)
        g = Cl_2_16.generator
        a = (g^10000)^128
        self.assertEqual(a @ (a^-1), Cl_2_16.identity)

        Cl_2_32 = fg.ClassGroup(l=32)
        self.assertEqual(Cl_2_32.generator^20021, Cl_2_32.identity)
        self.assertEqual(Cl_2_32.decode(*Cl_2_32.encode(24)), 24)

        self.assertRaises(ValueError, fg.ClassGroup, -13)
        self.assertRaises(ValueError, fg.ClassGroup, 13)
        self.assertRaises(ValueError, fg.ClassGroup, -12)
        self.assertRaises(ValueError, Cl23, (-2, 1, -3))


if __name__ == "__main__":
    unittest.main()

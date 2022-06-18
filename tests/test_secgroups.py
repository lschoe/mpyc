import operator
import unittest
import mpyc.fingroups as fg
from mpyc.runtime import mpc


class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    # TODO: test caching

    def test_Sn(self):
        group = fg.SymmetricGroup(5)
        a = group([3, 4, 2, 1, 0])
        b = a @ a
        secgrp = mpc.SecGrp(group)
        c = secgrp(a)
        d = a @ c
        self.assertEqual(mpc.run(mpc.output(d)), b)
        e = ~c
        f = e @ b
        self.assertEqual(mpc.run(mpc.output(f)), a)
        self.assertTrue(mpc.run(mpc.output(f == c)))

        group = fg.SymmetricGroup(11)
        secgrp = mpc.SecGrp(group)
        a = group([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0])
        secfld = mpc.SecFld(11)  # ord(a) = 11
        a7 = secgrp.repeat(a, secfld(7))
        self.assertEqual(mpc.run(mpc.output(a7)), a^7)
        a7 = secgrp.repeat_public(a, secfld(7))
        self.assertEqual(mpc.run(a7), a^7)
        a6 = a^6
        a12 = a6 @ a6
        self.assertEqual(mpc.run(mpc.output(secgrp(a6).inverse())), a^5)
        self.assertEqual(mpc.run(mpc.output((secgrp(a)^6) @ secgrp.identity)), a6)
        self.assertEqual(mpc.run(mpc.output(secgrp.repeat(a6, secfld(2)))), a12)
        self.assertEqual(mpc.run(mpc.output(secgrp.repeat(secgrp(a), secfld(6)))), a6)

        p = secgrp(a)
        self.assertRaises(TypeError, operator.add, p, p)
        self.assertRaises(TypeError, operator.mul, p, p)
        self.assertRaises(TypeError, operator.mul, 1, p)
        group.is_multiplicative = True
        self.assertTrue(mpc.run(mpc.output(a * p == a^2)))
        group.is_multiplicative = False
        self.assertRaises(ValueError, secgrp, [0, 1, 2, 3])

    def test_QR_SG(self):
        for group in fg.QuadraticResidues(l=768), fg.SchnorrGroup(l=768):
            secgrp = mpc.SecGrp(group)
            g = group.generator
            g2 = mpc.run(mpc.output(secgrp(g) * g))
            self.assertEqual(int(g), int(group.identity * g))
            self.assertFalse(mpc.run(mpc.output(secgrp(g)/g != group.identity)))
            self.assertTrue(mpc.run(mpc.output(g * secgrp(g) == g2)))
            secfld = mpc.SecFld(modulus=secgrp.group.order)
            self.assertEqual(mpc.run(mpc.output(secgrp.repeat(g, -secfld(2)))), 1/g2)
            self.assertEqual(mpc.run(mpc.output(secgrp.repeat(secgrp(g), 2))), g2)
            m, z = group.encode(15)
            self.assertEqual(mpc.run(mpc.output(secgrp.decode(secgrp(m), secgrp(z)))), 15)
            h = secgrp.if_else(secgrp.sectype(0), g, secgrp(g2))
            self.assertEqual(mpc.run(mpc.output(h)), g2)

            a = secgrp(g)
            self.assertRaises(TypeError, operator.truediv, 2, a)
            self.assertRaises(TypeError, operator.add, a, a)
            self.assertRaises(TypeError, operator.add, g, a)
            self.assertRaises(TypeError, operator.neg, a)
            self.assertRaises(TypeError, operator.sub, a, a)
            self.assertRaises(TypeError, operator.sub, g, a)

    def test_EC(self):
        curves = (fg.EllipticCurve('Ed25519'),  # affine coordinates
                  fg.EllipticCurve('Ed25519', coordinates='projective'),
                  fg.EllipticCurve('Ed25519', coordinates='extended'),
                  fg.EllipticCurve('Ed448', coordinates='projective'),
                  fg.EllipticCurve('secp256k1', coordinates='projective'),
                  fg.EllipticCurve('BN256', coordinates='projective'),
                  fg.EllipticCurve('BN256_twist', coordinates='projective'))
        for group in curves:
            secgrp = mpc.SecGrp(group)
            secfld = mpc.SecFld(modulus=secgrp.group.order)
            g = group.generator
            self.assertFalse(mpc.run(mpc.output(secgrp(g) != g)))
            b = secgrp(g.value)
            self.assertEqual(mpc.run(mpc.output(b - b)), group.identity)
            c = secgrp(g)
            self.assertEqual(mpc.run(mpc.output(b)), mpc.run(mpc.output(c)))
            self.assertEqual(mpc.run(secgrp.repeat_public(g, -secfld(2))), g^-2)
            self.assertEqual(mpc.run(mpc.output(secfld(2)*g)), g^2)
            self.assertEqual(mpc.run(mpc.output(2*secgrp(g))), g^2)
            bp4 = 4*g
            sec_bp4 = 4*secgrp(g) + secgrp.identity
            self.assertEqual(mpc.run(mpc.output(-sec_bp4)), -bp4)
            sec_bp8 = secgrp.repeat(bp4, secfld(2))
            self.assertEqual(mpc.run(mpc.output(sec_bp8)), bp4 + bp4)
            self.assertEqual(mpc.run(mpc.output(secgrp.repeat(bp4, secfld(3)))), 3*bp4)
            self.assertEqual(mpc.run(mpc.output(group.identity + b)), g)
            self.assertEqual(mpc.run(mpc.output(g - b)), group.identity)
            if group.curvename != 'BN256_twist':
                m, z = group.encode(42)
                self.assertEqual(mpc.run(mpc.output(secgrp.decode(secgrp(m), secgrp(z)))), 42)

            self.assertRaises(TypeError, operator.mul, sec_bp4, 13)
            self.assertRaises(TypeError, operator.truediv, sec_bp4, sec_bp4)
            self.assertRaises(TypeError, operator.truediv, 1, sec_bp4)
            self.assertRaises(TypeError, operator.pow, sec_bp4, 1)
            self.assertRaises(ValueError, secgrp, [0])

    def test_Cl(self):
        Cl23 = fg.ClassGroup(Delta=-23)
        secgrp = mpc.SecGrp(Cl23)
        secint = secgrp.sectype
        g = Cl23.generator
        self.assertFalse(mpc.run(mpc.output(secgrp(g) != g)))
        self.assertEqual(mpc.run(secgrp.repeat_public(g, -secint(2))), g)
        self.assertEqual(mpc.run(mpc.output(g**secint(-2))), g)
        self.assertEqual(mpc.run(mpc.output(g * secgrp(g))), Cl23((2, -1, 3)))

        Cl227 = fg.ClassGroup(Delta=-227)  # Example 9.6.2 from Buchman&Vollmer
        secgrp = mpc.SecGrp(Cl227)
        g = Cl227((3, 1, 19))
        self.assertEqual(mpc.run(mpc.output(secgrp(g)^5)), g^5)

        Cl1123 = fg.ClassGroup(Delta=-1123)  # Example 9.7.5 from Buchman&Vollmer
        secgrp = mpc.SecGrp(Cl1123)
        self.assertEqual(Cl1123((1, 1, 281)), Cl1123.identity)
        g = Cl1123((7, 5, 41))
        self.assertEqual(mpc.run(mpc.output(secgrp(g)^5)), g^5)
        self.assertEqual(mpc.run(mpc.output(secgrp(g)**3)), g^3)

        group = fg.ClassGroup(l=28)
        secgrp = mpc.SecGrp(group)
        g = group.generator
        a = secgrp(g)^6
        self.assertEqual(mpc.run(mpc.output(a)), g^6)
        self.assertEqual(mpc.run(mpc.output(a * (a^-1))), group.identity)
        m, z = group.encode(5)
        self.assertEqual(mpc.run(mpc.output(secgrp.decode(secgrp(m), secgrp(z)))), 5)

        self.assertRaises(ValueError, secgrp, [0])


if __name__ == "__main__":
    unittest.main()

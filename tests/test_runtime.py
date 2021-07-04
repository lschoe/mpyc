import operator
import math
import unittest
from mpyc.runtime import mpc


class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mpc.logging(False)
        mpc.run(mpc.start())

    @classmethod
    def tearDownClass(cls):
        mpc.run(mpc.shutdown())

    def test_io(self):
        x = ({4, 3}, [1 - 1j, 2.5], 0, range(7))
        self.assertEqual(mpc.run(mpc.transfer(x))[0], x)
        self.assertEqual(mpc.run(mpc.transfer(x, senders=0)), x)
        self.assertEqual(mpc.run(mpc.transfer(x, senders=[0]))[0], x)
        self.assertEqual(mpc.run(mpc.transfer(x, senders=iter(range(1))))[0], x)
        self.assertEqual(mpc.run(mpc.transfer(x, receivers=0))[0], x)
        self.assertEqual(mpc.run(mpc.transfer(x, receivers=[0]))[0], x)
        self.assertEqual(mpc.run(mpc.transfer(x, receivers=iter(range(1))))[0], x)
        self.assertEqual(mpc.run(mpc.transfer(x, senders=0, receivers=0)), x)
        self.assertEqual(mpc.run(mpc.transfer(x, senders=[0], receivers=[0]))[0], x)
        self.assertEqual(mpc.run(mpc.transfer(x, sender_receivers=[(0, 0)]))[0], x)
        self.assertEqual(mpc.run(mpc.transfer(x, sender_receivers={0: {0}}))[0], x)

        a = mpc.SecInt()(7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a)))[0], 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a)[0])), 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, senders=0))), 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, senders=[0])))[0], 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, senders=iter(range(1)))))[0], 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, senders=[0])))[0], 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, senders=[0])[0])), 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a), receivers=0))[0], 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a)[0], receivers=0)), 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a), receivers=[0]))[0], 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a), receivers=iter(range(1))))[0], 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a)[0], receivers=[0])), 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, senders=0), receivers=0)), 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, senders=[0]), receivers=[0]))[0], 7)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, senders=[0])[0], receivers=[0])), 7)

        x = [a, a]
        self.assertEqual(mpc.run(mpc.output(mpc.input(x)[0])), [7, 7])
        self.assertEqual(mpc.run(mpc.output(mpc.input(x, senders=0))), [7, 7])
        self.assertEqual(mpc.run(mpc.output(mpc.input(x, senders=[0])[0])), [7, 7])
        self.assertEqual(mpc.run(mpc.output(mpc.input(x)[0], receivers=0)), [7, 7])
        self.assertEqual(mpc.run(mpc.output(mpc.input(x)[0], receivers=[0])), [7, 7])
        self.assertEqual(mpc.run(mpc.output(mpc.input(x, senders=0), receivers=0)), [7, 7])
        self.assertEqual(mpc.run(mpc.output(mpc.input(x, senders=[0])[0], receivers=[0])), [7, 7])

    def test_pickle(self):
        xsecfld = mpc.SecFld(256)
        psecfld = mpc.SecFld(257)
        secint = mpc.SecInt()
        secfxp = mpc.SecFxp()
        secflt = mpc.SecFlt()
        # NB: mpc.transfer() calls pickle.dumps() and pickle.loads()
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(xsecfld(12), senders=0)))), 12)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(psecfld(12), senders=0)))), 12)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(secint(12), senders=0)))), 12)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(secfxp(12.5), senders=0)))), 12.5)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(secflt(12.5), senders=0)))), 12.5)
        self.assertEqual(mpc.run(mpc.transfer(xsecfld.field(12), senders=0)), 12)
        self.assertEqual(mpc.run(mpc.transfer(psecfld.field(12), senders=0)), 12)
        self.assertEqual(mpc.run(mpc.transfer(secint.field(12), senders=0)), 12)
        self.assertEqual(mpc.run(mpc.transfer(secfxp.field(13), senders=0)), 13)
        self.assertEqual(mpc.run(mpc.transfer(xsecfld.field.modulus, 0)), xsecfld.field.modulus)

        x = [(xsecfld(12), psecfld(12), secint(12), secfxp(12.5), secflt(12.5)),
             [xsecfld.field(12), psecfld.field(12), secint.field(12), secfxp.field(13)],
             xsecfld.field.modulus]
        y = mpc.run(mpc.transfer(x, senders=0))
        self.assertTrue(all(mpc.run(mpc.output(a == b)) for a, b in zip(y[0], x[0])))
        self.assertEqual(y[1], x[1])
        self.assertEqual(y[2], x[2])

    def test_psecfld(self):
        secfld = mpc.SecFld(min_order=2**16)
        a = secfld(1)
        b = secfld(0)
        self.assertEqual(mpc.run(mpc.output(a + b)), 1)
        self.assertEqual(mpc.run(mpc.output(a * b)), 0)

        self.assertEqual(mpc.run(mpc.output(mpc.to_bits(secfld(0), 8))), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mpc.run(mpc.output(mpc.to_bits(secfld(1), 8))), [1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mpc.run(mpc.output(mpc.to_bits(secfld(255), 8))), [1, 1, 1, 1, 1, 1, 1, 1])

        secfld = mpc.SecFld(modulus=101)
        a = secfld(1)
        b = secfld(-1)
        self.assertEqual(mpc.run(mpc.output(a + b)), 0)
        self.assertEqual(mpc.run(mpc.output(a * b)), 100)
        self.assertEqual(mpc.run(mpc.output(a / b)), 100)
        self.assertEqual(mpc.run(mpc.output(a == b)), 0)
        self.assertEqual(mpc.run(mpc.output(a == -b)), 1)
        self.assertEqual(mpc.run(mpc.output(a**2 == b**2)), 1)
        self.assertEqual(mpc.run(mpc.output(a != b)), 1)
        a = secfld(0)
        b = secfld(1)
        self.assertEqual(mpc.run(mpc.output(a & b)), 0)
        self.assertEqual(mpc.run(mpc.output(a | b)), 1)
        self.assertEqual(mpc.run(mpc.output(a ^ b)), 1)
        self.assertEqual(mpc.run(mpc.output(~a)), 1)

        self.assertIn(mpc.run(mpc.output(mpc.random_bit(secfld))), [0, 1])
        self.assertIn(mpc.run(mpc.output(mpc.random_bit(secfld, signed=True))), [-1, 1])

    def test_qsecfld(self):
        secfld = mpc.SecFld(7**3)
        a = secfld(1)
        b = secfld(0)
        self.assertEqual(mpc.run(mpc.output(+a)), 1)
        self.assertNotEqual(id(a), id(+a))  # NB: +a creates a copy
        self.assertEqual(mpc.run(mpc.output(a + b)), 1)
        self.assertEqual(mpc.run(mpc.output(a * b)), 0)

    def test_bsecfld(self):
        secfld = mpc.SecFld(char=2, min_order=2**8)
        a = secfld(57)
        b = secfld(67)
        self.assertEqual(int(mpc.run(mpc.output(mpc.input(a, 0)))), 57)
        self.assertEqual(int(mpc.run(mpc.output(+a - -a))), 0)
        self.assertEqual(int(mpc.run(mpc.output(a * b))), 137)
        self.assertEqual(int(mpc.run(mpc.output(a * b / a))), 67)
        self.assertEqual(int(mpc.run(mpc.output(a**254 * a))), 1)
        self.assertEqual(int(mpc.run(mpc.output(a & b))), 1)
        self.assertEqual(int(mpc.run(mpc.output(a | b))), 123)
        self.assertEqual(int(mpc.run(mpc.output(a ^ b))), 122)
        self.assertEqual(int(mpc.run(mpc.output(~a))), 198)
        c = mpc.run(mpc.output(mpc.to_bits(secfld(0))))
        self.assertEqual(c, [0, 0, 0, 0, 0, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secfld(1))))
        self.assertEqual(c, [1, 0, 0, 0, 0, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secfld(255))))
        self.assertEqual(c, [1, 1, 1, 1, 1, 1, 1, 1])
        c = mpc.run(mpc.output(mpc.to_bits(secfld(255), 1)))
        self.assertEqual(c, [1])
        c = mpc.run(mpc.output(mpc.to_bits(secfld(255), 4)))
        self.assertEqual(c, [1, 1, 1, 1])
        self.assertEqual(mpc.run(mpc.output(mpc.matrix_sub([[a]], [[a]])[0])), [0])
        self.assertEqual(mpc.run(mpc.output(mpc.matrix_prod([c], [[a]*4], True)[0])), [0])
        self.assertEqual(mpc.run(mpc.output(mpc.matrix_prod([[a]*4], [c], True)[0])), [0])

    def test_secint(self):
        secint = mpc.SecInt()
        a = secint(12)
        b = secint(13)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, 0))), 12)
        self.assertEqual(mpc.run(mpc.output(mpc.input([a, b], 0))), [12, 13])
        self.assertEqual(mpc.run(mpc.output(-a)), -12)
        self.assertEqual(mpc.run(mpc.output(+a)), 12)
        self.assertNotEqual(id(a), id(+a))  # NB: +a creates a copy
        self.assertEqual(mpc.run(mpc.output(a * b + b)), 12 * 13 + 13)
        self.assertEqual(mpc.run(mpc.output((a * b) / b)), 12)
        self.assertEqual(mpc.run(mpc.output((a * b) / 12)), 13)
        self.assertEqual(mpc.run(mpc.output(a**11 * a**-6 * a**-5)), 1)
        self.assertEqual(mpc.run(mpc.output(a**(secint.field.modulus - 1))), 1)
        c = mpc.to_bits(mpc.SecInt(0)(0))  # mpc.output() only works for nonempty lists
        self.assertEqual(c, [])
        c = mpc.run(mpc.output(mpc.to_bits(mpc.SecInt(1)(0))))
        self.assertEqual(c, [0])
        c = mpc.run(mpc.output(mpc.to_bits(mpc.SecInt(1)(1))))
        self.assertEqual(c, [1])
        c = mpc.to_bits(secint(0), 0)  # mpc.output() only works for nonempty lists
        self.assertEqual(c, [])
        c = mpc.run(mpc.output(mpc.to_bits(secint(0))))
        self.assertEqual(c, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secint(1))))
        self.assertEqual(c, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secint(8113))))
        self.assertEqual(c, [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secint(2**31 - 1))))
        self.assertEqual(c, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secint(2**31 - 1), 16)))
        self.assertEqual(c, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        c = mpc.run(mpc.output(mpc.to_bits(secint(-1), 8)))
        self.assertEqual(c, [1, 1, 1, 1, 1, 1, 1, 1])
        c = mpc.run(mpc.output(mpc.to_bits(secint(-2**31))))
        self.assertEqual(c, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        c = mpc.run(mpc.output(mpc.to_bits(secint(-2**31), 16)))
        self.assertEqual(c, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.from_bits(mpc.to_bits(secint(8113)))))
        self.assertEqual(c, 8113)
        c = mpc.run(mpc.output(mpc.from_bits(mpc.to_bits(secint(2**31 - 1)))))
        self.assertEqual(c, 2**31 - 1)
        # TODO: from_bits for negative numbers
        # c = mpc.run(mpc.output(mpc.from_bits(mpc.to_bits(secint(-2**31)))))
        # self.assertEqual(c, -2**31)
        self.assertFalse(mpc.run(mpc.eq_public(secint(4), secint(2))))
        self.assertTrue(mpc.run(mpc.eq_public(secint(42), secint(42))))

        self.assertEqual(mpc.run(mpc.output(abs(secint(1)))), 1)
        self.assertEqual(mpc.run(mpc.output(secint(-2**31) % 2)), 0)
        self.assertEqual(mpc.run(mpc.output(secint(-2**31 + 1) % 2)), 1)
        self.assertEqual(mpc.run(mpc.output(secint(-1) % 2)), 1)
        self.assertEqual(mpc.run(mpc.output(secint(0) % 2)), 0)
        self.assertEqual(mpc.run(mpc.output(secint(1) % 2)), 1)
        self.assertEqual(mpc.run(mpc.output(secint(2**31 - 1) % 2)), 1)
        self.assertEqual(mpc.run(mpc.output(secint(5) % 2)), 1)
        self.assertEqual(mpc.run(mpc.output(secint(-5) % 2)), 1)
        self.assertEqual(mpc.run(mpc.output(secint(50) % 2)), 0)
        self.assertEqual(mpc.run(mpc.output(secint(50) % 4)), 2)
        self.assertEqual(mpc.run(mpc.output(secint(50) % 32)), 18)
        self.assertEqual(mpc.run(mpc.output(secint(-50) % 2)), 0)
        self.assertEqual(mpc.run(mpc.output(secint(-50) % 32)), 14)
        self.assertEqual(mpc.run(mpc.output(secint(5) // 2)), 2)
        self.assertEqual(mpc.run(mpc.output(secint(50) // 2)), 25)
        self.assertEqual(mpc.run(mpc.output(secint(50) // 4)), 12)
        self.assertEqual(mpc.run(mpc.output(secint(11) << 3)), 88)
        self.assertEqual(mpc.run(mpc.output(secint(-11) << 3)), -88)
        self.assertEqual(mpc.run(mpc.output(secint(70) >> 2)), 17)
        self.assertEqual(mpc.run(mpc.output(secint(-70) >> 2)), -18)
        self.assertEqual(mpc.run(mpc.output(secint(50) % 17)), 16)
        self.assertEqual(mpc.run(mpc.output(secint(177) % 17)), 7)
        self.assertEqual(mpc.run(mpc.output(secint(-50) % 17)), 1)
        self.assertEqual(mpc.run(mpc.output(secint(-177) % 17)), 10)
        self.assertEqual(mpc.run(mpc.output(secint(3)**0)), 1)
        self.assertEqual(mpc.run(mpc.output(secint(3)**18)), 3**18)

        self.assertIn(mpc.run(mpc.output(mpc.random_bit(secint))), [0, 1])
        self.assertIn(mpc.run(mpc.output(mpc.random_bit(secint, signed=True))), [-1, 1])

    def test_secfxp(self):
        secfxp = mpc.SecFxp()
        self.assertEqual(mpc.run(mpc.output(mpc.input(secfxp(7.75), senders=0))), 7.75)
        c = mpc.to_bits(secfxp(0), 0)  # mpc.output() only works for nonempty lists
        self.assertEqual(c, [])
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(0))))
        self.assertEqual(c, [0.0]*32)
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(1))))
        self.assertEqual(c, [0.0]*16 + [1.0] + [0.0]*15)
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(0.5))))
        self.assertEqual(c, [0.0]*15 + [1.0] + [0.0]*16)
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(8113))))
        self.assertEqual(c, [0.0]*16 + [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(2**15 - 1))))
        self.assertEqual(c, [0]*16 + [1]*15 + [0])
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(-1))))
        self.assertEqual(c, [0]*16 + [1]*16)
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(-2**15))))
        self.assertEqual(c, [0]*31 + [1])

        for f in [8, 16, 32, 64]:
            secfxp = mpc.SecFxp(2*f)
            c = mpc.run(mpc.output(secfxp(1) + secfxp(1)))
            self.assertEqual(c, 2)
            c = mpc.run(mpc.output(secfxp(2**-f) + secfxp(1)))
            if f != 64:  # NB: 1 + 2**-64 == 1 in Python
                self.assertEqual(c, 1 + 2**-f)
            self.assertEqual(mpc.run(mpc.output(secfxp(0.5) * secfxp(2.0))), 1)
            self.assertEqual(mpc.run(mpc.output(secfxp(2.0) * secfxp(0.5))), 1)
            c = mpc.run(mpc.output(secfxp(2**(f//2 - 1) - 0.5) * secfxp(-2**(f//2) + 0.5)))
            self.assertEqual(c, -2**(f-1) + 1.5 * 2**(f//2 - 1) - 0.25)

            s = [10.75, -3.375, 0.125, -0.125]
            self.assertEqual(mpc.run(mpc.output(list(map(secfxp, s)))), s)

            s = [10.5, -3.25, 0.125, -0.125]
            a, b, c, d = list(map(secfxp, s))
            t = [v * v for v in s]
            self.assertEqual(mpc.run(mpc.output([a*a, b*b, c*c, d*d])), t)
            x = [a, b, c, d]
            self.assertEqual(mpc.run(mpc.output(mpc.schur_prod(x, x))), t)
            self.assertEqual(mpc.run(mpc.output(mpc.schur_prod(x, x[:]))), t)
            t = sum(t)
            self.assertEqual(mpc.run(mpc.output(mpc.in_prod(x, x))), t)
            self.assertEqual(mpc.run(mpc.output(mpc.in_prod(x, x[:]))), t)
            self.assertEqual(mpc.run(mpc.output(mpc.matrix_prod([x], [x], True)[0])), [t])
            u = mpc.unit_vector(secfxp(3), 4)
            self.assertEqual(mpc.run(mpc.output(mpc.matrix_prod([x], [u], True)[0])), [s[3]])
            self.assertEqual(mpc.run(mpc.output(mpc.matrix_prod([u], [x], True)[0])), [s[3]])
            self.assertEqual(mpc.run(mpc.output(mpc.gauss([[a]], b, [a], [b])[0])), [0])
            t = [_ for a, b, c, d in [s] for _ in [a + b, a * b, a - b]]
            self.assertEqual(mpc.run(mpc.output([a + b, a * b, a - b])), t)
            t = [_ for a, b, c, d in [s] for _ in [(a + b)**2, (a + b)**2 + 3*c]]
            self.assertEqual(mpc.run(mpc.output([(a + b)**2, (a + b)**2 + 3*c])), t)
            t = [_ for a, b, c, d in [s] for _ in [a < b, b < c, c < d]]
            self.assertEqual(mpc.run(mpc.output([a < b, b < c, c < d])), t)
            t = s[0] < s[1] and s[1] < s[2]
            self.assertEqual(mpc.run(mpc.output((a < b) & (b < c))), t)
            t = s[0] < s[1] or s[1] < s[2]
            self.assertEqual(mpc.run(mpc.output((a < b) | (b < c))), t)
            t = (int(s[0] < s[1]) ^ int(s[1] < s[2]))
            self.assertEqual(mpc.run(mpc.output((a < b) ^ (b < c))), t)
            t = (int(not s[0] < s[1]) ^ int(s[1] < s[2]))
            self.assertEqual(mpc.run(mpc.output(~(a < b) ^ b < c)), t)
            t = [s[0] > 1, 10*s[1] < 5, 10*s[0] == 5]
            self.assertEqual(mpc.run(mpc.output([a > 1, 10*b < 5, 10*a == 5])), t)

            s[3] = -0.120
            d = secfxp(s[3])
            t = s[3] / 0.25
            self.assertAlmostEqual(mpc.run(mpc.output(d / 0.25)), t, delta=2**(1-f))
            t = round(s[3] / s[2] + s[0])
            self.assertEqual(round(mpc.run(mpc.output(d / c + a))), t)
            t = ((s[0] + s[1])**2 + 3*s[2]) / s[2]
            self.assertAlmostEqual(mpc.run(mpc.output(((a+b)**2+3*c)/c)), t, delta=2**(8-f))
            t = 1 / s[3]
            self.assertAlmostEqual(mpc.run(mpc.output(1 / d)), t, delta=2**(6-f))
            t = s[2] / s[3]
            self.assertAlmostEqual(mpc.run(mpc.output(c / d)), t, delta=2**(3-f))
            t = -s[3] / s[2]
            self.assertAlmostEqual(mpc.run(mpc.output(-d / c)), t, delta=2**(3-f))

            self.assertEqual(mpc.run(mpc.output(mpc.sgn(+a))), s[0] > 0)
            self.assertEqual(mpc.run(mpc.output(mpc.sgn(-a))), -(s[0] > 0))
            self.assertEqual(mpc.run(mpc.output(mpc.sgn(secfxp(0)))), 0)
            self.assertEqual(mpc.run(mpc.output(abs(secfxp(-1.5)))), 1.5)

            self.assertEqual(mpc.run(mpc.output(mpc.min(a, b, c, d))), min(s))
            self.assertEqual(mpc.run(mpc.output(mpc.min(a, 0))), min(s[0], 0))
            self.assertEqual(mpc.run(mpc.output(mpc.min(0, b))), min(0, s[1]))
            self.assertEqual(mpc.run(mpc.output(mpc.max(a, b, c, d))), max(s))
            self.assertEqual(mpc.run(mpc.output(mpc.max(a, 0))), max(s[0], 0))
            self.assertEqual(mpc.run(mpc.output(mpc.max(0, b))), max(0, s[1]))
            self.assertEqual(mpc.run(mpc.output(list(mpc.min_max(a, b, c, d)))), [min(s), max(s)])
            self.assertEqual(mpc.run(mpc.output(mpc.argmin([a, b, c, d])[0])), 1)
            self.assertEqual(mpc.run(mpc.output(mpc.argmin([a, b], key=operator.neg)[1])), max(s))
            self.assertEqual(mpc.run(mpc.output(mpc.argmax([a, b, c, d])[0])), 0)
            self.assertEqual(mpc.run(mpc.output(mpc.argmax([a, b], key=operator.neg)[1])), min(s))

            self.assertEqual(mpc.run(mpc.output(secfxp(5) % 2)), 1)
            self.assertEqual(mpc.run(mpc.output(secfxp(1) % 2**(1-f))), 0)
            self.assertEqual(mpc.run(mpc.output(secfxp(2**-f) % 2**(1-f))), 2**-f)
            self.assertEqual(mpc.run(mpc.output(secfxp(2*2**-f) % 2**(1-f))), 0)
            self.assertEqual(mpc.run(mpc.output(secfxp(1) // 2**(1-f))), 2**(f-1))
            self.assertEqual(mpc.run(mpc.output(secfxp(27.0) % 7.0)), 6.0)
            self.assertEqual(mpc.run(mpc.output(secfxp(-27.0) // 7.0)), -4.0)
            self.assertEqual(mpc.run(mpc.output(list(divmod(secfxp(27.0), 6.0)))), [4.0, 3.0])
            self.assertEqual(mpc.run(mpc.output(secfxp(21.5) % 7.5)), 6.5)
            self.assertEqual(mpc.run(mpc.output(secfxp(-21.5) // 7.5)), -3.0)
            self.assertEqual(mpc.run(mpc.output(list(divmod(secfxp(21.5), 0.5)))), [43.0, 0.0])

    def test_secflt(self):
        secflt = mpc.SecFlt()
        a = secflt(1.25)
        b = secflt(2.5)
        self.assertEqual(mpc.run(mpc.output(mpc.input(a, 0))), 1.25)
        self.assertEqual(mpc.run(mpc.output(a + b)), 3.75)
        self.assertEqual(mpc.run(mpc.output(-a + -b)), -3.75)
        self.assertEqual(mpc.run(mpc.output(a * 2**10 + 2**10 * b)), 3.75 * 2**10)
        self.assertEqual(mpc.run(mpc.output(-b + b)), 0)
        self.assertEqual(mpc.run(mpc.output(abs(1.25 - +b))), 1.25)
        self.assertEqual(mpc.run(mpc.output(a * b)), 1.25 * 2.5)
        self.assertAlmostEqual(mpc.run(mpc.output(a / b)), 0.5, delta=2**-21)
        self.assertTrue(mpc.run(mpc.output(a < b)))
        self.assertTrue(mpc.run(mpc.output(a <= b)))
        self.assertFalse(mpc.run(mpc.output(a == b)))
        self.assertFalse(mpc.run(mpc.output(a >= b)))
        self.assertFalse(mpc.run(mpc.output(a > b)))
        self.assertTrue(mpc.run(mpc.output(a != b)))
        self.assertFalse(mpc.run(mpc.eq_public(a, b)))
        self.assertTrue(mpc.run(mpc.eq_public(a, a)))
        phi = secflt((math.sqrt(5) + 1) / 2)
        self.assertAlmostEqual(mpc.run(mpc.output(phi**2 - phi - 1)), 0, delta=2**-21)

        @mpc.coroutine
        async def nop(a) -> secflt:
            return a
        self.assertEqual(mpc.run(mpc.output(nop(a))), 1.25)

    def test_if_else_if_swap(self):
        secfld = mpc.SecFld()
        a = secfld(0)
        b = secfld(1)
        c = secfld(1)
        self.assertEqual(mpc.run(mpc.output(c.if_else(a, b))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.if_swap(1-c, a, b))), [0, 1])
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(c, [a, b], [b, a]))), [0, 1])
        self.assertEqual(mpc.run(mpc.output((1-c).if_swap(a, b))), [0, 1])
        secint = mpc.SecInt()
        a = secint(-1)
        b = secint(1)
        c = secint(1)
        self.assertEqual(mpc.run(mpc.output(c.if_swap(a, b))), [1, -1])
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(1-c, a, b))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.if_swap(c, a, b))), [1, -1])
        self.assertEqual(mpc.run(mpc.output((1-c).if_else([a, b], [b, a]))), [1, -1])
        secfxp = mpc.SecFxp()
        a = secfxp(-1.0)
        b = secfxp(1.0)
        c = secfxp(1)
        self.assertEqual(mpc.run(mpc.output(c.if_else([a, a], [b, b]))), [-1.0, -1.0])
        self.assertEqual(mpc.run(mpc.output(mpc.if_swap(1-c, a, b))), [-1.0, 1.0])
        self.assertEqual(mpc.run(mpc.output(mpc.if_swap(c, 0.0, 1.0))), [1.0, 0.0])
        self.assertEqual(mpc.run(mpc.output((1-c).if_else(0.0, 1.0))), 1.0)

    def test_convert(self):
        secint = mpc.SecInt()
        secint8 = mpc.SecInt(8)
        secint16 = mpc.SecInt(16)
        secfld257 = mpc.SecFld(257)
        secfld263 = mpc.SecFld(263)
        secfld37s = mpc.SecFld(37, signed=True)
        secfld47s = mpc.SecFld(47, signed=True)
        secfxp = mpc.SecFxp()
        secfxp16 = mpc.SecFxp(16)

        x = [secint8(-100), secint8(100)]
        y = mpc.convert(x, secint)
        self.assertEqual(mpc.run(mpc.output(y)), [-100, 100])
        y = mpc.convert(y, secint8)
        self.assertEqual(mpc.run(mpc.output(y)), [-100, 100])

        x = [secint16(i) for i in range(10)]
        y = mpc.convert(x, secfld257)
        self.assertEqual(mpc.run(mpc.output(y)), list(range(10)))

        x = [secfld257(i) for i in range(10)]
        y = mpc.convert(x, secfld263)
        self.assertEqual(mpc.run(mpc.output(y)), list(range(10)))

        x = [secfld47s(i) for i in range(-5, 10)]
        y = mpc.convert(x, secfld37s)
        self.assertEqual(mpc.run(mpc.output(y)), list(range(-5, 10)))
        y = mpc.convert(y, secfxp)
        self.assertEqual(mpc.run(mpc.output(y)), list(range(-5, 10)))

        x = [secint(-100), secint(100)]
        y = mpc.convert(x, secfxp)
        self.assertEqual(mpc.run(mpc.output(y)), [-100, 100])
        y = mpc.convert(y, secint)
        self.assertEqual(mpc.run(mpc.output(y)), [-100, 100])

        x = [secfxp16(-100.25), secfxp16(100.875)]
        y = mpc.convert(x, secfxp)
        self.assertEqual(mpc.run(mpc.output(y)), [-100.25, 100.875])
        y = mpc.convert(y, secfxp16)
        self.assertEqual(mpc.run(mpc.output(y)), [-100.25, 100.875])

    def test_empty_input(self):
        secint = mpc.SecInt()
        self.assertEqual(mpc.run(mpc.gather([])), [])
        self.assertEqual(mpc.run(mpc.output([])), [])
        self.assertEqual(mpc._reshare([]), [])
        self.assertEqual(mpc.convert([], None), [])
        self.assertEqual(mpc.sum([]), 0)
        self.assertEqual(mpc.sum([], start=1), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.sum([], start=secint(1)))), 1)
        self.assertEqual(mpc.prod([]), 1)
        self.assertEqual(mpc.all([]), 1)
        self.assertEqual(mpc.any([]), 0)
        self.assertEqual(mpc.sorted([]), [])
        self.assertEqual(mpc.in_prod([], []), 0)
        self.assertEqual(mpc.vector_add([], []), [])
        self.assertEqual(mpc.vector_sub([], []), [])
        self.assertEqual(mpc.scalar_mul(secint(0), []), [])
        self.assertEqual(mpc.schur_prod([], []), [])
        self.assertEqual(mpc.from_bits([]), 0)

    def test_errors(self):
        secfxp = mpc.SecFxp()
        self.assertRaises(ValueError, mpc.all, [secfxp(0.5)])
        self.assertRaises(ValueError, mpc.any, [secfxp(0.5)])
        self.assertRaises(ValueError, mpc.min, [])
        self.assertRaises(ValueError, mpc.max, [])
        self.assertRaises(ValueError, mpc.min_max, [])
        self.assertRaises(ValueError, mpc.argmin, [])
        self.assertRaises(ValueError, mpc.argmax, [])
        self.assertRaises(ValueError, mpc.if_else, secfxp(1.5), [0], [0])
        self.assertRaises(ValueError, mpc.if_swap, secfxp(1.5), [0], [0])
        self.assertRaises(ValueError, mpc.unit_vector, secfxp(1.5), 2)

    def test_misc(self):
        secint = mpc.SecInt()
        secfxp = mpc.SecFxp()
        secfld = mpc.SecFld()
        for secnum in (secint, secfxp, secfld):
            self.assertEqual(type(mpc.run(mpc.output(secnum(0), raw=True))), secnum.field)
            self.assertEqual(mpc.run(mpc.output(mpc._reshare([secnum(0)]))), [0])
            self.assertEqual(mpc.run(mpc.output(mpc.all(secnum(1) for _ in range(5)))), True)
            self.assertEqual(mpc.run(mpc.output(mpc.all([secnum(1), secnum(1), secnum(0)]))), False)
            self.assertEqual(mpc.run(mpc.output(mpc.any(secnum(0) for _ in range(5)))), False)
            self.assertEqual(mpc.run(mpc.output(mpc.any([secnum(0), secnum(1), secnum(1)]))), True)
            self.assertEqual(mpc.run(mpc.output(mpc.sum([secnum(1)], start=1))), 2)
            self.assertEqual(mpc.run(mpc.output(mpc.prod([secnum(1)], start=1))), 1)
            self.assertEqual(mpc.run(mpc.output(mpc.sum([secnum(1)], start=secnum(1)))), 2)
            self.assertEqual(mpc.run(mpc.output(mpc.find([secnum(1)], 0, e=-1))), -1)
            self.assertEqual(mpc.run(mpc.output(mpc.find([secnum(1)], 1))), 0)
            self.assertEqual(mpc.run(mpc.output(mpc.find([secnum(1)], 1, f=lambda i: i))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.min(secint(i) for i in range(-1, 2, 1)))), -1)
        self.assertEqual(mpc.run(mpc.output(mpc.argmin(secint(i) for i in range(-1, 2, 1))[0])), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.max(secfxp(i) for i in range(-1, 2, 1)))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.argmax(secfxp(i) for i in range(-1, 2, 1))[0])), 2)
        self.assertEqual(mpc.run(mpc.output(list(mpc.min_max(map(secfxp, range(5)))))), [0, 4])
        x = (secint(i) for i in range(-3, 3))
        s = [0, -1, 1, -2, 2, -3]
        self.assertEqual(mpc.run(mpc.output(mpc.sorted(x, key=lambda a: a*(2*a+1)))), s)
        x = (secfxp(i) for i in range(5))
        self.assertEqual(mpc.run(mpc.output(mpc.sorted(x, reverse=True))), [4, 3, 2, 1, 0])
        self.assertEqual(mpc.run(mpc.output(mpc.sum(map(secint, range(5))))), 10)
        self.assertEqual(mpc.run(mpc.output(mpc.sum([secfxp(2.75)], start=3.125))), 5.875)
        self.assertEqual(int(mpc.run(mpc.output(mpc.prod(map(secfxp, range(1, 5)))))), 24)
        self.assertEqual(int(mpc.run(mpc.output(mpc.prod([secfxp(1.414214)]*4)))), 4)
        self.assertEqual(mpc.find([], 0), 0)
        self.assertEqual(mpc.find([], 0, e=None), (1, 0))
        self.assertEqual(mpc.run(mpc.output(list(mpc.find([secfld(1)], 1, e=None)))), [0, 0])
        self.assertEqual(mpc.run(mpc.output(mpc.find([secfld(2)], 2, bits=False))), 0)
        x = [secint(i) for i in range(5)]
        f = lambda i: [i**2, 3**i]
        self.assertEqual(mpc.run(mpc.output(mpc.find(x, 2, bits=False, f=f))), [4, 9])
        cs_f = lambda b, i: [b * (2*i+1) + i**2, (b*2+1) * 3**i]
        self.assertEqual(mpc.run(mpc.output(mpc.find(x, 2, bits=False, cs_f=cs_f))), [4, 9])


if __name__ == "__main__":
    unittest.main()

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

    def test_psecfld(self):
        secfld = mpc.SecFld(l=16)
        a = secfld(1)
        b = secfld(0)
        self.assertEqual(mpc.run(mpc.output(a + b)), 1)
        self.assertEqual(mpc.run(mpc.output(a * b)), 0)

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
        for a, b in [(secfld(1), secfld(1)), (1, secfld(1)), (secfld(1), 1)]:
            with self.assertRaises(TypeError):
                a < b
            with self.assertRaises(TypeError):
                a <= b
            with self.assertRaises(TypeError):
                a > b
            with self.assertRaises(TypeError):
                a >= b
            with self.assertRaises(TypeError):
                a // b
            with self.assertRaises(TypeError):
                a % b
            with self.assertRaises(TypeError):
                divmod(a, b)
        b = mpc.random_bit(secfld)
        self.assertIn(mpc.run(mpc.output(b)), [0, 1])
        b = mpc.random_bit(secfld, signed=True)
        self.assertIn(mpc.run(mpc.output(b)), [-1, 1])

    def test_bsecfld(self):
        secfld = mpc.SecFld(char2=True, l=8)
        a = secfld(57)
        b = secfld(67)
        c = mpc.run(mpc.output(mpc.input(a, 0)))
        self.assertEqual(c.value.value, 57)
        c = mpc.run(mpc.output(a - (-a)))
        self.assertEqual(c.value.value, 0)
        c = mpc.run(mpc.output(a * b))
        self.assertEqual(c.value.value, 137)
        c = mpc.run(mpc.output(c / a))
        self.assertEqual(c.value.value, 67)
        c = mpc.run(mpc.output(a**254 * a))
        self.assertEqual(c.value.value, 1)
        c = mpc.run(mpc.output(a & b))
        self.assertEqual(c.value.value, 1)
        c = mpc.run(mpc.output(a | b))
        self.assertEqual(c.value.value, 123)
        c = mpc.run(mpc.output(a ^ b))
        self.assertEqual(c.value.value, 122)
        c = mpc.run(mpc.output(~a))
        self.assertEqual(c.value.value, 198)
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

    def test_secint(self):
        secint = mpc.SecInt()
        a = secint(12)
        b = secint(13)
        c = mpc.run(mpc.output(mpc.input(a, 0)))
        self.assertEqual(c, 12)
        c = mpc.run(mpc.output(mpc.input([a, b], 0)))
        self.assertEqual(c, [12, 13])
        c = mpc.run(mpc.output(a * b + b))
        self.assertEqual(c, 12 * 13 + 13)
        c = mpc.run(mpc.output((a * b) / b))
        self.assertEqual(c, 12)
        c = mpc.run(mpc.output((a * b) / 12))
        self.assertEqual(c, 13)
        c = mpc.run(mpc.output(a**11 * a**(-6) * a**(-5)))
        self.assertEqual(c, 1)
        c = mpc.run(mpc.output(a**(secint.field.modulus - 1)))
        self.assertEqual(c, 1)
        self.assertEqual(mpc.run(mpc.output(secint(12)**73)), 12**73)
        c = mpc.to_bits(mpc.SecInt(0)(0))  # mpc.output() only works for non-empty lists
        self.assertEqual(c, [])
        c = mpc.run(mpc.output(mpc.to_bits(mpc.SecInt(1)(0))))
        self.assertEqual(c, [0])
        c = mpc.run(mpc.output(mpc.to_bits(mpc.SecInt(1)(1))))
        self.assertEqual(c, [1])
        c = mpc.to_bits(secint(0), 0)  # mpc.output() only works for non-empty lists
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

        self.assertEqual(mpc.run(mpc.output(secint(3)**73)), 3**73)
        b = mpc.random_bit(secint)
        self.assertIn(mpc.run(mpc.output(b)), [0, 1])
        b = mpc.random_bit(secint, signed=True)
        self.assertIn(mpc.run(mpc.output(b)), [-1, 1])

    def test_secfxp(self):
        secfxp = mpc.SecFxp()
        c = mpc.to_bits(secfxp(0), 0)  # mpc.output() only works for non-empty lists
        self.assertEqual(c, [])
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(0))))
        self.assertEqual(c, [0.0] * 32)
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(1))))
        self.assertEqual(c, [0.0] * 16 + [1.0] + [0.0] * 15)
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(8113))))
        self.assertEqual(c, [0.0] * 16 + [float(b) for b in
                                          [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(2**15 - 1))))
        self.assertEqual(c, [float(b) for b in [0] * 16 + [1] * 15 + [0]])
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(-1))))
        self.assertEqual(c, [float(b) for b in [0] * 16 + [1] * 16])
        c = mpc.run(mpc.output(mpc.to_bits(secfxp(-2**15))))
        self.assertEqual(c, [float(b) for b in [0] * 31 + [1]])

        for f in [8, 16, 32, 64]:
            secfxp = mpc.SecFxp(2*f)
            d = mpc.run(mpc.output(secfxp(1) + secfxp(1)))
            self.assertEqual(d.frac_length, f)
            self.assertEqual(d, 2)
            d = mpc.run(mpc.output(secfxp(2**-f) + secfxp(1)))
            f == 64 or self.assertEqual(d, 1 + 2**-f)  # NB: 1 + 2**-64 == 1 in Python
            d = mpc.run(mpc.output(secfxp(0.5) * secfxp(2.0)))
            self.assertEqual(d, 1)
            d = mpc.run(mpc.output(secfxp(2.0) * secfxp(0.5)))
            self.assertEqual(d, 1)

            s = [10.7, -3.4, 0.1, -0.11]
            self.assertEqual(mpc.run(mpc.output(list(map(secfxp, s)))), s)

            s = [10.5, -3.25, 0.125, -0.125]
            x, y, z, w = list(map(secfxp, s))
            s2 = [v*v for v in s]
            self.assertEqual(mpc.run(mpc.output([x*x, y*y, z*z, w*w])), s2)
            self.assertEqual(mpc.run(mpc.output(mpc.schur_prod([x, y, z, w], [x, y, z, w]))), s2)
            s2 = [_ for x, y, z, w in [s] for _ in [x+y, x*y, x-y]]
            self.assertEqual(mpc.run(mpc.output([x+y, x*y, x-y])), s2)
            s2 = [_ for x, y, z, w in [s] for _ in [(x+y)**2, (x+y)**2 + 3*z]]
            self.assertEqual(mpc.run(mpc.output([(x+y)**2, (x+y)**2 + 3*z])), s2)
            s2 = [int(_) for x, y, z, w in [s] for _ in [x < y, y < z, z < w]]
            self.assertEqual(mpc.run(mpc.output([x < y, y < z, z < w])), s2)
            s2 = int(s[0] < s[1] and s[1] < s[2])
            self.assertEqual(mpc.run(mpc.output((x < y) & (y < z))), s2)
            s2 = int(s[0] < s[1] or s[1] < s[2])
            self.assertEqual(mpc.run(mpc.output((x < y) | (y < z))), s2)
            s2 = (int(s[0] < s[1]) ^ int(s[1] < s[2]))
            self.assertEqual(mpc.run(mpc.output((x < y) ^ (y < z))), s2)
            s2 = (int(not s[0] < s[1]) ^ int(s[1] < s[2]))
            self.assertEqual(mpc.run(mpc.output(~(x < y) ^ y < z)), s2)
            s2 = [int(s[0] < 1), int(10*s[1] < 5), int(10*s[0] == 5)]
            self.assertEqual(mpc.run(mpc.output([x < 1, 10 * y < 5, 10 * x == 5])), s2)

            s[3] = -0.120
            w = secfxp(s[3])
            for _ in range(3):
                s2 = s[3]/s[2] + s[0]
                self.assertAlmostEqual(mpc.run(mpc.output(w / z + x)).signed(), s2, delta=1)
                ss2 = round(s2 * (1 << f))
                self.assertAlmostEqual(mpc.run(mpc.output(w / z + x)), ss2, delta=1)
                s2 = ((s[0]+s[1])**2 + 3*s[2])/s[2]
                self.assertAlmostEqual(mpc.run(mpc.output(((x + y)**2 + 3 * z) / z)).signed(), s2, delta=2)
                s2 = 1/s[3]
                self.assertAlmostEqual((mpc.run(mpc.output(1 / w))).signed(), s2, delta=1)
                s2 = s[2]/s[3]
                self.assertAlmostEqual(mpc.run(mpc.output(z / w)).signed(), s2, delta=1)
                s2 = -s[3]/s[2]
                ss2 = round(s2 * (1 << f))
                self.assertAlmostEqual(mpc.run(mpc.output(-w / z)), ss2, delta=1)
                s2 = s[2]/s[3]
                ss2 = round(s2 * (1 << f))
                self.assertAlmostEqual(mpc.run(mpc.output(w / z)), ss2, delta=1)

            self.assertEqual(mpc.run(mpc.output(mpc.sgn(x))), int(s[0] > 0))
            self.assertEqual(mpc.run(mpc.output(mpc.sgn(-x))), -int(s[0] > 0))
            self.assertEqual(mpc.run(mpc.output(mpc.sgn(secfxp(0)))), 0)

            self.assertEqual(mpc.run(mpc.output(mpc.min(x, y, w, z))), min(s))
            self.assertEqual(mpc.run(mpc.output(mpc.min(x, 0))), min(s[0], 0))
            self.assertEqual(mpc.run(mpc.output(mpc.min(0, y))), min(0, s[1]))
            self.assertEqual(mpc.run(mpc.output(mpc.max(x, y, w, z))), max(s))
            self.assertEqual(mpc.run(mpc.output(mpc.max(x, 0))), max(s[0], 0))
            self.assertEqual(mpc.run(mpc.output(mpc.max(0, y))), max(0, s[1]))

            self.assertEqual(mpc.run(mpc.output(secfxp(5) % 2)), 1*(2**f))
            self.assertEqual(mpc.run(mpc.output(secfxp(1) % 2**(1-f))), 0*(2**f))
            self.assertEqual(mpc.run(mpc.output(secfxp(1/2**f) % 2**(1-f))), 1)
            self.assertEqual(mpc.run(mpc.output(secfxp(2/2**f) % 2**(1-f))), 0)
            self.assertEqual(mpc.run(mpc.output(secfxp(1) // 2**(1-f))), 2**(f-1))
            self.assertEqual(mpc.run(mpc.output(secfxp(27.0) % 7.0)), 6.0)
            self.assertEqual(mpc.run(mpc.output(secfxp(-27.0) // 7.0)), -4.0)
            self.assertEqual(mpc.run(mpc.output(list(divmod(secfxp(27.0), 6.0)))), [4.0, 3.0])
            self.assertEqual(mpc.run(mpc.output(secfxp(21.5) % 7.5)), 6.5)
            self.assertEqual(mpc.run(mpc.output(secfxp(-21.5) // 7.5)), -3.0)
            self.assertEqual(mpc.run(mpc.output(list(divmod(secfxp(21.5), 0.5)))), [43.0, 0.0])

    def test_if_else(self):
        secfld = mpc.SecFld()
        a = secfld(0)
        b = secfld(1)
        c = secfld(1)
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(c, a, b))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(1 - c, a, b))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(c, [a, b], [b, a]))), [0, 1])
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(1 - c, [a, b], [b, a]))), [1, 0])
        secint = mpc.SecInt()
        a = secint(-1)
        b = secint(1)
        c = secint(1)
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(c, a, b))), -1)
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(1 - c, a, b))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(c, [a, b], [b, a]))), [-1, 1])
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(1 - c, [a, b], [b, a]))), [1, -1])
        secfxp = mpc.SecFxp()
        a = secfxp(-1.0)
        b = secfxp(1.0)
        c = secfxp(1)
        self.assertEqual(float(mpc.run(mpc.output(mpc.if_else(c, a, b)))), -1.0)
        self.assertEqual(float(mpc.run(mpc.output(mpc.if_else(1 - c, a, b)))), 1.0)
        self.assertEqual(float(mpc.run(mpc.output(mpc.if_else(c, 0.0, 1.0)))), 0.0)
        self.assertEqual(float(mpc.run(mpc.output(mpc.if_else(1 - c, 0.0, 1.0)))), 1.0)

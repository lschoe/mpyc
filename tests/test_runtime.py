import unittest
from mpyc.runtime import mpc

class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mpc.start()

    @classmethod
    def tearDownClass(cls):
        mpc.shutdown()

    def test_secfld(self):
        secfld = mpc.SecFld()
        self.assertEqual(secfld.field.modulus, 2)
        secfld = mpc.SecFld(l=16)
        a = secfld(1)
        b = secfld(0)
        self.assertEqual(mpc.run(mpc.output(a + b)), 1)
        self.assertEqual(mpc.run(mpc.output(a * b)), 0)

        secfld = mpc.SecFld(101)
        a = secfld(1)
        b = secfld(-1)
        self.assertEqual(mpc.run(mpc.output(a + b)), 0)
        self.assertEqual(mpc.run(mpc.output(a * b)), 100)
        self.assertEqual(mpc.run(mpc.output(a == b)), 0)
        self.assertEqual(mpc.run(mpc.output(a == -b)), 1)
        self.assertEqual(mpc.run(mpc.output(a**2 == b**2)), 1)
        self.assertEqual(mpc.run(mpc.output(a != b)), 1)
        with self.assertRaises(TypeError):
            a < b
        with self.assertRaises(TypeError):
            a <= b
        with self.assertRaises(TypeError):
            a > b
        with self.assertRaises(TypeError):
            a >= b

    def test_secint(self):
        secint = mpc.SecInt()
        a = secint(12)
        b = secint(13)
        c = mpc.run(mpc.output(a * b + b))
        self.assertEqual(c, 12 * 13 + 13)
        c = mpc.run(mpc.output(a**11 * a**(-6) * a**(-5)))
        self.assertEqual(c, 1)
        c = mpc.run(mpc.output(a**(secint.field.modulus - 1)))
        self.assertEqual(c, 1)
        self.assertEqual(mpc.run(mpc.output(secint(12)**73)), 12**73)
        c = mpc.to_bits(mpc.SecInt(0)(0)) # mpc.output() only works for non-empty lists
        self.assertEqual(c, [])
        c = mpc.run(mpc.output(mpc.to_bits(mpc.SecInt(1)(0))))
        self.assertEqual(c, [0])
        c = mpc.run(mpc.output(mpc.to_bits(mpc.SecInt(1)(1))))
        self.assertEqual(c, [1])
        c = mpc.run(mpc.output(mpc.to_bits(secint(0))))
        self.assertEqual(c, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secint(1))))
        self.assertEqual(c, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secint(8113))))
        self.assertEqual(c, [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secint(2**31 - 1))))
        self.assertEqual(c, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        c = mpc.run(mpc.output(mpc.to_bits(secint(-1))))
        self.assertEqual(c, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        c = mpc.run(mpc.output(mpc.to_bits(secint(-2**31))))
        self.assertEqual(c, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(-2**31)))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(-2**31 + 1)))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(-1)))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(0)))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(1)))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(2**31 - 1)))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(5)))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(-5)))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(50)))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.lsb(secint(-50)))), 0)

        self.assertEqual(mpc.run(mpc.output(secint(3)**73)), 3**73)

    def test_secfxp(self):
        secfxp = mpc.SecFxp()
        f = 16
        d = mpc.run(mpc.output(secfxp(1) + secfxp(1)))
        self.assertEqual(d.frac_length, f)
        self.assertEqual(d, 2 * 2**f)
        d = mpc.run(mpc.output(secfxp(2**-f) + secfxp(1)))
        self.assertEqual(d, 1 + 2**f)

        s = [10.7, -3.4, 0.1, -0.11]
        ss = [round(v * (1 << f)) for v in s]
        self.assertEqual(mpc.run(mpc.output(list(map(secfxp, s)))), ss)

        s = [10.5, -3.25, 0.125, -0.125]
        x, y, z, w = list(map(secfxp, s))
        s2 = [v*v for v in s]
        ss2 = [round(v * (1 << f)) for v in s2]
        self.assertEqual(mpc.run(mpc.output([x*x, y*y, z*z, w*w])), ss2)
        s2 = [s[0]+s[1], s[0]*s[1], s[0]-s[1]]
        ss2 = [round(v * (1 << f)) for v in s2]
        self.assertEqual(mpc.run(mpc.output([x+y, x*y, x-y])), ss2)
        s2 = [(s[0]+s[1])**2, (s[0]+s[1])**2 + 3*s[2]]
        ss2 = [round(v * (1 << f)) for v in s2]
        self.assertEqual(mpc.run(mpc.output([(x+y)**2, (x+y)**2 + 3*z])), ss2)

        ss2 = [int(s[i] < s[i+1]) * (1 << f) for i in range(len(s)-1)]
        self.assertEqual(mpc.run(mpc.output([x < y, y < z, z < w])), ss2)
        ss2 = int(s[0] < s[1] and s[1] < s[2]) * (1 << f)
        self.assertEqual(mpc.run(mpc.output(x < y & y < z)), ss2)
        ss2 = int(s[0] < s[1] or s[1] < s[2]) * (1 << f)
        self.assertEqual(mpc.run(mpc.output(x < y | y < z)), ss2)
        ss2 = (int(s[0] < s[1]) ^ int(s[1] < s[2])) * (1 << f)
        self.assertEqual(mpc.run(mpc.output(x < y ^ y < z)), ss2)
        ss2 = (int(not s[0] < s[1]) ^ int(s[1] < s[2])) * (1 << f)
        self.assertEqual(mpc.run(mpc.output(~(x < y) ^ y < z)), ss2)
        s2 = [s[0] < 1, 10*s[1] < 5, 10*s[0] == 5]
        ss2 = [round(v * (1 << f)) for v in s2]
        self.assertEqual(mpc.run(mpc.output([x < 1, 10 * y < 5, 10 * x == 5])), ss2)

        s[3] = -0.120
        w = secfxp(s[3])
        for _ in range(3):
            s2 = s[3]/s[2] + s[ 0]
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

        a = mpc._norm(w)

        self.assertEqual(mpc.run(mpc.output(mpc.sgn(x))), int(s[0] > 0) * (1 << f))
        self.assertEqual(mpc.run(mpc.output(mpc.sgn(-x))), -int(s[0] > 0) * (1 << f))
        self.assertEqual(mpc.run(mpc.output(mpc.sgn(secfxp(0)))), 0)

        ss2 = round(min(s) * (1 << f))
        self.assertEqual(mpc.run(mpc.output(mpc.min(x, y, w, z))), ss2)
        ss2 = round(min(s[0], 0) * (1 << f))
        self.assertEqual(mpc.run(mpc.output(mpc.min(x, 0))), ss2)
        ss2 = round(min(0, s[1]) * (1 << f))
        self.assertEqual(mpc.run(mpc.output(mpc.min(0, y))), ss2)
        ss2 = round(max(s) * (1 << f))
        self.assertEqual(mpc.run(mpc.output(mpc.max(x, y, w, z))), ss2)
        ss2 = round(max(s[0], 0) * (1 << f))
        self.assertEqual(mpc.run(mpc.output(mpc.max(x, 0))), ss2)
        ss2 = round(max(0, s[1]) * (1 << f))
        self.assertEqual(mpc.run(mpc.output(mpc.max(0, y))), ss2)

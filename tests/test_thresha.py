import unittest
from mpyc import pfield
from mpyc import thresha

class Arithmetic(unittest.TestCase):

    def setUp(self):
        self.f2 = pfield.GF(2)
        self.f19 = pfield.GF(19)

    def test_secretsharing(self):
        field = self.f2
        t = 0
        m = 1
        a = [field(0), field(1)]
        shares = thresha.random_split(a, t, m)
        b = thresha.recombine(field, [(j + 1, shares[j]) for j in range(len(shares))])
        self.assertEqual(a, b)

        field = self.f19
        for t in range(8):
            m = 2 * t + 1
            for i in range(t):
                a = [field(i), field(-i), field(i**2), field(-i**2)]
                shares = thresha.random_split(a, t, m)
                b = thresha.recombine(field, [(j + 1, shares[j]) for j in range(len(shares))])
                self.assertEqual(a, b)
        m = 17
        for t in range((m + 1) // 2):
            for i in range(t):
                a = [field(i), field(-i), field(i**2), field(-i**2)]
                shares = thresha.random_split(a, t, m)
                b = thresha.recombine(field, [(j + 1, shares[j]) for j in range(len(shares))])
                self.assertEqual(a, b)

    def test_prf(self):
        key = b'00112233445566778899aabbccddeeff'
        max = 100
        F = thresha.PRF(key, max)
        x = ''
        y = F(x)
        self.assertTrue(0 <= y < max)
        y2 = F(x)
        self.assertEqual(y, y2)

    def test_prss(self):
        field = self.f19
        key = b'00112233445566778899aabbccddeeff'
        max = field.modulus
        F = thresha.PRF(key, max)
        m = 1
        id = 0
        prfs = {frozenset([0]): F}
        uci = 'test uci'
        n = 8
        a = F(uci, n)
        shares = thresha.pseudorandom_share(field, m, id, prfs, uci, n)
        b = thresha.recombine(field, [(1, shares)])
        self.assertEqual(a, b)
        a = [0] * n
        shares = thresha.pseudorandom_share_zero(field, m, id, prfs, uci, n)
        b = thresha.recombine(field, [(1, shares)])
        self.assertEqual(a, b)

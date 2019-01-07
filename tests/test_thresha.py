import unittest
from mpyc import bfield
from mpyc import pfield
from mpyc import thresha

class Arithmetic(unittest.TestCase):

    def setUp(self):
        self.f2 = pfield.GF(2)
        self.f19 = pfield.GF(19)
        self.f256 = bfield.GF(283)

    def test_secretsharing(self):
        for field in (self.f2, self.f256):
            t = 0
            m = 1
            a = [field(0), field(1)]
            shares = thresha.random_split(a, t, m)
            b = thresha.recombine(field, [(j + 1, shares[j]) for j in range(len(shares))])
            self.assertEqual(a, b)

        for field in (self.f19, self.f256):
            for t in range(8):
                m = 2 * t + 1
                for i in range(t):
                    a = [field(i), field(i+1), field(i**2), field((i+1)**2)]
                    shares = thresha.random_split(a, t, m)
                    b = thresha.recombine(field, [(j + 1, shares[j]) for j in range(len(shares))])
                    self.assertEqual(a, b)
            m = 17
            for t in range((m + 1) // 2):
                for i in range(t):
                    a = [field(i), field(i+1), field(i**2), field((i+1)**2)]
                    shares = thresha.random_split(a, t, m)
                    b = thresha.recombine(field, [(j + 1, shares[j]) for j in range(len(shares))])
                    self.assertEqual(a, b)

    def test_prf(self):
        key = b'00112233445566778899aabbccddeeff'
        bound = 100
        F = thresha.PRF(key, bound)
        x = ''
        y = F(x.encode())
        self.assertTrue(0 <= y < bound)
        y2 = F(x.encode())
        self.assertEqual(y, y2)

    def test_prss(self):
        field = self.f256
        key = b'00112233445566778899aabbccddeeff'
        bound = 256 #field.modulus
        F = thresha.PRF(key, bound)
        m = 1
        pid = 0
        prfs = {frozenset([0]): F}
        uci = 'test uci'
        n = 8
        a = F(uci.encode(), n)
        shares = thresha.pseudorandom_share(field, m, pid, prfs, uci, n)
        b = thresha.recombine(field, [(1, shares)])
        self.assertEqual(a, [s.value for s in b])
        a = [0] * n
        shares = thresha.pseudorandom_share_zero(field, m, pid, prfs, uci, n)
        b = thresha.recombine(field, [(1, shares)])
        self.assertEqual(a, b)

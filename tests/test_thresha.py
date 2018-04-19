import unittest
from mpyc import pfield, thresha

class Arithmetic(unittest.TestCase):

    def setUp(self):
        self.f2 = pfield.GF(2)
        self.f19 = pfield.GF(19)

    def test_secretsharing(self):
        field = self.f2
        t = 0
        n = 1
        a = [field(0), field(1)]
        shares = thresha.random_split(a, t, n)
        b = thresha.recombine(field, [(j + 1, shares[j]) for j in range(len(shares))])
        self.assertEqual(a, b)

        field = self.f19
        for t in range(8):
            n = 2 * t + 1
            for i in range(t):
                a = [field(i), field(-i), field(i**2), field(-i**2)]
                shares = thresha.random_split(a, t, n)
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
        n = 1
        id = 0
        prfs = {frozenset([0]): F}
        uci = 'test uci'
        m = 8
        a = [F(uci + str(h)) for h in range(m)]
        shares = thresha.pseudorandom_share(field, n, id, prfs, uci, m)
        b = thresha.recombine(field, [(1, shares)])
        self.assertEqual(a, b)

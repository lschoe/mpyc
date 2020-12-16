import unittest
from mpyc import gfpx
from mpyc import finfields
from mpyc import thresha


class Arithmetic(unittest.TestCase):

    def setUp(self):
        self.f2 = finfields.GF(2)
        self.f19 = finfields.GF(19)
        self.f27 = finfields.GF(gfpx.GFpX(3)(46))
        self.f256 = finfields.GF(gfpx.GFpX(2)(283))

    def test_secretsharing(self):
        for field in (self.f2, self.f256):
            t = 0
            m = 1
            a = [field(0), field(1)]
            shares = thresha.random_split(a, t, m)
            b = thresha.recombine(field, [(j + 1, shares[j]) for j in range(len(shares))])
            self.assertEqual(a, b)
            b = thresha.recombine(field, [(j + 1, shares[j]) for j in range(len(shares))], [0])[0]
            self.assertEqual(a, b)

        for field in (self.f19, self.f27, self.f256):
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
        key = int('0x00112233445566778899aabbccddeeff', 16).to_bytes(16, byteorder='little')
        bound = 1
        F = thresha.PRF(key, bound)
        y = F('test')
        self.assertEqual(y, 0)
        bound = 100
        F = thresha.PRF(key, bound)
        x = ''
        y = F(x.encode())
        self.assertTrue(0 <= y < bound)
        y2 = F(x.encode())
        self.assertEqual(y, y2)

    def test_prss(self):
        field = self.f256
        key = int('0x00112233445566778899aabbccddeeff', 16).to_bytes(16, byteorder='little')
        bound = 256  # field.modulus
        F = thresha.PRF(key, bound)
        m = 1
        pid = 0
        prfs = {(0,): F}
        uci = 'test uci'.encode()
        n = 8
        a = F(uci, n)
        shares = thresha.pseudorandom_share(field, m, pid, prfs, uci, n)
        b = thresha.recombine(field, [(1, shares)])
        self.assertEqual(a, [s.value for s in b])
        a = [0] * n
        shares = thresha.pseudorandom_share_zero(field, m, pid, prfs, uci, n)
        b = thresha.recombine(field, [(1, shares)])
        self.assertEqual(a, b)

        m = 3
        pid = 0
        prfs = {(0, 1): F, (0, 2): F}  # reuse dummy PRF
        shares0 = thresha.pseudorandom_share_zero(field, m, pid, prfs, uci, n)
        pid = 1
        prfs = {(0, 1): F, (1, 2): F}  # reuse dummy PRF
        shares1 = thresha.pseudorandom_share_zero(field, m, pid, prfs, uci, n)
        pid = 2
        prfs = {(0, 2): F, (1, 2): F}  # reuse dummy PRF
        shares2 = thresha.pseudorandom_share_zero(field, m, pid, prfs, uci, n)
        b = thresha.recombine(field, [(1, shares0), (2, shares1), (3, shares2)])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

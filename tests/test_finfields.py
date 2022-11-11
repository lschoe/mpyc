import operator
import unittest
from mpyc.numpy import np
from mpyc import gfpx
from mpyc import finfields


class Arithmetic(unittest.TestCase):

    def setUp(self):
        self.f2 = finfields.GF(gfpx.GFpX(2)(2))
        self.f256 = finfields.GF(gfpx.GFpX(2)(283))  # AES polynomial (283)_2 = X^8+X^4+X^3+X+1

        self.f2p = finfields.GF(2)
        self.f19 = finfields.GF(19)    # 19 % 4 = 3
        self.f101 = finfields.GF(101)  # 101 % 4 = 1
        self.f101.is_signed = False

        self.f27 = finfields.GF(gfpx.GFpX(3)(46))  # irreducible polynomial X^3 + 2X^2 + 1
        self.f81 = finfields.GF(gfpx.GFpX(3)(115))  # irreducible polynomial X^4 + X^3 + 2X + 1

    def test_field_caching(self):
        self.assertNotEqual(self.f2(1), self.f2p(1))
        f2_cached = finfields.GF(gfpx.GFpX(2)(2))
        self.assertEqual(self.f2(1), f2_cached(1))
        self.assertEqual(self.f2(1) * f2_cached(1), self.f2(1))
        f256_cached = finfields.GF(gfpx.GFpX(2)(283))
        self.assertEqual(self.f256(3), f256_cached(3))
        self.assertEqual(self.f256(3) * f256_cached(3), self.f256(5))
        self.assertEqual(self.f256(48) * f256_cached(16), self.f256(45))

        f2_cached = finfields.GF(2)
        self.assertEqual(self.f2p(1), f2_cached(1))
        self.assertEqual(self.f2p(1) * f2_cached(1), 1)
        f19_cached = finfields.GF(19)
        self.assertEqual(self.f19(3), f19_cached(3))
        self.assertEqual(self.f19(3) * f19_cached(3), 9)
        f101_cached = finfields.GF(101)
        self.assertEqual(self.f101(3), f101_cached(3))
        self.assertEqual(self.f101(3) * f101_cached(23), 69)

    def test_to_from_bytes(self):
        for F in [self.f2, self.f256, self.f2p, self.f19, self.f101]:
            self.assertEqual(F.from_bytes(F.to_bytes([])), [])
            self.assertEqual(F.from_bytes(F.to_bytes([0, 1])), [0, 1])
            self.assertEqual(F.from_bytes(F.to_bytes([F.order - 1])), [F.order - 1])

    def test_find_prime_root(self):
        f = finfields.find_prime_root
        pnw = f(2, False)
        self.assertEqual(pnw, (2, 1, 1))
        pnw = f(2)
        self.assertEqual(pnw, (3, 2, 3-1))
        pnw = f(5, n=1)
        self.assertEqual(pnw, (19, 1, 1))
        pnw = f(5, n=2)
        self.assertEqual(pnw, (19, 2, 19-1))
        p, n, w = f(5, n=3)
        self.assertEqual((w**3) % p, 1)
        p, n, w = f(10, n=4)
        self.assertEqual((w**n) % p, 1)

    def test_f2(self):
        f2 = self.f2
        self.assertFalse(f2(0))
        self.assertTrue(f2(1))
        self.assertEqual(f2(1) + f2(0), f2(0) + f2(1))
        self.assertEqual(1 + f2(0), 0 + f2(1))
        self.assertEqual(1 + f2(1), 0)
        self.assertEqual(1 - f2(1), 0)
        self.assertEqual(f2(1) / f2(1), f2(1))
        self.assertEqual(bool(f2(0)), False)
        self.assertEqual(bool(f2(1)), True)

        a = f2(1)
        b = f2(1)
        a += b
        self.assertEqual(a, f2(0))
        a -= b
        self.assertEqual(a, f2(1))
        a *= b
        self.assertEqual(a, f2(1))
        a /= b
        self.assertEqual(a, f2(1))

    def test_f256(self):
        f256 = self.f256
        self.assertFalse(f256(0))
        self.assertTrue(f256(1))
        self.assertEqual(f256(1) + 0, f256(0) + f256(1))
        self.assertEqual(f256(1) + 1, f256(0))
        self.assertEqual(f256(3) * 0, f256(0))
        self.assertEqual(f256(3) * 1, f256(3))
        self.assertEqual(f256(16) * f256(16), f256(27))
        self.assertEqual(f256(32) * f256(16), f256(54))
        self.assertEqual(f256(57) * f256(67), f256(137))
        self.assertEqual(f256(67) * f256(57), f256(137))
        self.assertEqual(f256(137) / f256(57), f256(67))
        self.assertEqual(f256(137) / f256(67), f256(57))

        a = f256(0)
        b = f256(1)
        a += b
        self.assertEqual(a, f256(1))
        a += 1
        self.assertEqual(a, f256(0))
        a -= b
        self.assertEqual(a, f256(1))
        a *= b
        self.assertEqual(a, f256(1))
        a *= 1
        self.assertEqual(a, f256(1))
        a /= 1
        self.assertEqual(a, f256(1))
        a <<= 0
        a = a >> 0
        self.assertEqual(a, f256(1))
        a <<= 2
        self.assertEqual(a, f256(4))
        a >>= 2
        self.assertEqual(a, f256(1))

        a = f256(3)  # generator X + 1
        s = [int((a**i).value) for i in range(255)]
        self.assertListEqual(sorted(s), list(range(1, 256)))
        s = [int((a**i).value) for i in range(-255, 0)]
        self.assertListEqual(sorted(s), list(range(1, 256)))

        f256 = finfields.GF(gfpx.GFpX(2)(391))  # primitive polynomial X^8 + X^7 + X^2 + X + 1
        a = f256(2)  # generator X
        s = [int((a**i).value) for i in range(255)]
        self.assertListEqual(sorted(s), list(range(1, 256)))

        a = f256(0)
        self.assertEqual(a.sqrt(), 0)
        with self.assertRaises(ZeroDivisionError):
            a.sqrt(INV=True)
        a = f256(177)
        self.assertTrue(a.is_sqr())
        self.assertEqual(a.sqrt()**2, a)
        self.assertEqual(a.sqrt(INV=True)**2, 1/a)

        self.assertEqual(len({f256(i) for i in range(-120, 259)}), 256)

    def test_f2p(self):
        f2 = self.f2p
        self.assertEqual(f2.nth, 1)
        self.assertEqual(f2.root, 1)
        self.assertEqual(f2.root ** f2.nth, 1)
        self.assertFalse(f2(0))
        self.assertTrue(f2(1))
        self.assertEqual(f2(1) + f2(0), f2(0) + f2(1))
        self.assertEqual(1 + f2(0), 0 + f2(1))
        self.assertEqual(1 + f2(1), 0)
        self.assertEqual(1 - f2(1), 0)
        self.assertEqual(f2(1) / f2(1), 1)
        self.assertEqual(f2(1).sqrt(), 1)
        self.assertEqual(bool(f2(0)), False)
        self.assertEqual(bool(f2(1)), True)
        self.assertTrue(f2(1).is_sqr())

        a = f2(1)
        b = f2(1)
        a += b
        self.assertEqual(a, 0)
        a -= b
        self.assertEqual(a, 1)
        a *= b
        self.assertEqual(a, 1)
        a /= b
        self.assertEqual(a, 1)

    def test_f19(self):
        f19 = self.f19
        self.assertEqual(f19.nth, 2)
        self.assertEqual(f19.root, 19 - 1)
        self.assertEqual(f19(f19.root) ** f19.nth, 1)
        self.assertEqual(bool(f19(0)), False)
        self.assertEqual(bool(f19(1)), True)
        self.assertEqual(bool(f19(-1)), True)
        self.assertEqual(int(f19(-1)), -1)
        self.assertEqual(abs(f19(-1)), 1)

        a = f19(12)
        b = f19(11)
        c = a + b
        self.assertEqual(c, (a.value + b.value) % 19)
        c = c - b
        self.assertEqual(c, a)
        c = c - a
        self.assertEqual(c, 0)
        self.assertEqual(a / a, 1)
        self.assertEqual(1 / a, 8)
        self.assertEqual(f19(0).sqrt(), 0)
        self.assertEqual((f19(1).sqrt())**2, 1)
        self.assertEqual(((a**2).sqrt())**2, a**2)
        self.assertNotEqual(((a**2).sqrt())**2, -a**2)
        self.assertEqual(a**f19.modulus, a)
        b = -a
        self.assertEqual(-b, a)
        a = f19(0)
        self.assertEqual(a.sqrt(), 0)
        with self.assertRaises(ZeroDivisionError):
            a.sqrt(INV=True)

        a = f19(12)
        b = f19(11)
        a += b
        self.assertEqual(a, 4)
        a -= b
        self.assertEqual(a, 12)
        a *= b
        self.assertEqual(a, 18)
        a <<= 2
        self.assertEqual(a, 15)
        a <<= 0
        self.assertEqual(a, 15)
        a >>= 2
        self.assertEqual(a, 18)
        a >>= 0
        self.assertEqual(a, 18)

        self.assertEqual(len({f19(i) for i in range(-20, 25)}), 19)

    def test_f101(self):
        f101 = self.f101
        self.assertEqual(f101.nth, 2)
        self.assertEqual(f101.root, 101 - 1)
        self.assertEqual(f101(f101.root) ** f101.nth, 1)

        a = f101(12)
        b = f101(11)
        c = a + b
        self.assertEqual(c, (a.value + b.value) % 101)
        c = c - b
        self.assertEqual(c, a)
        c = c - a
        self.assertEqual(c, 0)
        self.assertEqual(a / a, 1)
        self.assertEqual((f101(1).sqrt())**2, 1)
        self.assertEqual((f101(4).sqrt())**2, 4)
        self.assertEqual(((a**2).sqrt())**2, a**2)
        self.assertNotEqual(((a**2).sqrt())**2, -a**2)
        self.assertEqual(a**f101.modulus, a)
        b = -a
        self.assertEqual(-b, a)

        a = f101(120)
        b = f101(110)
        a += b
        self.assertEqual(a, 28)
        a -= b
        self.assertEqual(a, 19)
        a *= b
        self.assertEqual(a, 70)
        a /= b
        self.assertEqual(a, 19)

    def test_f27(self):
        f27 = self.f27  # 27 = 3 (mod 4)
        a = f27(0)
        self.assertEqual(a.sqrt(), 0)
        with self.assertRaises(ZeroDivisionError):
            a.sqrt(INV=True)
        a = f27(10)
        self.assertTrue((a**2).is_sqr())
        self.assertFalse((-a**2).is_sqr())
        b = (a**2).sqrt()
        self.assertEqual(b**2, a**2)
        b = (a**2).sqrt(INV=True)
        self.assertEqual((a * b)**2, 1)

    def test_f81(self):
        f81 = self.f81  # 81 = 1 (mod 4)
        a = f81(0)
        self.assertEqual(a.sqrt(), 0)
        with self.assertRaises(ZeroDivisionError):
            a.sqrt(INV=True)
        a = f81(21)
        self.assertTrue((a**2).is_sqr())
        self.assertTrue((-a**2).is_sqr())
        b = (a**2).sqrt()
        self.assertEqual(b**2, a**2)
        b = (a**2).sqrt(INV=True)
        self.assertEqual((a * b)**2, 1)

        self.assertEqual(len({f81(i) for i in range(-20, 125)}), 81)

    def test_errors(self):
        self.assertRaises(ValueError, finfields.GF, 4)
        self.assertRaises(ValueError, finfields.GF, gfpx.GFpX(2)(4))
        f2 = self.f2
        f2p = self.f2p
        f256 = self.f256
        f19 = self.f19
        self.assertRaises(TypeError, f19, 3.14)
        self.assertRaises(TypeError, operator.add, f2(1), f2p(2))
        self.assertRaises(TypeError, operator.iadd, f2(1), f2p(2))
        self.assertRaises(TypeError, operator.sub, f2(1), f256(2))
        self.assertRaises(TypeError, operator.isub, f2(1), f256(2))
        self.assertRaises(TypeError, operator.mul, f2(1), f19(2))
        self.assertRaises(TypeError, operator.imul, f2(1), f19(2))
        self.assertRaises(TypeError, operator.truediv, f256(1), f19(2))
        self.assertRaises(TypeError, operator.itruediv, f256(1), f19(2))
        self.assertRaises(TypeError, operator.truediv, 3.14, f19(2))
        self.assertRaises(TypeError, operator.lshift, f2(1), f2(1))
        self.assertRaises(TypeError, operator.ilshift, f2(1), f2(1))
        self.assertRaises(TypeError, operator.lshift, 1, f2(1))
        self.assertRaises(TypeError, operator.rshift, f19(1), f19(1))
        self.assertRaises(TypeError, operator.rshift, 1, f19(1))
        self.assertRaises(TypeError, operator.irshift, f19(1), f19(1))
        self.assertRaises(TypeError, operator.irshift, f256(1), f256(1))
        self.assertRaises(TypeError, operator.pow, f2(1), f19(2))
        self.assertRaises(TypeError, operator.pow, f19(1), 3.14)

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_array(self):
        np.assertEqual = np.testing.assert_array_equal

        F = self.f101
        self.assertEqual(np.add(100, F(8)), 7)
        self.assertEqual(np.subtract(1, F(8)), 94)
        self.assertEqual(np.multiply(100, F(-8)), 8)
        self.assertEqual(np.divide(1, F(2)), 51)
        self.assertEqual(np.sqrt(F(9))**2, 9)

        a = np.array([[-1, 1], [3, -3]])
        F_a = F.array(a)

        self.assertEqual(F.from_bytes(F.to_bytes(F_a.value.flat)), list(F_a.value.flat))
        self.assertTrue(isinstance(F_a[0, 0], F))
        self.assertTrue(isinstance(F_a[:, 0], F.array))
        F_a[0, 0] = 2
        F_a[0, :] = [109]
        self.assertEqual(int(F_a[0, 1]), 8)
        self.assertRaises(ValueError, operator.setitem, F_a, (0, 0), [2, 2])
        self.assertRaises(ValueError, operator.setitem, F_a, (0, ...), [2, 2, 2])
        self.assertRaises(TypeError, operator.pow, 2, F_a)
        self.assertRaises(TypeError, int, F_a)

        self.assertEqual(np.ndim(F_a), np.ndim(a))
        self.assertEqual(np.shape(F_a), np.shape(a))
        self.assertEqual(np.size(F_a), np.size(a))

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_array_ufunc(self):
        np.assertEqual = np.testing.assert_array_equal

        F = self.f101
        a = F.array([[8, 8], [3, -3]])

        a = np.add(a, a) + np.negative(a)
        np.negative.at(a, (1, 1))  # NB: in-place
        np.negative.at(a, (1, 1))  # NB: in-place
        np.assertEqual(np.add.reduce(a, 1), [16, 0])
        np.assertEqual(np.add.reduce(np.add.reduce(a, 1)), 16)
        a != a
        a += 2
        a -= 2
        a *= 3
        a >>= 2
        a <<= 1
        a = np.right_shift(np.left_shift(a, 2), 1)
        np.add(np.array([1], dtype=np.int32), a)
        np.add(a, np.array([1], dtype=np.int64))
        self.assertRaises(TypeError, np.add, np.array([1], dtype=np.float64), a)
        self.assertRaises(TypeError, np.add, a, np.array([1], dtype=np.float32))
#        a = np.power(a, 2)
#        a = divmod(a, 2)
        a **= 2
        a = a @ a

        F = finfields.GF(2**127 - 1)
        a, b = np.array([[-1, -1], [1, 1]]), np.array([[1, -5], [-1, -1]])
        F_a, F_b = F.array(a), F.array(b)
        a = a @ b
        F_a = F_a @ b
        b = a @ b
        F_b = a @ F_b
        np.assertEqual(np.multiply(F_a, F_b), np.multiply(a, b))
        np.assertEqual(F_a @ F_b, np.matmul(a, b))
        np.assertEqual(a @ F_b, np.matmul(a, b))
        np.assertEqual(np.matmul(F_a, F_b), np.matmul(a, b))
        np.assertEqual(np.add(F_a, F_b), np.add(a, b))

        np.assertEqual(np.reciprocal(F_b) * F_b, np.ones(b.shape, dtype='O'))
        np.assertEqual(np.sqrt(F_a**2)**2, F_a**2)

        F27_b = self.f27.array(b)
        F27_b = 1 / (1 / F27_b)
        np.assertEqual(np.sqrt(F27_b**2)**2, F27_b**2)

        F81_b2 = self.f81.array(b)**2
        self.assertTrue((F81_b2).is_sqr().all())
        np.assertEqual(F81_b2.sqrt(INV=True)**2, 1/F81_b2)

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_ndarray(self):
        np.assertEqual = np.testing.assert_array_equal

        F = self.f256
        a = np.array([[1, 2, 3, 4], [8, 7, 6, 5]])
        F_a = F.array(a)
        F_a_v = np.vectorize(lambda a: a.value, otypes='O')(F_a)

        self.assertEqual(F_a.ndim, a.ndim)
        self.assertEqual(F_a.shape, a.shape)
        self.assertEqual(F_a.size, a.size)
        np.assertEqual(F_a.T, a.T)
        np.assertEqual(F_a.transpose(), a.transpose())
        np.assertEqual(F_a.swapaxes(0, 1), a.swapaxes(0, 1))
        self.assertEqual(F_a.tolist(), a.tolist())
        np.assertEqual(F_a.ravel(), a.ravel())
        np.assertEqual(F_a.compress([0, 1]), a.compress([0, 1]))
        self.assertEqual(F_a.sum(), F_a_v.sum())
        np.assertEqual(F_a.prod(axis=1), F_a_v.prod(axis=1))
        np.assertEqual(F_a.repeat(4, axis=1), a.repeat(4, axis=1))
        np.assertEqual(F_a.diagonal(), a.diagonal())
        self.assertEqual(F_a.trace(), F_a_v.trace())

        self.assertTrue(F_a.is_sqr().all())
        np.assertEqual((F_a**2).sqrt(), a)

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_array_function(self):
        np.assertEqual = np.testing.assert_array_equal

        F = finfields.GF(2**127 - 1)
        a = np.array([[-1, -2, -3, -4], [0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 3, 4]])
        b = np.array([[10, 31, 1, -5], [76, 111, 11, 89], [67, 111, 1, -89], [-1, 10, 10, -1]])
        F_a, F_b = F.array(a), F.array(b)

        np.assertEqual(np.dot(F_a, F_b), np.dot(a, b))
        np.assertEqual(np.dot(F_a, 3), np.dot(a, 3))
        np.assertEqual(np.dot(F(3), F_a), np.dot(3, a))
        np.assertEqual(np.dot(F_a, F(3)), np.dot(a, 3))
        np.assertEqual(np.vdot(F_a, F_b), np.vdot(a, b))
        np.assertEqual(np.tensordot(F_a, F_b), np.tensordot(a, b))
        np.assertEqual(np.inner(F_a, F_b), np.inner(a, b))
        np.assertEqual(np.outer(F_a, F_b), np.outer(a, b))
        np.assertEqual(np.concatenate((F_a, F_b, F_a.T)), np.concatenate((a, b, a.T)))
        np.assertEqual(np.stack([F_a, F_b]), np.stack([a, b]))
        np.assertEqual(np.stack([a, F_b], axis=1), np.stack([a, b], axis=1))
        np.assertEqual(np.block([[F(0)]]), np.block([[0]]))
        np.assertEqual(np.block([0, F(1)]), np.block([0, 1]))
        np.assertEqual(np.block([[F_a, F_b], [F_a.T, F_b.T]]), np.block([[a, b], [a.T, b.T]]))
        np.assertEqual(np.vstack([F_a, F_b]), np.vstack([a, b]))
        np.assertEqual(np.hstack([F_a, F_b]), np.hstack([a, b]))
        np.assertEqual(np.dstack([F_a, F_b]), np.dstack([a, b]))
        np.assertEqual(np.column_stack([F_a, F_b]), np.column_stack([a, b]))
        np.assertEqual(np.row_stack([F_a, F_b]), np.row_stack([a, b]))
        np.assertEqual(np.split(F_a, 2)[0], np.split(a, 2)[0])
        np.assertEqual(np.array_split(F_a, 2)[1], np.array_split(a, 2)[1])
        np.assertEqual(np.dsplit(F_a.reshape(1, 4, 4), 2)[1], np.dsplit(a.reshape(1, 4, 4), 2)[1])
        np.assertEqual(np.hsplit(F_a, 2)[1], np.hsplit(a, 2)[1])
        np.assertEqual(np.vsplit(F_a, 2)[0], np.vsplit(a, 2)[0])
        np.assertEqual(np.tile(F_a, 3), np.tile(a, 3))
        np.assertEqual(np.tile(F_a, (2, 2)), np.tile(a, (2, 2)))
        np.assertEqual(np.repeat(F(3), 4), np.repeat(3, 4))
        np.assertEqual(np.repeat(F_a, [1, 2, 2, 1], axis=0), np.repeat(a, [1, 2, 2, 1], axis=0))
        np.assertEqual(np.delete(F_a, 1, 0), np.delete(a, 1, 0))
        np.assertEqual(np.delete(F_a, [1, 3]), np.delete(a, [1, 3]))
        np.assertEqual(np.insert(F_a, 1, 5), np.insert(a, 1, 5))
        np.assertEqual(np.insert(F_a, 1, F(5), axis=1), np.insert(a, 1, 5, axis=1))
        np.assertEqual(np.append(F_a, F_b), np.append(a, b))
        np.assertEqual(np.append(F_a, [[0, 0, 0, 0]], axis=0), np.append(a, [[0, 0, 0, 0]], axis=0))
        np.assertEqual(np.append(F_a, F_b, axis=1), np.append(a, b, axis=1))
        np.assertEqual(np.resize(F_a, (3, 7)), np.resize(a, (3, 7)))
        np.assertEqual(np.trim_zeros(F_a[0], trim='b'), np.trim_zeros(a[0], trim='b'))
        np.assertEqual(np.flip(F_a), np.flip(a))
        np.assertEqual(np.fliplr(F_a), np.fliplr(a))
        np.assertEqual(np.flipud(F_a), np.flipud(a))
        np.assertEqual(np.reshape(F_a, (F_a.size,), order='F'), np.reshape(a, (a.size,), order='F'))
        np.assertEqual(np.roll(F_a, -1), np.roll(a, -1))
        np.assertEqual(np.rot90(F_a), np.rot90(a))
        np.assertEqual(np.sum(F_a), np.sum(a))
        np.assertEqual(np.sum(F_a, axis=0), np.sum(a, axis=0))
        np.assertEqual(np.prod(F_a), np.prod(a))
        np.assertEqual(np.prod(F_a, axis=1), np.prod(a, axis=1))
        np.assertEqual(np.kron(F_a, F_b), np.kron(a, b))

        np.assertEqual(np.linalg.matrix_power(F_a, 3), np.linalg.matrix_power(a, 3))
        np.assertEqual(F_b @ np.linalg.solve(F_b, F_a), F_a)
        np.assertEqual(np.linalg.inv(F_b) @ F_b, np.eye(len(F_b), dtype='O'))
        self.assertEqual(np.linalg.det(F_a), round(np.linalg.det(a)))
        self.assertEqual(np.linalg.det(F_b), round(np.linalg.det(b)))
        np.assertEqual(np.linalg.det(np.stack((F_a, F_b))),
                       np.vectorize(round)(np.linalg.det(np.stack((a, b)))))
        np.assertEqual(np.linalg.matrix_power(F_b, -5) @ np.linalg.matrix_power(F_b, 5),
                       np.eye(len(F_b), dtype='O'))

        F81_b = self.f81.array(b)
        np.assertEqual(np.linalg.inv(F81_b) @ F81_b, np.eye(len(F81_b), dtype='O'))

        np.assertEqual(np.tril(F_a), np.tril(a))
        np.assertEqual(np.triu(F_a), np.triu(a))
        np.assertEqual(np.diag(F_a), np.diag(a))
        np.assertEqual(np.diag(F_a[0]), np.diag(a[0]))
        np.assertEqual(np.diagflat(F_a), np.diagflat(a))
        np.assertEqual(np.diagonal(F_a), np.diagonal(a))
        np.assertEqual(np.vander(F_b[0]), np.vander(b[0]))

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_array_getsetitem(self):
        np.assertEqual = np.testing.assert_array_equal

        F = finfields.GF(257)
        a = np.array([[-1, -2, -3, -4], [0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 3, 4]])
        b = np.array([[0, 31, 1, -5], [76, 111, 11, 89], [67, 111, 1, -89], [-1, 0, 0, -1]])
        F_a, F_b = F.array(a), F.array(b)

        F_a[1] = F_b[1]
        F_a[2] = b[2]
        F_a[[0, 3]] = F_b[[0, 3]]
        np.assertEqual(F_a, F_b)
        F_a = F.array(a)
        F_a[:] = F_b[:]
        np.assertEqual(F_a, F_b)
        F_a = F.array(a)
        F_a[:, 1] = F_b[:, 1]
        F_a[:, 2] = b[:, 2]
        F_a[:, [0, 3]] = F_b[:, [0, 3]]
        np.assertEqual(F_a, F_b)
        F_a = F.array(a)

        self.assertRaises(ValueError, operator.setitem, F_a, 0, F_b)
        self.assertRaises(ValueError, operator.setitem, F_a, (slice(None), 1), F_b)

        self.assertTrue(F_a[0] in F_a)
        self.assertTrue(-F_a[0] in F_a)
        self.assertTrue(-F_a[2] in F_a)  # NB: tricky semantics for __contains__()
        self.assertFalse(100 in F_a)
        self.assertTrue(F_a[0, 0] in F_a)


if __name__ == "__main__":
    unittest.main()

import operator
import math
import unittest
from mpyc.numpy import np
from mpyc.runtime import mpc


class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mpc.logging(False)
        mpc.run(mpc.start())

    @classmethod
    def tearDownClass(cls):
        mpc.run(mpc.shutdown())

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_secint_array(self):
        np.assertEqual = np.testing.assert_array_equal

        secint = mpc.SecInt()
        FF = secint.field
        a = FF.array([[[-1, 1], [1, -1]]])  # 3D array
        np.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(a, senders=0)))), a)
        c = secint.array(a)
        mpc.peek(c, label='secint.array')
        a = a.copy()  # NB: needed to pass tests with inplace operations
        self.assertTrue((a == mpc.run(mpc.output(mpc.np_sgn(c)))).all())  # via FF array __eq__
        self.assertTrue((mpc.run(mpc.output(mpc.np_sgn(c))) == a).all())  # via FF array ufunc equal
        np.assertEqual(mpc.run(mpc.output(mpc.np_sgn(c))), a)

        np.assertEqual(mpc.run(mpc.output(c + np.array([1, 2]))), a + np.array([1, 2]))
        np.assertEqual(mpc.run(mpc.output(c * np.array([1, 2]))), a * np.array([1, 2]))
        np.assertEqual(mpc.run(mpc.output(c * secint(2))), a * 2)
        np.assertEqual(mpc.run(mpc.output(c @ c)), a @ a)
        self.assertEqual(mpc.run(mpc.output(c[0][0] @ c[0][1])), a[0][0] @ a[0][1])

        np.assertEqual(mpc.run(mpc.output(c)), a)
        np.assertEqual(mpc.run(mpc.output(mpc._reshare(c) @ c)), a @ a)
        np.assertEqual(mpc.run(mpc.output(np.append(c, c, axis=0))), np.append(a, a, axis=0))
        np.assertEqual(mpc.run(mpc.output(np.append(c, c))), np.append(a, a))
        np.assertEqual(mpc.run(mpc.output(np.concatenate((c, c, c)))), np.concatenate((a, a, a)))
        np.assertEqual(mpc.run(mpc.output(np.concatenate((c, a, c), axis=None))),
                       np.concatenate((a, a, a), axis=None))
        np.assertEqual(mpc.run(mpc.output(np.concatenate((a, c)))), np.concatenate((a, a)))
        b = np.stack((a, a, a))
        d = np.stack((c, c, c))
        np.assertEqual(mpc.run(mpc.output(c @ d)), a @ b)
        np.assertEqual(mpc.run(mpc.output(d @ d)), b @ b)
        np.assertEqual(mpc.run(mpc.output(np.outer(c, a))), np.outer(a, a))
        np.assertEqual(mpc.run(mpc.output(np.stack((c, c), axis=1))), np.stack((a, a), axis=1))
        np.assertEqual(mpc.run(mpc.output(np.block([[c, c], [c, c]]))), np.block([[a, a], [a, a]]))
        np.assertEqual(mpc.run(mpc.output(np.block([[secint(9), -1]]))), np.block([[9, -1]]))
        np.assertEqual(mpc.run(mpc.output(np.vstack((c, c, c)))), np.vstack((a, a, a)))
        np.assertEqual(mpc.run(mpc.output(np.hstack((c, c, c)))), np.hstack((a, a, a)))
        np.assertEqual(mpc.run(mpc.output(np.hstack((c[0, 0],)))), np.hstack((a[0, 0],)))
        np.assertEqual(mpc.run(mpc.output(np.dstack((c, c, c)))), np.dstack((a, a, a)))
        np.assertEqual(mpc.run(mpc.output(np.dstack((c[0], c[0])))), np.dstack((a[0], a[0])))
        np.assertEqual(mpc.run(mpc.output(np.dstack((c[0, 0],)))), np.dstack((a[0, 0],)))
        np.assertEqual(mpc.run(mpc.output(np.column_stack((c, c, c)))), np.column_stack((a, a, a)))
        if np.lib.NumpyVersion(np.__version__) < '2.0.0b1':
            np.assertEqual(mpc.run(mpc.output(np.row_stack((c, c, c)))), np.row_stack((a, a, a)))
        np.assertEqual(mpc.run(mpc.output(np.split(c, 2, 1)[0])), np.split(a, 2, 1)[0])
        np.assertEqual(mpc.run(mpc.output(np.dsplit(d, 1)[0])), np.dsplit(b, 1)[0])
        np.assertEqual(mpc.run(mpc.output(np.hsplit(c, 2)[0])), np.hsplit(a, 2)[0])
        np.assertEqual(mpc.run(mpc.output(np.vsplit(c, np.array([1]))[0])), np.vsplit(a, [1])[0])
        np.assertEqual(mpc.run(mpc.output(np.reshape(c, (-1,)))), np.reshape(a, (-1,)))
        np.assertEqual(mpc.run(mpc.output(np.reshape(c, -1))), np.reshape(a, -1))
        np.assertEqual(mpc.run(mpc.output(np.flip(c))), np.flip(a))
        np.assertEqual(mpc.run(mpc.output(np.fliplr(c))), np.fliplr(a))
        np.assertEqual(mpc.run(mpc.output(np.flipud(c))), np.flipud(a))
        np.assertEqual(mpc.run(mpc.output(np.rot90(c))), np.rot90(a))
        np.assertEqual(mpc.run(mpc.output(np.rot90(d))), np.rot90(b))
        self.assertEqual(np.rot90(d).shape, np.rot90(b).shape)
        a1, a2 = a[:, :, 1], a[:, 0, :].reshape(2, 1)
        np.assertEqual(mpc.run(mpc.output(np.add(secint.array(a1), secint.array(a2)))), a1 + a2)
        np.assertEqual(mpc.run(mpc.output(a1 + secint.array(a[:, 0, :]).reshape(2, 1))), a1 + a2)
        np.assertEqual(mpc.run(mpc.output(c)), a)
        np.assertEqual(mpc.run(mpc.output(c, raw=True)), a)
        np.assertEqual(mpc.run(mpc.output(mpc.input(c, senders=0))), a)
        np.assertEqual(mpc.run(mpc.output(c + c)), a + a)
        np.assertEqual(mpc.run(mpc.output(c + a.value)), a + a)
        np.assertEqual(mpc.run(mpc.output(np.add(c, a))), np.add(a, a))
        np.assertEqual(mpc.run(mpc.output(np.add(2, c))), 2 + a)
        np.assertEqual(mpc.run(mpc.output(np.add(secint(2), c))), 2 + a)
        np.assertEqual(mpc.run(mpc.output(np.add(c, FF(2)))), 2 + a)
        np.assertEqual(mpc.run(mpc.output(np.subtract(c, FF(2)))), a - 2)
        np.assertEqual(mpc.run(mpc.output(np.subtract(c, secint(2)))), a - 2)
        np.assertEqual(mpc.run(mpc.output(np.subtract(2, c))), 2 - a)
        np.assertEqual(mpc.run(mpc.output(np.subtract(FF(2), c))), 2 - a)
        np.assertEqual(mpc.run(mpc.output(np.subtract(secint(2), c))), 2 - a)
        np.assertEqual(mpc.run(mpc.output(-c)), -a)
        a += 2
        c += 2
        np.assertEqual(mpc.run(mpc.output(c)), a)
        np.assertEqual(mpc.run(mpc.output(c * c)), a * a)
        a *= 2
        c *= 2
        np.assertEqual(mpc.run(mpc.output(c)), a)
        a /= 2
        c /= 2
        np.assertEqual(mpc.run(mpc.output(c)), a)
        np.assertEqual(mpc.run(mpc.output(c @ c)), a @ a)
        np.assertEqual(mpc.run(mpc.output(c @ a)), a @ a)
        np.assertEqual(mpc.run(mpc.output(a @ c)), a @ a)
        self.assertAlmostEqual(mpc.run(mpc.output(mpc.trunc(np.sum(np.abs(d)), 3))), 1, delta=1)
        self.assertAlmostEqual(mpc.run(mpc.output(mpc.trunc(-np.sum(np.abs(d)), 3))), -2, delta=1)

        self.assertEqual(mpc.run(mpc.output(c)).dtype, object)

        np.assertEqual(mpc.run(mpc.output(c == c)), True)
        np.assertEqual(mpc.run(mpc.output(c != c)), False)
        np.assertEqual(mpc.run(mpc.output(c < c)), False)
        np.assertEqual(mpc.run(mpc.output(c < c+1)), True)
        np.assertEqual(mpc.run(mpc.output(c <= c)), True)
        np.assertEqual(mpc.run(mpc.output(c > c)), False)
        np.assertEqual(mpc.run(mpc.output(c >= c)), True)
        np.assertEqual(mpc.run(mpc.output(c < -c)), a.signed_() < (-a).signed_())
        np.assertEqual(mpc.run(mpc.output(np.negative(c))), np.negative(a))
        np.assertEqual(mpc.run(mpc.output(np.absolute(-c))), a)
        np.assertEqual(mpc.run(mpc.output(np.minimum(c, -c))), -a)
        np.assertEqual(mpc.run(mpc.output(np.minimum(c, 10))), a)
        np.assertEqual(mpc.run(mpc.output(np.minimum(c, secint(10)))), a)
        np.assertEqual(mpc.run(mpc.output(np.maximum(c, -c))), a)
        np.assertEqual(mpc.run(mpc.output(np.maximum(c, -10))), a)
        np.assertEqual(mpc.run(mpc.output(np.maximum(c, secint(-10)))), a)
        np.assertEqual(mpc.run(mpc.output(np.where(c < 0, -c, c))), a)
        np.assertEqual(mpc.run(mpc.output(np.where(c[0] < 0, -c, c))), a)
        np.assertEqual(mpc.run(mpc.output(np.where(True, c, 10))), a)
        np.assertEqual(mpc.run(mpc.output(np.where(False, 10, c))), a)
        a_ = a.signed_()
        np.assertEqual(mpc.run(mpc.output(np.amin(c))), np.amin(a_))
        np.assertEqual(mpc.run(mpc.output(np.amin(c, keepdims=True))), np.amin(a_, keepdims=True))
        np.assertEqual(mpc.run(mpc.output(np.amin(c[0], axis=0))), np.amin(a_[0], axis=0))
        np.assertEqual(mpc.run(mpc.output(np.amin(c, axis=2))), np.amin(a_, axis=2))
        np.assertEqual(mpc.run(mpc.output(np.amin(c, axis=2, keepdims=True))),
                       np.amin(a_, axis=2, keepdims=True))
        np.assertEqual(mpc.run(mpc.output(np.amin(c, axis=(1, 2)))), np.amin(a_, axis=(1, 2)))
        np.assertEqual(mpc.run(mpc.output(np.amin(c, axis=(0, 1), keepdims=True))),
                       np.amin(a_, axis=(0, 1), keepdims=True))
        np.assertEqual(mpc.run(mpc.output(np.amax(c))), np.amax(a_))
        np.assertEqual(mpc.run(mpc.output(np.amax(c, keepdims=True))), np.amax(a_, keepdims=True))
        np.assertEqual(mpc.run(mpc.output(np.amax(c[0], axis=1))), np.amax(a_[0], axis=1))
        np.assertEqual(mpc.run(mpc.output(np.amax(c, axis=-2))), np.amax(a_, axis=-2))
        np.assertEqual(mpc.run(mpc.output(np.amax(c, axis=-2, keepdims=True))),
                       np.amax(a_, axis=-2, keepdims=True))
        np.assertEqual(mpc.run(mpc.output(np.amax(c, axis=(2, -3)))), np.amax(a_, axis=(2, -3)))
        np.assertEqual(mpc.run(mpc.output(np.amax(c, axis=(1, 2), keepdims=True))),
                       np.amax(a_, axis=(1, 2), keepdims=True))
        np.assertEqual(mpc.run(mpc.output(np.sort(c[..., :1]))), np.sort(a_[..., :1]))

        np.assertEqual(mpc.run(mpc.output(np.all(c == 0))), False)
        np.assertEqual(mpc.run(mpc.output(np.all(c != 0))), True)
        np.assertEqual(mpc.run(mpc.output(np.any(c == 0, axis=1))), False)
        np.assertEqual(mpc.run(mpc.output(np.any(c - c[0, 1, 1] == 0))), True)
        np.assertEqual(mpc.run(mpc.output(np.any(c - c[0, 1, 1] == 0, axis=(1, 2)))), True)
        np.assertEqual(mpc.run(mpc.output(np.any(c - c[0, 1] == 0, axis=(0, 2)))), [False, True])

        np.assertEqual(mpc.run(mpc.output(np.argmax(c[0], axis=0))), np.argmax(a[0], axis=0))
        b = np.concatenate((a, 2+a, 3+a, a))
        d = np.concatenate((c, 2+c, 3+c, c))
        np.assertEqual(mpc.run(mpc.output(np.sort(d, axis=None))), np.sort(b.signed_(), axis=None))
        np.assertEqual(mpc.run(mpc.output(d.sort(key=lambda a: a+1))), np.sort(b.signed_()))
        np.assertEqual(mpc.run(mpc.output(np.argmin(d, axis=0))), np.argmin(b, axis=0))
        np.assertEqual(mpc.run(mpc.output(np.argmin(d, axis=0, keepdims=True))),
                       np.argmin(b, axis=0, keepdims=True))
        np.assertEqual(mpc.run(mpc.output(np.argmax(d, axis=2))), np.argmax(b, axis=2))
        b3 = np.stack((b, b+1, b-1)).signed_()
        d3 = np.stack((d, d+1, d-1))
        np.assertEqual(mpc.run(mpc.output(np.argmin(d3, axis=2))), np.argmin(b3, axis=2))
        np.assertEqual(mpc.run(mpc.output(d3.argmin(axis=2, arg_unary=False, arg_only=True))),
                       np.argmin(b3, axis=2))
        u = np.array([0, 1])
        np.assertEqual(mpc.run(mpc.output(d3.argmin(axis=2, arg_unary=True, arg_only=True) @ u)),
                       np.argmin(b3, axis=2))
        np.assertEqual(mpc.run(mpc.output(np.argmax(d3, axis=1))), np.argmax(b3, axis=1))
        np.assertEqual(mpc.run(mpc.output(np.argmax(d3, keepdims=True))), b3.argmax(keepdims=True))
        np.assertEqual(mpc.run(mpc.output(np.argmin(d3, axis=0))), np.argmin(b3, axis=0))
        np.assertEqual(mpc.run(mpc.output(d3.argmin(keepdims=True, arg_unary=False)[0])),
                       b3.argmin(keepdims=True))
        np.assertEqual(mpc.run(mpc.output(d3.argmin(axis=0, keepdims=True, arg_unary=False)[0])),
                       b3.argmin(axis=0, keepdims=True))
        np.assertEqual(mpc.run(mpc.output(d3.argmin(axis=1, keepdims=True)[1])),
                       np.amin(b3, axis=1, keepdims=True))
        np.assertEqual(mpc.run(mpc.output(d3.argmax(axis=2, keepdims=True, arg_unary=False)[0])),
                       b3.argmax(axis=2, keepdims=True))
        np.assertEqual(mpc.run(mpc.output(d3.argmax(axis=-1, keepdims=True)[1])),
                       np.amax(b3, axis=-1, keepdims=True))
        np.assertEqual(mpc.run(mpc.output(d3.argmax(keepdims=True, arg_unary=False)[0])),
                       np.argmax(b3, keepdims=True))

        class key:
            size = 2  # __lt__() assumes last dimension of size 2

            def __init__(self, a):
                self.a = a

            def __lt__(self, other):
                return self.a[..., 0] < other.a[..., 0]

        np.assertEqual(mpc.run(mpc.output(d3.argmin(axis=2, key=key, arg_unary=False)[0])),
                       np.argmin((np.delete(b3, 1, 3).reshape(b3.shape[:-1])), axis=-1))
        np.assertEqual(mpc.run(mpc.output(d3.argmin(key=key, arg_unary=False, arg_only=True))),
                       np.argmin((np.delete(b3, 1, 3).reshape(b3.shape[:-1]))))
        np.assertEqual(mpc.run(mpc.output(d3.argmax(axis=2, key=key, arg_unary=False)[0])),
                       np.argmin((np.delete(b3, 0, 3).reshape(b3.shape[:-1])), axis=-1))
        np.assertEqual(mpc.run(mpc.output(d3.argmax(key=key, arg_unary=False, arg_only=True))),
                       np.argmax((np.delete(b3, 1, 3).reshape(b3.shape[:-1]))))

        c_, a_ = c.flatten()[:3], a.signed_().flatten()[:3]
        np.assertEqual(mpc.run(mpc.output(np.argmin(c_[:1]))), np.argmin(a_[:1]))
        np.assertEqual(mpc.run(mpc.output(c_.argmin(key=operator.neg, arg_unary=False)[0])),
                       a_.argmax())
        np.assertEqual(mpc.run(mpc.output(list(c_.argmin(arg_unary=False)))),
                       [a_.argmin(), np.amin(a_)])
        np.assertEqual(mpc.run(mpc.output(c_.argmin(arg_only=True))), [1, 0, 0])
        np.assertEqual(mpc.run(mpc.output(np.argmax(c_[:1]))), np.argmax(a_[:1]))
        np.assertEqual(mpc.run(mpc.output(c_.argmax(key=operator.neg, arg_unary=False)[0])),
                       a_.argmin())
        np.assertEqual(mpc.run(mpc.output(list(c_.argmax(arg_unary=False)))),
                       [a_.argmax(), np.amax(a_)])
        np.assertEqual(mpc.run(mpc.output(c_.argmax(arg_only=True))), [0, 1, 0])

        self.assertEqual(mpc.run(mpc.output(c_.tolist())), a_.tolist())
        np.assertEqual(mpc.run(mpc.output(mpc.np_fromlist(c_.tolist()))), a_)

        self.assertEqual(mpc.run(mpc.output(list(c.flat))), list(a.flat))
        self.assertEqual(len(c), len(a))
        self.assertRaises(TypeError, len, secint.array(np.array(42)))
        np.assertEqual(mpc.run(mpc.output(c.copy())), mpc.run(mpc.output(c)))
        np.assertEqual(mpc.run(mpc.output(c.transpose([1, 2, 0]))), a.transpose([1, 2, 0]))
        np.assertEqual(mpc.run(mpc.output(c[0, 0].transpose())), a[0, 0].transpose())
        np.assertEqual(mpc.run(mpc.output(c.swapaxes(0, 1))), a.swapaxes(0, 1))
        self.assertEqual(mpc.run(mpc.output(c.sum(initial=secint(-1)))), a.sum(initial=-1))
        np.assertEqual(mpc.run(mpc.output(c.sum(axis=0))), a.sum(axis=0))
        np.assertEqual(mpc.run(mpc.output(c.sum(keepdims=True))), a.sum(keepdims=True))
        np.assertEqual(mpc.run(mpc.output(c.sum(axis=(0, 2), keepdims=True))),
                       a.sum(axis=(0, 2), keepdims=True))
        np.assertEqual(mpc.run(mpc.output(c**254 * c**0 * c**-253)), a)

        # TODO: c //= 2 secure int __floordiv__() etc.

        np.assertEqual(mpc.run(mpc.output(np.linalg.det(c[0]))), np.linalg.det(a[0]))

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_secfxp_array(self):
        np.assertEqual = np.testing.assert_array_equal
        np.assertAlmostEqual = np.testing.assert_allclose

        secfxp = mpc.SecFxp(12)
        a = np.array([[-1.5, 2.5], [4.5, -8.5]])
        c = secfxp.array(a)
        mpc.peek(c)

        np.assertEqual(mpc.run(mpc.output(c + np.array([1, 2]))), a + np.array([1, 2]))
        np.assertEqual(mpc.run(mpc.output(c * np.array([1, 2]))), a * np.array([1, 2]))
        np.assertEqual(mpc.run(mpc.output(c * secfxp(2))), a * 2)
        np.assertEqual(mpc.run(mpc.output(c + 2.5)), a + 2.5)
        np.assertEqual(mpc.run(mpc.output(np.add(c, np.float32(2.5)))), a + 2.5)
        np.assertEqual(mpc.run(mpc.output(c + np.array([1.5, 2.5]))), a + np.array([1.5, 2.5]))
        np.assertEqual(mpc.run(mpc.output(c * np.array([1.5, 2.5]))), a * np.array([1.5, 2.5]))
        np.assertEqual(mpc.run(mpc.output(c * secfxp(2.5))), a * 2.5)
        np.assertEqual(mpc.run(mpc.output(c * 2.5)), a * 2.5)
        np.assertEqual(mpc.run(mpc.output(c / secfxp.field(2))), a / 2)
        np.assertEqual(mpc.run(mpc.output(c / secfxp.field.array([2]))), a / 2)

        # NB: NumPy dispatcher converts np.int8 to int
        np.assertEqual(mpc.run(mpc.output(c * np.int8(2))), a * 2)

        np.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(c, senders=0)))), a)
        np.assertEqual(mpc.run(mpc.output(mpc.input(c, senders=0))), a)
        np.assertEqual(mpc.run(mpc.output(mpc._reshare(c) @ c)), a @ a)
        np.assertEqual(mpc.run(mpc.output(np.outer(secfxp.array(np.array([2, 2])), c))),
                       np.outer([2, 2], a))
        b = mpc.run(mpc.output(c * c))
        np.assertEqual(b, a * a)
        self.assertTrue(np.issubdtype(b.dtype, np.floating))
        np.assertEqual(mpc.run(mpc.output(np.block([[secfxp(9.5)]]))), np.block([[9.5]]))

        np.assertEqual(mpc.run(mpc.output(np.equal(c, c))), True)
        np.assertEqual(mpc.run(mpc.output(np.equal(c, 0))), False)
        np.assertEqual(mpc.run(mpc.output(np.all(c == c))), True)
        self.assertEqual(np.any(c != c).integral, True)
        np.assertEqual(mpc.run(mpc.output(np.any(c != c))), False)
        np.assertEqual(mpc.run(mpc.output(np.sort(c))), np.sort(a))
        np.assertEqual(mpc.run(mpc.output(np.sort(c, axis=None))), np.sort(a, axis=None))
        np.assertEqual(mpc.run(mpc.output(np.sort(c, axis=0))), np.sort(a, axis=0))

        f = 32
        secfxp = mpc.SecFxp(2*f)
        c = secfxp.array(a)
        np.assertAlmostEqual(mpc.run(mpc.output(c / 0.5)), a / 0.5, rtol=0, atol=0)
        np.assertAlmostEqual(mpc.run(mpc.output(c / 2.45)), a / 2.45, rtol=0, atol=2**(1-f))
        np.assertAlmostEqual(mpc.run(mpc.output(c / 2.5)), a / 2.5, rtol=0, atol=2**(2-f))
        np.assertAlmostEqual(mpc.run(mpc.output(c / c[0, 1])), a / 2.5, rtol=0, atol=2**(3-f))
        np.assertAlmostEqual(mpc.run(mpc.output(1 / c)), 1 / a, rtol=0, atol=2**(1-f))
        np.assertAlmostEqual(mpc.run(mpc.output(secfxp(1.5) / c)), 1.5 / a, rtol=0, atol=2**(1-f))
        np.assertAlmostEqual(mpc.run(mpc.output(1.5 / c)), 1.5 / a, rtol=0, atol=2**(1-f))
        np.assertAlmostEqual(mpc.run(mpc.output(a / c)), 1, rtol=0, atol=2**(3-f))
        np.assertAlmostEqual(mpc.run(mpc.output((2*a).astype(int) / c)), 2, rtol=0, atol=2**(4-f))
        np.assertAlmostEqual(mpc.run(mpc.output(c / a)), 1, rtol=0, atol=2**(0-f))
        np.assertAlmostEqual(mpc.run(mpc.output(c / c)), 1, rtol=0, atol=2**(3-f))
        np.assertEqual(mpc.run(mpc.output(np.equal(c, c))), True)
        np.assertEqual(mpc.run(mpc.output(np.equal(c, 0))), False)
        np.assertEqual(mpc.run(mpc.output(np.sum(c, axis=(-2, 1)))), np.sum(a, axis=(-2, 1)))
        np.assertEqual(mpc.run(mpc.output(c.sum(axis=1, initial=1.5))), a.sum(axis=1, initial=1.5))
        self.assertEqual(np.prod(c, axis=(-2, 1)).integral, False)
        np.assertEqual(mpc.run(mpc.output(np.prod(c, axis=(-2, 1)))), np.prod(a, axis=(-2, 1)))
        a = a.flatten()[:3]
        c = c.flatten()[:3]
        np.assertEqual(mpc.run(mpc.output(np.amin(c))), np.amin(a))
        np.assertEqual(mpc.run(mpc.output(np.argmin(c))), np.argmin(a))
        np.assertEqual(mpc.run(mpc.output(np.amax(c))), np.amax(a))
        np.assertEqual(mpc.run(mpc.output(np.argmax(c))), np.argmax(a))
        np.assertEqual(mpc.run(mpc.output(np.minimum(c, 10))), a)
        np.assertEqual(mpc.run(mpc.output(np.minimum(c, secfxp(10)))), a)
        np.assertEqual(mpc.run(mpc.output(np.maximum(c, -10))), a)
        np.assertEqual(mpc.run(mpc.output(np.maximum(c, secfxp(-10)))), a)
        np.assertEqual(mpc.run(mpc.output(np.where(c < 0, -c, c))), abs(a))
        np.assertEqual(mpc.run(mpc.output(np.where(c[0] < 0, -c, c))), -a)
        np.assertEqual(mpc.run(mpc.output(np.where(True, c, 10))), a)
        np.assertEqual(mpc.run(mpc.output(np.where(False, 10, c))), a)

        u = mpc.np_unit_vector(secfxp(3), 7)
        self.assertEqual(mpc.run(mpc.output(np.argmax(u))), 3)

        # Test the integral property
        a1 = np.array([[-1.5, 2.5], [4.5, -8.5]])
        a2 = np.array([[-1, 2], [4, -8]])
        c1 = secfxp.array(a1)
        c2 = secfxp.array(a2)
        self.assertEqual(c1.integral, False)
        self.assertEqual(c2.integral, True)
        self.assertEqual(c1.copy().integral, False)
        self.assertEqual(c2.copy().integral, True)
        self.assertEqual(c1[0, 0].integral, False)
        self.assertEqual(c2[0, 0].integral, True)

        self.assertEqual(c1.flatten().tolist()[0].integral, False)
        self.assertEqual(c2.flatten().tolist()[0].integral, True)
        self.assertEqual(mpc.np_fromlist([secfxp(-1.5), secfxp(2.5)]).integral, False)
        self.assertEqual(mpc.np_fromlist([secfxp(-1), secfxp(2)]).integral, True)

        self.assertEqual((c1 + c2).integral, False)
        self.assertEqual(np.add(c2, c2).integral, True)
        self.assertEqual((c1 * c2).integral, False)
        self.assertEqual(np.multiply(c2, c2).integral, True)
        self.assertEqual(np.matmul(c1, c1).integral, False)
        self.assertEqual(np.matmul(c1, c2).integral, False)
        self.assertEqual((c2 @ c2).integral, True)
        self.assertEqual(np.outer(c1, c1).integral, False)
        self.assertEqual(np.outer(c2, c2).integral, True)
        self.assertEqual(np.outer(c1, c2).integral, False)
        self.assertEqual((c2[0, :] @ c2[:, 0]).integral, True)  # Cover lines related scalar outputs
        self.assertEqual((c1[0, :] @ c1[:, 0]).integral, False)
        self.assertEqual((c1[0, :] @ c2[:, 0]).integral, False)

        self.assertEqual(np.hstack((c1, c2)).integral, False)
        self.assertEqual(np.hstack((c1, c1)).integral, False)
        self.assertEqual(np.hstack((c2, c2)).integral, True)
        self.assertEqual(np.column_stack((c1, c2)).integral, False)
        self.assertEqual(np.column_stack((c1, a1)).integral, False)
        self.assertEqual(np.column_stack((c2, c2)).integral, True)
        self.assertEqual(np.stack((c1, c2)).integral, False)
        self.assertEqual(np.stack((c2, a1)).integral, False)
        self.assertEqual(np.stack((c2, c2, a2)).integral, True)
        self.assertEqual(np.vstack((c1, a2)).integral, False)
        self.assertEqual(np.vstack((c1, c1)).integral, False)
        self.assertEqual(np.vstack((c2, c2)).integral, True)
        self.assertEqual(np.block([c1, c1]).integral, False)
        self.assertEqual(np.block([c2, c2]).integral, True)
        self.assertEqual(np.block([c1, c2]).integral, False)

        self.assertEqual(np.sum(c2).integral, True)
        self.assertEqual(np.sum(c2, axis=0).integral, True)
        self.assertEqual(np.sum(c1).integral, False)
        self.assertEqual(np.sum(c1, axis=0).integral, False)

        self.assertEqual(mpc.np_sgn(c1).integral, True)
        self.assertEqual(mpc.np_sgn(c2).integral, True)
        self.assertEqual(np.absolute(c1).integral, False)
        self.assertEqual(np.absolute(c2).integral, True)

        self.assertEqual(np.minimum(c1, c2).integral, False)
        self.assertEqual(np.minimum(c1, c1).integral, False)
        self.assertEqual(np.minimum(c2, c2).integral, True)
        self.assertEqual(np.maximum(c1, c2).integral, False)
        self.assertEqual(np.maximum(c1, c1).integral, False)
        self.assertEqual(np.maximum(c2, c2).integral, True)

        self.assertEqual(np.amin(c2).integral, True)
        self.assertEqual(np.amin(c2, axis=0).integral, True)
        self.assertEqual(np.amin(c1).integral, False)
        self.assertEqual(np.amin(c1, axis=0).integral, False)
        self.assertEqual(np.amax(c2).integral, True)
        self.assertEqual(np.amax(c2, axis=0).integral, True)
        self.assertEqual(np.amax(c1).integral, False)
        self.assertEqual(np.amax(c1, axis=0).integral, False)

        self.assertEqual(np.argmin(c2).integral, True)
        self.assertEqual(np.argmin(c2, axis=0).integral, True)
        self.assertEqual(np.argmin(c1).integral, True)
        self.assertEqual(np.argmin(c1, axis=0).integral, True)
        self.assertEqual(np.argmax(c2).integral, True)
        self.assertEqual(np.argmax(c2, axis=0).integral, True)
        self.assertEqual(np.argmax(c1).integral, True)
        self.assertEqual(np.argmax(c1, axis=0).integral, True)

        self.assertEqual(np.flip(c1).integral, False)
        self.assertEqual(np.flip(c2).integral, True)
        self.assertEqual(np.fliplr(c1).integral, False)
        self.assertEqual(np.fliplr(c2).integral, True)
        self.assertEqual(np.flipud(c1).integral, False)
        self.assertEqual(np.flipud(c2).integral, True)
        self.assertEqual(np.reshape(c1, -1).integral, False)
        self.assertEqual(np.reshape(c2, -1).integral, True)
        self.assertEqual(np.roll(c1, -1).integral, False)
        self.assertEqual(np.roll(c2, -1).integral, True)
        self.assertEqual(np.rot90(c1).integral, False)
        self.assertEqual(np.rot90(c2).integral, True)
        self.assertEqual(c1.swapaxes(0, 1).integral, False)
        self.assertEqual(c2.swapaxes(0, 1).integral, True)
        self.assertEqual(c1.transpose().integral, False)
        self.assertEqual(c2.transpose().integral, True)

        self.assertEqual(np.split(c1, 2, 1)[0].integral, False)
        self.assertEqual(np.split(c2, 2, 1)[0].integral, True)
        c1_3d = np.stack((c1, c1))
        c2_3d = np.stack((c2, c2))
        self.assertEqual(np.dsplit(c1_3d, 1)[0].integral, False)
        self.assertEqual(np.dsplit(c2_3d, 1)[0].integral, True)
        self.assertEqual(np.hsplit(c1, 1)[0].integral, False)
        self.assertEqual(np.hsplit(c2, 1)[0].integral, True)
        self.assertEqual(np.vsplit(c1, 1)[0].integral, False)
        self.assertEqual(np.vsplit(c2, 1)[0].integral, True)

        self.assertEqual(np.concatenate((c1, c1)).integral, False)
        self.assertEqual(np.concatenate((c2, c2)).integral, True)
        self.assertEqual(np.concatenate((c1, c2)).integral, False)
        self.assertEqual(np.append(c1, c1).integral, False)
        self.assertEqual(np.append(c2, c2).integral, True)
        self.assertEqual(np.append(c1, c2).integral, False)

        c3 = c1.copy()
        c3 = mpc.np_update(c3, (0, 0), secfxp(3))
        self.assertEqual(c3.integral, False)
        self.assertEqual(mpc.run(mpc.output(c3[0, 0])), 3)
        c3 = c2.copy()
        c3 = mpc.np_update(c3, (0, 0), secfxp(3))
        self.assertEqual(c3.integral, True)
        self.assertEqual(mpc.run(mpc.output(c3[0, 0])), 3)
        c1 = c1.copy()
        c3 = mpc.np_update(c3, (0, 0), secfxp(3.5))
        self.assertEqual(mpc.run(mpc.output(c3[0, 0])), 3.5)
        self.assertEqual(c3.integral, False)
        c3 = c2.copy()
        c3 = mpc.np_update(c3, (0, 0), secfxp(3.5))
        self.assertEqual(mpc.run(mpc.output(c3[0, 0])), 3.5)
        self.assertEqual(c3.integral, False)

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_secfld_array(self):
        np.assertEqual = np.testing.assert_array_equal

        secfld = mpc.SecFld(2**2)
        c = secfld.array(np.array([[-3, 0], [1, 2]]))
        mpc.peek(c)
        np.assertEqual(mpc.run(mpc.output(mpc.np_to_bits(c))), [[[1, 1], [0, 0]], [[1, 0], [0, 1]]])
        np.assertEqual(mpc.run(mpc.output(mpc.np_from_bits(mpc.np_to_bits(c[1])))), [1, 2])
        c = mpc._np_randoms(secfld, 5)
        np.assertEqual(mpc.run(mpc.output(np.equal(c**6, c**3))), True)
        c = mpc.np_random_bits(secfld, 15)
        np.assertEqual(mpc.run(mpc.output(np.equal(c**2, c))), True)
        a = secfld.field.array([1, 2, 3] * 10)
        c = secfld.array(a)
        np.assertEqual(mpc.run(mpc.output(1/c)), 1/a)

        secfld = mpc.SecFld(3**2)
        self.assertRaises(TypeError, mpc.np_to_bits, mpc.SecFld(3**2).array(np.array(2)))
        c = mpc.np_random_bits(secfld, 15)
        np.assertEqual(mpc.run(mpc.output(np.equal(c**2, c))), True)

        secfld = mpc.SecFld(min_order=2**16)
        a = np.array([[[-1, 0], [0, -1]]])
        c = secfld.array(a)
        np.assertEqual(mpc.run(mpc.np_is_zero_public(c)), a == 0)
        self.assertEqual(mpc.run(mpc.output(c.flatten().tolist())), [-1, 0, 0, -1])
        np.assertEqual(mpc.run(mpc.output(np.outer(a, c))), np.outer(a, a))
        np.assertEqual(mpc.run(mpc.output(np.roll(c, 1))), np.roll(a, 1))
        self.assertEqual(len(c), 1)
        self.assertEqual(len(c.T), 2)
        self.assertTrue(bool(c))
        np.assertEqual(mpc.run(mpc.output(np.equal(c, c))), True)
        np.assertEqual(mpc.run(mpc.output(np.equal(c, c+1))), False)
        np.assertEqual(mpc.run(mpc.output(1/(c-1))), 1/(secfld.field.array(a)-1))

    @unittest.skipIf(not np, 'NumPy not available or inside MPyC disabled')
    def test_np_errors(self):
        secfld = mpc.SecFld(2)
        c = secfld.array(np.array([[1, 1], [0, 0]]))
        self.assertRaises(ValueError, c.reshape, -1, -1)
        self.assertRaises(ValueError, c.reshape, 3, -1)
        self.assertRaises(ValueError, np.rot90, c, 1, (1,))
        self.assertRaises(ValueError, np.rot90, c, 1, (1, 1))
        self.assertRaises(ValueError, np.rot90, c, 1, (1, 2))

    def test_async(self):
        mpc.options.no_async = False
        a = mpc.SecInt()(7)
        b = a * a
        mpc.run(mpc.barrier())
        mpc.run(mpc.throttler(0.5))
        mpc.run(mpc.throttler(0.5))
        self.assertRaises(ValueError, mpc.run, mpc.throttler(1.5))
        self.assertEqual(mpc.run(mpc.output(b)), 49)
        self.assertEqual(mpc.run(mpc.output(mpc.scalar_mul(a, [-a, a]))), [-49, 49])
        mpc.options.no_async = True

    @unittest.skipIf(mpc.options.no_prss, 'PRSS (pseudorandom secret sharing) disabled')
    def test_prss_keys(self):
        from mpyc.runtime import Party, Runtime
        p0 = Party(0)
        p1 = Party(1)
        rt0 = Runtime(0, [p0, p1], mpc.options)
        rt1 = Runtime(1, [p0, p1], mpc.options)
        rt1._prss_keys_from_peer(0, rt0._prss_keys_to_peer(1)[0])
        self.assertEqual(rt0._prss_keys, rt1._prss_keys)

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
        seccl = mpc.SecClassGroup(-23)
        a = seccl.group((2, 1))
        # NB: mpc.transfer() calls pickle.dumps() and pickle.loads()
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(xsecfld(12), senders=0)))), 12)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(psecfld(12), senders=0)))), 12)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(secint(12), senders=0)))), 12)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(secfxp(12.5), senders=0)))), 12.5)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(secflt(12.5), senders=0)))), 12.5)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(seccl((2, 1, 3)), senders=0)))), a)
        self.assertEqual(mpc.run(mpc.transfer(xsecfld.field(12), senders=0)), 12)
        self.assertEqual(mpc.run(mpc.transfer(psecfld.field(12), senders=0)), 12)
        self.assertEqual(mpc.run(mpc.transfer(secint.field(12), senders=0)), 12)
        self.assertEqual(mpc.run(mpc.transfer(secfxp.field(13), senders=0)), 13)
        self.assertEqual(mpc.run(mpc.transfer(xsecfld.field.modulus, 0)), xsecfld.field.modulus)

        x = [(xsecfld(12), psecfld(12), secint(12), secfxp(12.5), secflt(12.5), seccl((2, 1, 3))),
             [xsecfld.field(12), psecfld.field(12), secint.field(12), secfxp.field(13), a],
             xsecfld.field.modulus]
        y = mpc.run(mpc.transfer(x, senders=0))
        self.assertTrue(all(mpc.run(mpc.output(a == b)) for a, b in zip(y[0], x[0])))
        self.assertEqual(y[1], x[1])
        self.assertEqual(y[2], x[2])

        if not np:
            return

        a = np.array(12)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(xsecfld.array(a), 0)))), 12)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(psecfld.array(a), 0)))), 12)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(secint.array(a), senders=0)))), 12)
        a = np.array(12.5)
        self.assertEqual(mpc.run(mpc.output(mpc.run(mpc.transfer(secfxp.array(a), 0)))), 12.5)
        self.assertEqual(mpc.run(mpc.transfer(xsecfld.field.array(12), senders=0)), 12)
        self.assertEqual(mpc.run(mpc.transfer(psecfld.field.array(12), senders=0)), 12)
        self.assertEqual(mpc.run(mpc.transfer(secint.field.array(12), senders=0)), 12)
        self.assertEqual(mpc.run(mpc.transfer(secfxp.field.array(13), senders=0)), 13)

    def test_psecfld(self):
        secfld = mpc.SecFld(min_order=2**16)
        a = secfld(1)
        b = secfld(0)
        self.assertEqual(mpc.run(mpc.output(a + b)), 1)
        self.assertEqual(mpc.run(mpc.output(a * b)), 0)

        self.assertEqual(mpc.run(mpc.output(mpc.to_bits(secfld(0), 8))), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mpc.run(mpc.output(mpc.to_bits(secfld(255), 8))), [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(mpc.run(mpc.output(mpc.to_bits(secfld(31), 17)))[:6], [1, 1, 1, 1, 1, 0])

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
        self.assertEqual(mpc.run(mpc.output(0 ^ b)), 1)
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
        self.assertRaises(TypeError, mpc.to_bits, a)

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
        mpc.peek(b)
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
        self.assertAlmostEqual(mpc.run(mpc.output(mpc.trunc(secint(50), 2))), 12, delta=1)
        self.assertAlmostEqual(mpc.run(mpc.output(mpc.trunc(secint(-50), 2))), -13, delta=1)
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

        # Test integral property
        c1 = secfxp(8)
        self.assertEqual(c1.integral, True)
        self.assertEqual((c1*0.5).integral, False)
        c2 = secfxp(1.5)
        self.assertEqual(c2.integral, False)
        self.assertEqual((c1*c2).integral, False)
        self.assertEqual((c1+c2).integral, False)
        c2 = secfxp(1)
        self.assertEqual((c1*c2).integral, True)
        self.assertEqual((c1+c2).integral, True)

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
            X = [x]
            self.assertEqual(mpc.run(mpc.output(mpc.matrix_prod(X, X, True)[0])), [t])
            U = [mpc.unit_vector(secfxp(3), 4)]
            self.assertEqual(mpc.run(mpc.output(mpc.matrix_prod(X, U, True)[0])), [s[3]])
            self.assertEqual(mpc.run(mpc.output(mpc.matrix_prod(U, X, True)[0])), [s[3]])
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
            self.assertEqual(mpc.run(mpc.output(secfxp(2) / secfxp.field(2))), 1)

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

            if f != 64:
                delta = 1.2 * 2**-f
                self.assertAlmostEqual(mpc.run(mpc.output(mpc.sin(secfxp(math.pi/16)))),
                                       math.sin(math.pi/16), delta=delta)
                self.assertAlmostEqual(mpc.run(mpc.output(mpc.cos(secfxp(-math.pi/8)))),
                                       math.cos(-math.pi/8), delta=delta)
                self.assertAlmostEqual(mpc.run(mpc.output(mpc.sin(secfxp(-math.pi/4)))),
                                       math.sin(-math.pi/4), delta=delta)
                self.assertAlmostEqual(mpc.run(mpc.output(mpc.cos(secfxp(math.pi/2)))),
                                       math.cos(math.pi/2), delta=delta)
                self.assertAlmostEqual(mpc.run(mpc.output(mpc.sin(secfxp(1)))),
                                       math.sin(1), delta=delta)
                self.assertAlmostEqual(mpc.run(mpc.output(mpc.cos(secfxp(-2)))),
                                       math.cos(-2), delta=delta)
                self.assertAlmostEqual(mpc.run(mpc.output(mpc.tan(secfxp(0.5)))),
                                       math.tan(0.5), delta=math.sqrt(1/delta)*delta)
                self.assertAlmostEqual(mpc.run(mpc.output(mpc.tan(secfxp(2)))),
                                       math.tan(2), delta=2*math.sqrt(1/delta)*delta)

    def test_secflt(self):
        secflt = mpc.SecFlt()
        a = secflt(1.25)
        b = secflt(2.5)
        mpc.peek(b)
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
        self.assertEqual(mpc.run(mpc.output(c.if_swap([a, b], [b, a])[0])), [1, -1])
        self.assertEqual(mpc.run(mpc.output(mpc.if_else(1-c, b, b))), 1)
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
        self.assertEqual(mpc.input([]), [[]])
        self.assertEqual(mpc.input([], senders=0), [])
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

    def test_gcd(self):
        secint = mpc.SecInt(16)
        self.assertEqual(mpc.run(mpc.output(mpc.trailing_zeros(secint(0)))), [0]*16)
        self.assertEqual(mpc.run(mpc.output(mpc.trailing_zeros(secint(7))[0])), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.trailing_zeros(secint(-6))[:2])), [0, 1])
        self.assertEqual(mpc.run(mpc.output(mpc.trailing_zeros(secint(4))[:3])), [0, 0, 1])
        self.assertEqual(mpc.run(mpc.output(mpc.trailing_zeros(secint(-5664))[:6])), [0]*5 + [1])
        self.assertEqual(mpc.run(mpc.output(mpc.gcp2(secint(0), secint(0)))), 1<<16)
        self.assertEqual(mpc.run(mpc.output(mpc.gcp2(secint(0), secint(-64)))), 64)
        self.assertEqual(mpc.run(mpc.output(mpc.gcp2(secint(5664), secint(64)))), 32)
        self.assertEqual(mpc.run(mpc.output(mpc.gcp2(secint(1), secint(-64)))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.gcd(secint(0), secint(0)))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.gcd(secint(0), secint(3), l=3))), 3)
        self.assertEqual(mpc.run(mpc.output(mpc.gcd(secint(-33), secint(30)))), 3)
        self.assertEqual(mpc.run(mpc.output(mpc.gcd(secint(-2**15), secint(1)))), 1)
        self.assertEqual(mpc.run(mpc.output(mpc.lcm(secint(0), secint(0)))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.lcm(secint(-33), secint(0)))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.lcm(secint(-66), secint(30)))), 330)
        self.assertEqual(mpc.run(mpc.output(mpc.lcm(secint(-120), secint(60), l=8))), 120)
        self.assertEqual(mpc.run(mpc.output(mpc.inverse(secint(10), secint(1)))), 0)
        self.assertEqual(mpc.run(mpc.output(mpc.inverse(secint(11), secint(16), l=6))), 3)
        self.assertEqual(mpc.run(mpc.output(mpc.inverse(secint(1234), secint(6789)))), 5089)
        self.assertEqual(mpc.run(mpc.output(mpc.inverse(secint(1234+6789), secint(6789)))), 5089)
        self.assertEqual(mpc.run(mpc.output(list(mpc.gcdext(secint(0), secint(0))))), [0, 0, 0])
        for a, b in ((-300, -300), (2345, 2345), (60, -88), (-360, 9), (0, 256)):
            g, s, t = mpc.run(mpc.output(list(mpc.gcdext(secint(a), secint(b)))))
            self.assertTrue(g == math.gcd(a, b) and g == s * a + t * b)

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
        s = [0, -1, 1, -2, 2, -3]
        x = [secint(i) for i in range(-3, 3)]
        self.assertEqual(mpc.run(mpc.output(mpc.in_prod(list(map(secint.field, s)), x))), -3)
        x = (secint(i) for i in range(-3, 3))
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

        secfld = mpc.SecFld(2)
        for _ in range(5):
            self.assertEqual(mpc.run(mpc.output(1/secfld(1))), 1)


if __name__ == "__main__":
    unittest.main()

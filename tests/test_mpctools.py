import unittest
import operator
import itertools
import functools
from mpyc.runtime import mpc
import mpyc.mpctools


class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mpc.logging(False)

    def test_reduce(self):
        secint = mpc.SecInt()
        r = range(1, 9)
        x = [secint(i) for i in r]
        y = [[secint(i)]*2 for i in r]
        z = [[[secint(i)]*2]*2 for i in r]
        for red in functools.reduce, mpyc.mpctools.reduce:
            self.assertEqual(mpc.run(mpc.output(red(mpc.add, x))), 36)
            self.assertEqual(mpc.run(mpc.output(red(mpc.mul, x))), 40320)
            self.assertEqual(mpc.run(mpc.output(red(mpc.max, x))), 8)
            self.assertEqual(mpc.run(mpc.output(red(mpc.min, x, secint(0)))), 0)
            self.assertEqual(mpc.run(mpc.output(red(mpc.max, x, secint(10)))), 10)
            self.assertEqual(mpc.run(mpc.output(red(mpc.vector_add, y))), [36]*2)
            self.assertEqual(mpc.run(mpc.output(red(mpc.schur_prod, y))), [40320]*2)
            self.assertEqual(mpc.run(mpc.output(red(mpc.matrix_add, z)[0])), [36]*2)
            self.assertEqual(mpc.run(mpc.output(red(mpc.matrix_prod, z)[1])), [5160960]*2)

    def test_accumulate(self):
        secint = mpc.SecInt()
        r = range(1, 9)
        x = [secint(i) for i in r]
        for acc in itertools.accumulate, mpyc.mpctools.accumulate:
            self.assertEqual(mpc.run(mpc.output(list(acc(x)))),
                             list(itertools.accumulate(r)))
            self.assertEqual(mpc.run(mpc.output(list(acc(x, mpc.mul)))),
                             list(itertools.accumulate(r, operator.mul)))
            self.assertEqual(mpc.run(mpc.output(list(acc(x, mpc.min)))),
                             list(itertools.accumulate(r, min)))
            self.assertEqual(mpc.run(mpc.output(list(acc(x, mpc.max)))),
                             list(itertools.accumulate(r, max)))
        a = secint(10)
        self.assertEqual(mpc.run(mpc.output(list(acc(itertools.repeat(a, 5), mpc.mul, secint(1))))),
                         [1, 10, 10**2, 10**3, 10**4, 10**5])


if __name__ == "__main__":
    unittest.main()

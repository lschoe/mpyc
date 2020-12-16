import unittest
import random
import statistics
from mpyc.runtime import mpc
from mpyc.statistics import (mean, variance, stdev, pvariance, pstdev,
                             mode, median, median_low, median_high)


class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mpc.logging(False)

    def test_plain(self):
        f = lambda: (i * j for i in range(-1, 2, 1) for j in range(2, -2, -1))
        self.assertEqual(mean(f()), statistics.mean(f()))
        self.assertEqual(variance(f()), statistics.variance(f()))
        self.assertEqual(stdev(f()), statistics.stdev(f()))
        self.assertEqual(pvariance(f()), statistics.pvariance(f()))
        self.assertEqual(pstdev(f()), statistics.pstdev(f()))
        self.assertEqual(mode(f()), statistics.mode(f()))
        self.assertEqual(median(f()), statistics.median(f()))

    def test_statistics_error(self):
        self.assertRaises(statistics.StatisticsError, mean, [])
        self.assertRaises(statistics.StatisticsError, variance, [0])
        self.assertRaises(statistics.StatisticsError, stdev, [0])
        self.assertRaises(statistics.StatisticsError, pvariance, [])
        self.assertRaises(statistics.StatisticsError, pstdev, [])
        self.assertRaises(statistics.StatisticsError, mode, [])
        self.assertRaises(statistics.StatisticsError, median, [])

    def test_secfld(self):
        secfld = mpc.SecFld()
        x = [secfld(0), secfld(0)]
        self.assertRaises(TypeError, mean, x)
        self.assertRaises(TypeError, variance, x)
        self.assertRaises(TypeError, stdev, x)
        self.assertRaises(TypeError, mode, x)
        self.assertRaises(TypeError, median, x)

    def test_secint(self):
        secint = mpc.SecInt()
        y = [1, 3, -2, 3, 1, -2, -2, 4] * 5
        random.shuffle(y)
        x = list(map(secint, y))
        self.assertEqual(mpc.run(mpc.output(mean(x))), round(statistics.mean(y)))
        self.assertEqual(mpc.run(mpc.output(variance(x))), round(statistics.variance(y)))
        self.assertEqual(mpc.run(mpc.output(variance(x, mean(x)))), round(statistics.variance(y)))
        self.assertEqual(mpc.run(mpc.output(stdev(x))), round(statistics.stdev(y)))
        self.assertEqual(mpc.run(mpc.output(pvariance(x))), round(statistics.pvariance(y)))
        self.assertEqual(mpc.run(mpc.output(pstdev(x))), round(statistics.pstdev(y)))
        self.assertEqual(mpc.run(mpc.output(mode(x))), round(statistics.mode(y)))
        self.assertEqual(mpc.run(mpc.output(median(x))), round(statistics.median(y)))
        self.assertEqual(mpc.run(mpc.output(median_low(x))), round(statistics.median_low(y)))
        self.assertEqual(mpc.run(mpc.output(median_high(x))), round(statistics.median_high(y)))

    def test_secfxp(self):
        secfxp = mpc.SecFxp()
        x = [1, 1, 2, 2, 3, 4, 4, 4, 6] * 5
        random.shuffle(x)
        x = list(map(secfxp, x))
        self.assertAlmostEqual(mpc.run(mpc.output(mean(x))), 3, delta=1)
        self.assertAlmostEqual(mpc.run(mpc.output(median(x))), 3)
        self.assertAlmostEqual(mpc.run(mpc.output(mode(x))), 4)

        x = [1, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6] * 100
        random.shuffle(x)
        x = list(map(lambda a: a * 2**-4, x))
        x = list(map(secfxp, x))
        self.assertAlmostEqual(mpc.run(mpc.output(mean(x))), (2**-4) * 10/3, delta=1)

        y = [1.75, 1.25, -0.25, 0.5, 1.25, -3.5] * 5
        random.shuffle(y)
        x = list(map(secfxp, y))
        self.assertAlmostEqual(mpc.run(mpc.output(mean(x))), statistics.mean(y), 4)
        self.assertAlmostEqual(mpc.run(mpc.output(variance(x))), statistics.variance(y), 2)
        self.assertAlmostEqual(mpc.run(mpc.output(stdev(x))), statistics.stdev(y), 3)
        self.assertAlmostEqual(mpc.run(mpc.output(pvariance(x))), statistics.pvariance(y), 2)
        self.assertAlmostEqual(mpc.run(mpc.output(pstdev(x))), statistics.pstdev(y), 3)
        self.assertAlmostEqual(mpc.run(mpc.output(median(x))), statistics.median(y), 4)

        x = list(map(secfxp, [1.0]*10))
        self.assertAlmostEqual(mpc.run(mpc.output(mode(x))), 1)
        k = mpc.options.sec_param
        mpc.options.sec_param = 1  # force no privacy case
        self.assertAlmostEqual(mpc.run(mpc.output(mode(x))), 1)
        mpc.options.sec_param = k

        x[0] = secfxp(1.5)
        self.assertRaises(ValueError, mode, x)


if __name__ == "__main__":
    unittest.main()

import unittest
import random
import statistics
from mpyc.runtime import mpc
from mpyc.statistics import mean, variance, stdev, pvariance, pstdev, mode, median


class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mpc.logging(False)
        mpc.run(mpc.start())

    @classmethod
    def tearDownClass(cls):
        mpc.run(mpc.shutdown())

    def test_plain(self):
        x = [1.23, 2.34, 5.67, 5.67]
        self.assertEqual(mean(x), statistics.mean(x))
        self.assertEqual(variance(x), statistics.variance(x))
        self.assertEqual(stdev(x), statistics.stdev(x))
        self.assertEqual(pvariance(x), statistics.pvariance(x))
        self.assertEqual(pstdev(x), statistics.pstdev(x))
        self.assertEqual(mode(x), statistics.mode(x))
        self.assertEqual(median(x), statistics.median(x))

    def test_secint(self):
        secint = mpc.SecInt()
        y = [1, 3, -2, 3, 1, -2, -2, 4] * 5
        random.shuffle(y)
        x = list(map(secint, y))
        self.assertEqual(mpc.run(mpc.output(mean(x))), round(statistics.mean(y)))
        self.assertEqual(mpc.run(mpc.output(variance(x))), round(statistics.variance(y)))
        self.assertEqual(mpc.run(mpc.output(stdev(x))), round(statistics.stdev(y)))
        self.assertEqual(mpc.run(mpc.output(pvariance(x))), round(statistics.pvariance(y)))
        self.assertEqual(mpc.run(mpc.output(pstdev(x))), round(statistics.pstdev(y)))
        self.assertEqual(mpc.run(mpc.output(mode(x))), round(statistics.mode(y)))
        self.assertEqual(mpc.run(mpc.output(median(x))), round(statistics.median(y)))

    def test_secfxp(self):
        secfxp = mpc.SecFxp()
        x = [1, 1, 2, 2, 3, 4, 4, 4, 6] * 5
        random.shuffle(x)
        x = list(map(secfxp, x))
        self.assertAlmostEqual(mpc.run(mpc.output(mean(x))).signed(), 3, delta=1)
        self.assertAlmostEqual(mpc.run(mpc.output(median(x))).signed(), 3)
        self.assertAlmostEqual(mpc.run(mpc.output(mode(x))).signed(), 4)

        x = [1, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6] * 100
        random.shuffle(x)
        x = list(map(lambda a: a * 2**-4, x))
        x = list(map(secfxp, x))
        self.assertAlmostEqual(mpc.run(mpc.output(mean(x))).signed(), (2**-4) * 10/3, delta=1)

        y = [2.75, 1.75, 1.25, -0.25, 0.5, 1.25, -3.5] * 5
        random.shuffle(y)
        x = list(map(secfxp, y))
        self.assertAlmostEqual(float(mpc.run(mpc.output(mean(x)))), statistics.mean(y), places=4)
        self.assertAlmostEqual(float(mpc.run(mpc.output(variance(x)))), statistics.variance(y), places=2)
        self.assertAlmostEqual(float(mpc.run(mpc.output(stdev(x)))), statistics.stdev(y), places=3)
        self.assertAlmostEqual(float(mpc.run(mpc.output(pvariance(x)))), statistics.pvariance(y), places=2)
        self.assertAlmostEqual(float(mpc.run(mpc.output(pstdev(x)))), statistics.pstdev(y), places=3)
        self.assertAlmostEqual(float(mpc.run(mpc.output(median(x)))), statistics.median(y), places=4)
        
        x = list(map(secfxp, [1.0] * 10))
        self.assertAlmostEqual(mpc.run(mpc.output(mode(x))).signed(), 1)


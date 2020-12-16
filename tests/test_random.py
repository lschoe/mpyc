import unittest
from mpyc.runtime import mpc
from mpyc.random import (getrandbits, randrange, random_unit_vector, randint,
                         shuffle, random_permutation, random_derangement,
                         choice, choices, sample, random, uniform)


class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mpc.logging(False)

    def test_secint(self):
        secint = mpc.SecInt()
        a = mpc.run(mpc.output(getrandbits(secint, 10)))
        self.assertGreaterEqual(a, 0)
        self.assertLessEqual(a, 2**10 - 1)

        x = mpc.run(mpc.output(random_unit_vector(secint, 4)))
        self.assertEqual(sum(x), 1)

        a = mpc.run(mpc.output(randrange(secint, 37)))  # French roulette
        self.assertGreaterEqual(a, 0)
        self.assertLessEqual(a, 36)
        a = mpc.run(mpc.output(randrange(secint, 1, 7)))  # one die
        self.assertGreaterEqual(a, 1)
        self.assertLessEqual(a, 6)
        a = mpc.run(mpc.output(randrange(secint, 1, 7) + randrange(secint, 1, 7)))  # two dice
        self.assertGreaterEqual(a, 2)
        self.assertLessEqual(a, 12)
        a = mpc.run(mpc.output(randrange(secint, 1, 32, 2)))
        self.assertGreaterEqual(a, 1)
        self.assertLessEqual(a, 31)
        self.assertEqual(a % 2, 1)
        a = mpc.run(mpc.output(randint(secint, -1, 1)))
        self.assertIn(a, [-1, 0, 1])

        x = list(range(8))
        shuffle(secint, x)
        shuffle(secint, x)
        x = mpc.run(mpc.output(x))
        self.assertSetEqual(set(x), set(range(8)))
        x = mpc.run(mpc.output(random_permutation(secint, 8)))
        self.assertSetEqual(set(x), set(range(8)))
        x = mpc.run(mpc.output(random_derangement(secint, 2)))
        self.assertListEqual(x, [1, 0])
        x = mpc.run(mpc.output(random_derangement(secint, range(1, 9))))
        self.assertSetEqual(set(x), set(range(1, 9)))

        x = mpc.run(mpc.output(choice(secint, [1, 2, 3, 4, 5])))
        self.assertIn(x, [1, 2, 3, 4, 5])
        x = mpc.run(mpc.output(choice(secint, [secint(-100), secint(200), secint(-300)])))
        self.assertIn(x, [-100, 200, -300])
        x = mpc.run(mpc.output(mpc.sum(choices(secint, [0, 1], k=5))))  # uniform
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 5)
        x = mpc.run(mpc.output(mpc.sum(choices(secint, [0, 1], [25, 75]))))  # Bernoulli
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 1)
        x = mpc.run(mpc.output(mpc.sum(choices(secint, [0, 1], [25, 75], k=10))))  # binomial
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 10)

        x = mpc.run(mpc.output(sample(secint, [2**i for i in range(16)], 8)))
        self.assertGreaterEqual(sum(x), 2**8 - 1)
        self.assertLessEqual(sum(x), 2**8 * (2**8 - 1))
        x = mpc.run(mpc.output(sample(secint, range(10**8), 10)))
        self.assertGreaterEqual(min(x), 0)
        self.assertLessEqual(max(x), 10**8 - 1)

        x = mpc.run(mpc.output(sample(secint, range(1000000, 3000000, 1000), 10)))
        self.assertGreaterEqual(min(x) // 1000, 1000)
        self.assertLessEqual(max(x) // 1000, 3000)
        self.assertEqual(sum(x) % 1000, 0)

    def test_secfxp(self):
        secfxp = mpc.SecFxp()
        a = getrandbits(secfxp, 10)
        self.assertTrue(a.integral)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 2**10 - 1)

        x = mpc.run(mpc.output(random_unit_vector(secfxp, 3)))
        self.assertEqual(int(sum(x)), 1)

        x = mpc.run(mpc.output(random_permutation(secfxp, range(1, 9))))
        self.assertSetEqual(set(map(int, x)), set(range(1, 9)))
        x = mpc.run(mpc.output(random_derangement(secfxp, [0.0, 1.0])))
        self.assertListEqual(list(map(int, x)), [1, 0])

        a = mpc.run(mpc.output(choice(secfxp, [0.08, 0.09, 0.1, 0.11, 0.12])))
        self.assertAlmostEqual(a, 0.1, 1)

        a = mpc.run(mpc.output(random(secfxp)))
        self.assertGreaterEqual(a, 0)
        self.assertLessEqual(a, 1)
        a = mpc.run(mpc.output(uniform(secfxp, 13.13, 13.17)))
        self.assertGreaterEqual(a, 13.13)
        self.assertLessEqual(a, 13.17)
        a = mpc.run(mpc.output(uniform(secfxp, -13.13, -13.17)))
        self.assertGreaterEqual(a, -13.17)
        self.assertLessEqual(a, -13.13)

    def test_secfld(self):
        secfld = mpc.SecFld(2, char=2)
        a = getrandbits(secfld, 1, True)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a[0]), 0)
        self.assertLessEqual(int(a[0]), 1)

        secfld = mpc.SecFld(3)
        a = getrandbits(secfld, 1)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 1)
        a = mpc.run(mpc.output(randrange(secfld, 1, 3)))
        self.assertGreaterEqual(int(a), 1)
        self.assertLessEqual(int(a), 2)

        secfld = mpc.SecFld(256)
        a = getrandbits(secfld, 8)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 2**8 - 1)
        a = mpc.run(mpc.output(randrange(secfld, 256)))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 2**8 - 1)
        a = mpc.run(mpc.output(randint(secfld, 0, 2)))
        self.assertIn(a, [0, 1, 2])
        x = mpc.run(mpc.output(random_unit_vector(secfld, 2)))
        self.assertEqual(int(sum(x)), 1)
        x = mpc.run(mpc.output(random_permutation(secfld, range(1, 9))))
        self.assertSetEqual(set(map(int, x)), set(range(1, 9)))
        x = mpc.run(mpc.output(random_derangement(secfld, range(2))))
        self.assertListEqual(list(map(int, x)), [1, 0])

        secfld = mpc.SecFld(257)
        a = getrandbits(secfld, 8)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 2**8 - 1)
        a = mpc.run(mpc.output(randrange(secfld, 257)))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 2**8)
        a = mpc.run(mpc.output(randint(secfld, -1, 1)))
        self.assertIn(a, [-1, 0, 1])
        x = mpc.run(mpc.output(random_unit_vector(secfld, 1)))
        self.assertEqual(int(sum(x)), 1)
        x = mpc.run(mpc.output(random_permutation(secfld, range(1, 9))))
        self.assertSetEqual(set(map(int, x)), set(range(1, 9)))
        x = mpc.run(mpc.output(random_derangement(secfld, range(2))))
        self.assertListEqual(list(map(int, x)), [1, 0])

    def test_errors(self):
        secint = mpc.SecInt()
        self.assertRaises(ValueError, randrange, secint, 0)
        self.assertRaises(IndexError, choice, secint, [])
        self.assertRaises(TypeError, choices, secint, [], weights=[1], cum_weights=[1])
        self.assertRaises(ValueError, choices, secint, [], weights=[1])
        self.assertRaises(ValueError, sample, secint, [], 1)
        self.assertRaises(TypeError, random, secint)
        self.assertRaises(TypeError, uniform, secint, -0.5, 0.5)


if __name__ == "__main__":
    unittest.main()

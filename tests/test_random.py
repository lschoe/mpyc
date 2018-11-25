import unittest
from mpyc.runtime import mpc
from mpyc.random import getrandbits, randrange, random_unit_vector, randint, shuffle, \
                        random_permutation, choice, choices, sample, random, uniform

class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mpc.logging(False)
        mpc.run(mpc.start())

    @classmethod
    def tearDownClass(cls):
        mpc.run(mpc.shutdown())

    def test_secint(self):
        secint = mpc.SecInt()
        a = mpc.run(mpc.output(getrandbits(secint, 10)))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 2**10 -  1)

        x = mpc.run(mpc.output(random_unit_vector(secint, 6)))
        self.assertEqual(sum(x), 1)

        a = mpc.run(mpc.output(randrange(secint, 37))) # French roulette
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 36)
        a = mpc.run(mpc.output(randrange(secint, 1, 7))) # one die
        self.assertGreaterEqual(int(a), 1)
        self.assertLessEqual(int(a), 6)
        a = mpc.run(mpc.output(randrange(secint, 1, 7) + randrange(secint, 1, 7))) # two dice
        self.assertGreaterEqual(int(a), 2)
        self.assertLessEqual(int(a), 12)
        a = mpc.run(mpc.output(randrange(secint, 1, 32, 2)))
        self.assertGreaterEqual(int(a), 1)
        self.assertLessEqual(int(a), 31)
        self.assertEqual(int(a) % 2, 1)
        a = mpc.run(mpc.output(randint(secint, -1, 1)))
        self.assertIn(a, [-1, 0, 1])

        x = list(range(8))
        shuffle(secint, x)
        shuffle(secint, x)
        x = mpc.run(mpc.output(x))
        self.assertSetEqual({a for a in range(8)}, {int(a) for a in x})
        x = mpc.run(mpc.output(random_permutation(secint, 8)))
        self.assertSetEqual({a for a in range(8)}, {int(a) for a in x})
        x = mpc.run(mpc.output(random_permutation(secint, range(1, 9))))
        self.assertSetEqual({a for a in range(1, 9)}, {int(a) for a in x})

        x = mpc.run(mpc.output(choice(secint, [1, 2, 3, 4, 5])))
        self.assertIn(x, [1, 2, 3, 4, 5])
        x = mpc.run(mpc.output(choice(secint, [secint(-100), secint(200), secint(-300)])))
        self.assertIn(x, [-100, 200, -300])
        x = mpc.run(mpc.output(mpc.sum(choices(secint, [0, 1], [25, 75])))) # Bernoulli
        self.assertGreaterEqual(int(x), 0)
        self.assertLessEqual(int(x), 1)
        x = mpc.run(mpc.output(mpc.sum(choices(secint, [0, 1], [25, 75], k=10)))) # binomial
        self.assertGreaterEqual(int(x), 0)
        self.assertLessEqual(int(x), 10)

        x = mpc.run(mpc.output(sample(secint, [2**i for i in range(16)], 8)))
        self.assertGreaterEqual(sum(map(int, x)), 2**8 - 1)
        self.assertLessEqual(sum(map(int, x)), 2**8 * (2**8 - 1))
        x = mpc.run(mpc.output(sample(secint, range(10**8), 10)))
        self.assertGreaterEqual(min(map(int, x)), 0)
        self.assertLessEqual(max(map(int, x)), 10**8 - 1)

        x = mpc.run(mpc.output(sample(secint, range(1000000, 3000000, 1000), 10)))
        self.assertGreaterEqual(min(map(int, x)) // 1000, 1000)
        self.assertLessEqual(max(map(int, x)) // 1000, 3000)
        self.assertEqual(sum(map(int, x)) % 1000, 0)

    def test_secfxp(self):
        secfxp = mpc.SecFxp()
        a = getrandbits(secfxp, 10)
        self.assertTrue(a.integral)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 2**10 -  1)

        x = mpc.run(mpc.output(random_unit_vector(secfxp, 6)))
        self.assertEqual(int(sum(x)), 1)

        x = mpc.run(mpc.output(random_permutation(secfxp, range(1, 9))))
        self.assertSetEqual({a for a in range(1, 9)}, {int(a) for a in x})

        a = mpc.run(mpc.output(choice(secfxp, [0.8, 0.9, 1.0, 1.1, 1.2])))
        self.assertAlmostEqual(float(a), 1.0, 0)
        a = mpc.run(mpc.output(choice(secfxp, [0.08, 0.09, 0.1, 0.11, 0.12])))
        self.assertAlmostEqual(float(a), 0.1, 1)

        a = mpc.run(mpc.output(random(secfxp)))
        self.assertGreaterEqual(float(a), 0)
        self.assertLessEqual(float(a), 1)
        a = mpc.run(mpc.output(uniform(secfxp, 13.13, 13.17)))
        self.assertGreaterEqual(float(a), 13.13)
        self.assertLessEqual(float(a), 13.17)
        a = mpc.run(mpc.output(uniform(secfxp, -13.13, -13.17)))
        self.assertGreaterEqual(float(a), -13.17)
        self.assertLessEqual(float(a), -13.13)

    def test_secfld(self):
        secfld = mpc.SecFld(2, char2=True)
        a = getrandbits(secfld, 1)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 1)

        secfld = mpc.SecFld(3)
        a = getrandbits(secfld, 1)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 1)

        secfld = mpc.SecFld(256)
        a = getrandbits(secfld, 8)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 2**8 -  1)
        a = mpc.run(mpc.output(randint(secfld, 0, 2)))
        self.assertIn(a, [0, 1, 2])
        x = mpc.run(mpc.output(random_unit_vector(secfld, 6)))
        self.assertEqual(int(sum(x)), 1)
        x = mpc.run(mpc.output(random_permutation(secfld, range(1, 9))))
        self.assertSetEqual({a for a in range(1, 9)}, {int(a) for a in x})

        secfld = mpc.SecFld(257)
        a = getrandbits(secfld, 8)
        a = mpc.run(mpc.output(a))
        self.assertGreaterEqual(int(a), 0)
        self.assertLessEqual(int(a), 2**8 -  1)
        a = mpc.run(mpc.output(randint(secfld, -1, 1)))
        self.assertIn(a, [-1, 0, 1])
        x = mpc.run(mpc.output(random_unit_vector(secfld, 6)))
        self.assertEqual(int(sum(x)), 1)
        x = mpc.run(mpc.output(random_permutation(secfld, range(1, 9))))
        self.assertSetEqual({a for a in range(1, 9)}, {int(a) for a in x})

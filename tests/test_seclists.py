import unittest
from mpyc.seclists import seclist, secindex
from mpyc.runtime import mpc


class Arithmetic(unittest.TestCase):

    def test_secfld(self):
        secfld = mpc.SecFld(101)
        s = seclist([], secfld)
        self.assertEqual(s, [])
        s = seclist(sectype=secfld)
        self.assertEqual(s, [])
        s.append(False)
        s.append(secfld(100))
        s[1:2] = (1,)
        s += [2, 3]
        s.reverse()
        s = [5, 4] + s
        s.reverse()
        s = s + [6, 7]
        del s[0]
        self.assertEqual(mpc.run(mpc.output(list(s))), [1, 2, 3, 4, 5, 6, 7])

    def test_secint(self):
        secint = mpc.SecInt()
        s = seclist([], secint)
        self.assertEqual(s, [])
        s = seclist(sectype=secint)
        self.assertEqual(s, [])
        s.append(False)
        s.append(secint(7))
        s[0] = secint(13)
        self.assertEqual(mpc.run(mpc.output(list(s))), [13, 7])  # NB: list to convert from seclist
        i = [secint(0), secint(1)]
        s[i] = 5
        self.assertEqual(mpc.run(mpc.output(s[1])), 5)

        i0 = [secint(1), secint(0)]
        i1 = [secint(0), secint(1)]
        s[i0], s[i1] = s[i1], s[i0]
        self.assertEqual(mpc.run(mpc.output(list(s))), [5, 13])
        s[i0], s[i1] = s[i1], s[i0]
        self.assertEqual(mpc.run(mpc.output(list(s))), [13, 5])
        s[0], s[1] = s[1], s[0]
        self.assertEqual(mpc.run(mpc.output(list(s))), [5, 13])
        s.append(secint(8))
        s.reverse()
        self.assertEqual(mpc.run(mpc.output(list(s))), [8, 13, 5])

        a = s[secint(1)]
        self.assertEqual(mpc.run(mpc.output(a)), 13)
        s[secint(1)] = secint(21)
        self.assertEqual(mpc.run(mpc.output(s[1])), 21)

        s = seclist([0]*7, secint)
        for a in [secint(3)]*3 + [secint(4)]*4:
            s[a] += 1
        self.assertEqual(mpc.run(mpc.output(list(s))), [0, 0, 0, 3, 4, 0, 0])

        self.assertTrue(mpc.run(mpc.output(s.__contains__(0))))
        self.assertFalse(mpc.run(mpc.output(s.__contains__(9))))
        self.assertEqual(mpc.run(mpc.output(s.count(0))), 5)
        self.assertEqual(mpc.run(mpc.output(s.index(4))), 4)
        self.assertRaises(ValueError, s.index, 9)
        self.assertRaises(ValueError, seclist([], secint).index, 0)
        s.sort(lambda a: -a**2, reverse=True)
        self.assertEqual(mpc.run(mpc.output(list(s))), 5*[0] + [3, 4])

    def test_secindex(self):
        secint = mpc.SecInt()
        i = secindex([secint(0), secint(0), secint(1), secint(0)])
        j = secindex([secint(0), secint(1), secint(0)])
        k = i + j

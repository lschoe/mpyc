import operator
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
        s.remove(4)
        s[5] = 9
        del s[2:4]
        self.assertEqual(mpc.run(mpc.output(list(s))), [1, 2, 6, 9])

        secfld2 = mpc.SecFld()
        self.assertRaises(TypeError, seclist, [secfld(1)], secfld2)
        self.assertRaises(ValueError, seclist, [])
        self.assertRaises(TypeError, operator.add, seclist([secfld(1)]), seclist([secfld2(1)]))

    def test_secint(self):
        secint = mpc.SecInt()
        s = seclist([], secint)
        self.assertEqual(s, [])
        s = seclist(sectype=secint)
        self.assertEqual(s, [])
        s.append(False)
        s.sort()
        s.append(secint(7))
        s[0] = secint(13)
        self.assertEqual(mpc.run(mpc.output(list(s))), [13, 7])
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
        s.append(secint(8))     # s = [5, 13, 8]
        s.reverse()             # s = [8, 13, 5]
        s.insert(secint(0), 9)  # s = [9, 8, 13, 5]
        del s[secint(1)]        # s = [9, 13, 5]
        s.pop(secint(2))        # s = [9, 13]
        s.insert(0, 99)         # s = [99, 9, 13]
        s.pop(0)                # s = [9, 13]
        self.assertRaises(ValueError, s.remove, secint(11))
        s *= 2                  # s = [9, 13, 9, 13]
        s.remove(9)             # s = [13, 9, 13]
        s[0:1] = []             # s = [9, 13]
        s = 1 * s + s * 0       # s = [9, 13]
        self.assertEqual(mpc.run(mpc.output(list(s))), [9, 13])
        self.assertEqual(mpc.run(mpc.output(s[secint(1)])), 13)
        s[secint(1)] = secint(21)
        self.assertEqual(mpc.run(mpc.output(s[1])), 21)
        self.assertRaises(IndexError, s.insert, [secint(1), secint(0)], 42)
        self.assertRaises(IndexError, s.pop, [secint(1)])
        self.assertRaises(IndexError, s.__getitem__, [secint(1)])
        self.assertRaises(TypeError, s.__setitem__, [secint(1)], 42.5)
        self.assertRaises(TypeError, s.__setitem__, slice(0, 2), seclist([0], mpc.SecFxp()))
        self.assertRaises(IndexError, s.__setitem__, [secint(1)], 42)
        self.assertRaises(IndexError, s.__delitem__, [secint(1)])

        s = seclist([0]*7, secint)
        for a in [secint(3)]*3 + [secint(4)]*4:
            s[a] += 1
        self.assertEqual(mpc.run(mpc.output(list(s))), [0, 0, 0, 3, 4, 0, 0])

        with self.assertRaises(NotImplementedError):
            0 in s
        self.assertTrue(mpc.run(mpc.output(s.contains(0))))
        self.assertFalse(mpc.run(mpc.output(s.contains(9))))
        self.assertEqual(mpc.run(mpc.output(s.count(0))), 5)
        self.assertEqual(mpc.run(mpc.output(s.find(3))), 3)
        self.assertEqual(mpc.run(mpc.output(s.index(4))), 4)
        self.assertRaises(ValueError, s.index, 9)
        self.assertEqual(mpc.run(mpc.output(seclist([], secint).find(9))), -1)
        self.assertRaises(ValueError, seclist([], secint).index, 0)
        s.sort(lambda a: -a**2, reverse=True)
        s.sort()
        self.assertEqual(mpc.run(mpc.output(list(s))), 5*[0] + [3, 4])
        self.assertFalse(mpc.run(mpc.output(s < s)))
        self.assertTrue(mpc.run(mpc.output(s <= s)))
        self.assertTrue(mpc.run(mpc.output(s == s)))
        self.assertFalse(mpc.run(mpc.output(s > s)))
        self.assertTrue(mpc.run(mpc.output(s >= s)))
        self.assertFalse(mpc.run(mpc.output(s != s)))
        self.assertFalse(mpc.run(mpc.output(s < [])))
        self.assertFalse(mpc.run(mpc.output(s <= [])))
        self.assertTrue(mpc.run(mpc.output(s >= [])))
        self.assertTrue(mpc.run(mpc.output(s > [])))
        self.assertFalse(mpc.run(mpc.output(s < s[:-1])))
        self.assertTrue(mpc.run(mpc.output(s[:-1] < s)))
        self.assertTrue(mpc.run(mpc.output(s[:-1] != s)))
        t = s.copy()
        t[-1] += 1
        self.assertTrue(mpc.run(mpc.output(s < t)))
        t[1] -= 1
        self.assertFalse(mpc.run(mpc.output(s < t)))
        self.assertFalse(mpc.run(mpc.output(s[:-1] <= t)))
        s = seclist([1, 2, 3, 4], secint)
        t = mpc.run(mpc.transfer(s, senders=0))
        self.assertTrue(mpc.run(mpc.output(s == t)))

    def test_secfxp(self):
        secfxp = mpc.SecFxp()
        s = seclist([5, -3, 2, 5, 5], secfxp)
        self.assertFalse(mpc.run(mpc.output(s < s)))
        t = s[:]
        t[-1] += 1
        self.assertTrue(mpc.run(mpc.output(s < t)))

        s = [[1, 0], [0, 1], [0, 0], [1, 1]]
        ss = mpc.sorted([[secfxp(a) for a in _] for _ in s], key=seclist)
        self.assertEqual([mpc.run(mpc.output(_)) for _ in ss], sorted(s))

    def test_secindex(self):
        secint = mpc.SecInt()
        i = secindex([secint(0), secint(0), secint(1), secint(0)])
        j = secindex([secint(0), secint(1), secint(0)])
        k = i + j


if __name__ == "__main__":
    unittest.main()

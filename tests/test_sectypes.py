import operator
import unittest
from mpyc import gfpx
from mpyc import sectypes

X = gfpx.X


class Arithmetic(unittest.TestCase):

    def test_typecaching(self):
        SecFld = sectypes.SecFld
        self.assertEqual(SecFld(), SecFld(2))
        self.assertEqual(SecFld(), SecFld(char=2))
        self.assertEqual(SecFld(), SecFld(min_order=2))
        SecInt = sectypes.SecInt
        self.assertEqual(SecInt(), SecInt(l=32))
        SecFxp = sectypes.SecFxp
        self.assertEqual(SecFxp(), SecFxp(l=32))
        self.assertEqual(SecFxp(), SecFxp(l=32, f=16))

    def test_bool(self):
        self.assertRaises(TypeError, bool, sectypes.SecFld()(0))
        self.assertRaises(TypeError, bool, sectypes.SecInt()(0))
        self.assertRaises(TypeError, bool, sectypes.SecFxp()(0))

    def test_SecFld(self):
        SecFld = sectypes.SecFld
        secfld = SecFld()
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(char=2)
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(modulus=f'{X}')
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(modulus=f'{X}+1')
        self.assertEqual(secfld.field.modulus, 3)
        secfld = SecFld(min_order=2)
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(min_order=2, char=2)
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(modulus=3, min_order=2)
        self.assertEqual(secfld.field.modulus, 3)
        self.assertEqual(secfld.field.order, 3)
        secfld = SecFld(modulus=3, char=2, min_order=2)
        self.assertEqual(secfld.field.modulus, 3)
        self.assertEqual(secfld.field.order, 2)
        secfld = SecFld(order=3, min_order=2)
        self.assertEqual(secfld.field.modulus, 3)
        self.assertEqual(secfld.field.order, 3)
        secfld = SecFld(order=4, char=2, min_order=4)
        self.assertEqual(secfld.field.modulus, 7)
        self.assertEqual(secfld.field.order, 4)
        secfld = SecFld(modulus=f'1+{X}^8+{X}^4+{X}^3+{X}')
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)
        secfld = SecFld(order=256)
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)
        secfld = SecFld(order=256, modulus=283)
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)
        secfld = SecFld(modulus=283, char=2)
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)
        secfld = SecFld(modulus=gfpx.GFpX(2)(283))
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)

        a = secfld(1)
        self.assertRaises(TypeError, abs, a)
        self.assertRaises(TypeError, operator.floordiv, a, a)
        self.assertRaises(TypeError, operator.floordiv, 1, a)  # tests __rfloordiv__
        self.assertRaises(TypeError, operator.mod, a, a)
        self.assertRaises(TypeError, operator.mod, 1, a)  # tests __rmod__
        self.assertRaises(TypeError, divmod, a, a)
        self.assertRaises(TypeError, divmod, 1, a)  # tests __rdivmod__
        self.assertRaises(TypeError, operator.lshift, a, a)
        self.assertRaises(TypeError, operator.lshift, 1, a)  # tests __rlshift__
        self.assertRaises(TypeError, operator.rshift, a, a)
        self.assertRaises(TypeError, operator.rshift, 1, a)  # tests __rrshift__
        self.assertRaises(TypeError, operator.ge, a, a)  # NB: also tests <=
        self.assertRaises(TypeError, operator.gt, a, a)  # NB: also tests <

    def test_SecNum(self):
        sectypes.SecInt(p=2**89 - 1)
        self.assertRaises(ValueError, sectypes.SecInt, p=2**61 - 1)
        sectypes.SecFxp(p=2**89 - 1)
        self.assertRaises(ValueError, sectypes.SecFxp, f=58, p=2**89 - 1)

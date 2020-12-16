import operator
import unittest
from mpyc import gfpx
from mpyc import sectypes
import mpyc.runtime  # NB: to set attribute runtime in sectypes


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
        SecFlt = sectypes.SecFlt
        self.assertEqual(SecFlt(), SecFlt(l=32))
        self.assertEqual(SecFlt(), SecFlt(s=24, e=8))

    def test_bool(self):
        self.assertRaises(TypeError, bool, sectypes.SecFld()(0))
        self.assertRaises(TypeError, bool, sectypes.SecInt()(0))
        self.assertRaises(TypeError, bool, sectypes.SecFxp()(0))
        self.assertRaises(TypeError, bool, sectypes.SecFlt()(0))

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

        secfld(None)
        secfld(False)
        secfld(True)
        secfld(secfld.field(0))
        self.assertRaises(TypeError, secfld, float(0))
        self.assertRaises(TypeError, secfld, SecFld().field(0))

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
        self.assertRaises(TypeError, operator.lt, a, a)  # NB: also tests >
        self.assertRaises(TypeError, operator.le, a, a)  # NB: also tests >=

    def test_SecInt(self):
        SecInt = sectypes.SecInt
        SecInt(p=2**89 - 1)
        self.assertRaises(ValueError, SecInt, p=2**61 - 1)
        secint = SecInt()
        secint16 = SecInt(16)
        secint(None)
        secint(False)
        secint(True)
        secint(secint.field(0))
        self.assertRaises(TypeError, secint, float(0))
        self.assertRaises(TypeError, secint, secint16.field(0))

    def test_SecFxp(self):
        SecFxp = sectypes.SecFxp
        SecFxp(p=2**89 - 1)
        self.assertRaises(ValueError, SecFxp, f=58, p=2**89 - 1)
        secfxp = SecFxp()
        secfxp16 = SecFxp(16)
        secfxp(None)
        secfxp(False)
        secfxp(True)
        secfxp(secfxp.field(0))
        self.assertRaises(TypeError, secfxp, complex(0))
        self.assertRaises(TypeError, secfxp, secfxp16.field(0))

    def test_SecFlt(self):
        SecFlt = sectypes.SecFlt
        secflt = SecFlt(16, e=6)
        secflt(None)
        secflt(False)
        secflt(True)
        self.assertRaises(TypeError, secflt, complex(0))
        self.assertRaises(TypeError, secflt, (2.0, 3))
        self.assertRaises(ValueError, SecFlt, l=16, s=10, e=7)

    def test_operatorerrors(self):
        secfld = sectypes.SecFld()
        secint = sectypes.SecInt()
        a = secfld(0)
        b = secint(1)
        self.assertRaises(TypeError, operator.add, a, b)
        self.assertRaises(TypeError, operator.add, a, 3.14)
        self.assertRaises(TypeError, operator.sub, a, b)
        self.assertRaises(TypeError, operator.mul, a, b)
        self.assertRaises(TypeError, operator.mul, 3.14, b)
        self.assertRaises(TypeError, operator.truediv, a, b)
        self.assertRaises(TypeError, operator.truediv, a, b)
        self.assertRaises(TypeError, operator.mod, a, b)
        self.assertRaises(TypeError, operator.mod, b, a)
        self.assertRaises(TypeError, operator.floordiv, a, b)
        self.assertRaises(TypeError, divmod, a, b)
        self.assertRaises(TypeError, divmod, b, a)
        self.assertRaises(TypeError, operator.pow, b, 3.14)
        self.assertRaises(TypeError, operator.lshift, b, 3.14)
        self.assertRaises(TypeError, operator.lshift, 3.14, b)
        self.assertRaises(TypeError, operator.rshift, b, 3.14)
        self.assertRaises(TypeError, operator.rshift, 3.14, b)


if __name__ == "__main__":
    unittest.main()

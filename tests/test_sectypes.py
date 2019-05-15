import unittest
from mpyc import gfpx
from mpyc import sectypes
from mpyc import runtime

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

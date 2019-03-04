import unittest
from mpyc import gf2x
from mpyc import sectypes
from mpyc.runtime import mpc


class Arithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mpc.logging(False)
        mpc.run(mpc.start())

    @classmethod
    def tearDownClass(cls):
        mpc.run(mpc.shutdown())

    def test_typecaching(self):
        SecFld = sectypes.SecFld
        self.assertEqual(SecFld(), SecFld(char2=False))
        self.assertNotEqual(SecFld(), SecFld(char2=True))
        SecInt = sectypes.SecInt
        self.assertEqual(SecInt(), SecInt(l=32))
        SecFxp = sectypes.SecFxp
        self.assertEqual(SecFxp(), SecFxp(l=32))
        self.assertEqual(SecFxp(), SecFxp(l=32, f=16))

    def test_SecFld(self):
        SecFld = sectypes.SecFld
        secfld = SecFld()
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(char2=True)
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(modulus='x')
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(modulus='x+1')
        self.assertEqual(secfld.field.modulus, 3)
        secfld = SecFld(l=1)
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(l=1, char2=True)
        self.assertEqual(secfld.field.modulus, 2)
        secfld = SecFld(modulus=3, l=1)
        self.assertEqual(secfld.field.modulus, 3)
        self.assertEqual(secfld.field.order, 3)
        secfld = SecFld(modulus=3, char2=True, l=1)
        self.assertEqual(secfld.field.modulus, 3)
        self.assertEqual(secfld.field.order, 2)
        secfld = SecFld(order=3, char2=False, l=1)
        self.assertEqual(secfld.field.modulus, 3)
        self.assertEqual(secfld.field.order, 3)
        secfld = SecFld(order=4, char2=True, l=2)
        self.assertEqual(secfld.field.modulus, 7)
        self.assertEqual(secfld.field.order, 4)
        secfld = SecFld(modulus='1+x^8+x^4+x^3+x')
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)
        secfld = SecFld(order=256)
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)
        secfld = SecFld(order=256, modulus=283)
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)
        secfld = SecFld(modulus=283, char2=True)
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)
        secfld = SecFld(modulus=gf2x.Polynomial(283))
        self.assertEqual(secfld.field.modulus, 283)  # AES polynomial
        self.assertEqual(secfld.field.order, 256)

"""This module supports Galois (finite) fields of characteristic 2.

Function GF creates types implementing binary fields.
Instantiate an object from a field and subsequently apply overloaded
operators such as + (addition), - (subtraction), * (multiplication),
and / (division), etc., to compute with field elements.
In-place versions of the field operators are also provided.
"""

from mpyc import gf2x

def find_irreducible(d):
    """Find smallest irreducible polynomial of degree d satisfying given constraints.

    Constraints ... primitive, low weight w=3, 5
    """
    return gf2x.next_irreducible(2**d - 1)

# Calls to GF with identical modulus return the same class.
_field_cache = {}
def GF(modulus):
    """Create a Galois (finite) field for given irreducible polynomial."""
    poly = gf2x.Polynomial(modulus)

    if poly in _field_cache:
        return _field_cache[poly]

    if not gf2x.is_irreducible(poly):
        raise ValueError(f'{poly} is not irreducible')

    GFElement = type(f'GF(2^{poly.degree()})', (BinaryFieldElement,), {'__slots__':()})
    GFElement.modulus = poly
    GFElement.ext_deg = poly.degree()
    GFElement.order = 2**poly.degree()
    _field_cache[poly] = GFElement
    return GFElement

class BinaryFieldElement():
    """Common base class for binary field elements.

    Invariant: attribute 'value' is reduced.
    """

    __slots__ = 'value'

    modulus = None
    ext_deg = None
    order = None
    frac_length = 0
    lshift_factor = 1
    rshift_factor = 1

    def __init__(self, value):
        if isinstance(value, int):
            assert 0 <= value < self.order
            value = gf2x.Polynomial(value)
        self.value = value % self.modulus

    def __int__(self):
        """Extract polynomial field element as an integer."""
        return self.value.value

    @classmethod
    def to_bytes(cls, x):
        """Return an array of bytes representing the given list of polynomials x."""
        r = (cls.ext_deg + 7) // 8 # -1
        data = bytearray(2 + len(x) * r)
        data[:2] = r.to_bytes(2, byteorder='little')
        j = 2
        for v in x:
            if not isinstance(v, int):
                v = v.value
            data[j:j + r] = v.to_bytes(r, byteorder='little')
            j += r
        return data

    @staticmethod
    def from_bytes(data):
        """Return the list of integers represented by the given array of bytes."""
        r = int.from_bytes(data[:2], byteorder='little')
        n = (len(data) - 2) // r
        x = [None] * n
        j = 2
        for i in range(n):
            x[i] = int.from_bytes(data[j:j + r], byteorder='little')
            j += r
        return x

    def __add__(self, other):
        """Addition."""
        if isinstance(other, type(self)):
            return type(self)(self.value + other.value)

        if isinstance(other, int):
            return type(self)(self.value + other)

        return NotImplemented

    __radd__ = __add__  # TODO: __radd__ may skip first test

    def __iadd__(self, other):
        """In-place addition."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented

        self.value += other
        return self

    __sub__ = __add__
    __rsub__ = __add__
    __isub__ = __iadd__

    def __mul__(self, other):
        """Multiplication."""
        if isinstance(other, type(self)):
            return type(self)(self.value * other.value)

        if isinstance(other, int):
            return type(self)(self.value * other)

        return NotImplemented

    __rmul__ = __mul__

    def __imul__(self, other):
        """In-place multiplication."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented

        self.value *= other
        self.value %= self.modulus.value
        return self

    def __pow__(self, other):
        """Exponentiation."""
        if not isinstance(other, int):
            return NotImplemented

        return type(self)(gf2x.powmod(self.value, other, self.modulus.value))

    def __neg__(self):
        """Negation."""
        return type(self)(self.value)

    def __truediv__(self, other):
        """Division."""
        if isinstance(other, type(self)):
            return self * other._reciprocal()

        if isinstance(other, int):
            return self * type(self)(other)._reciprocal()

        return NotImplemented

    def __rtruediv__(self, other):
        """Division (with reflected arguments)."""
        if isinstance(other, int):
            return type(self)(other) * self._reciprocal()

        return NotImplemented

    def __itruediv__(self, other):
        """In-place division."""
        if isinstance(other, int):
            other = type(self)(other)
        elif not isinstance(other, type(self)):
            return NotImplemented

        self.value *= other._reciprocal().value
        self.value %= self.modulus.value
        return self

    def _reciprocal(self):
        """Multiplicative inverse."""
        return type(self)(gf2x.invert(self.value, self.modulus))

    def __repr__(self):
        return f'{self.value}'

    def __eq__(self, other):
        """Equality test."""
        if isinstance(other, type(self)):
            return self.value == other.value

        if isinstance(other, int):
            return self.value == other

        return NotImplemented

    def __hash__(self):
        """Hash value."""
        return hash((type(self), self.value))

    def __bool__(self):
        """Truth value testing.

        Return False if this field element is zero, True otherwise.
        Field elements can thus be used directly in Boolean formulas.
        """
        return bool(self.value)

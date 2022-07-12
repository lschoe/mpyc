"""This module supports finite (Galois) fields.

Function GF creates types implementing finite fields.
Instantiate an object from a field and subsequently apply overloaded
operators such as +,-,*,/ etc., to compute with field elements.
In-place versions of the field operators are also provided.
Taking square roots and quadratic residuosity tests supported as well.
"""

import functools
from mpyc import gmpy as gmpy2
from mpyc import gfpx


def GF(modulus):
    """Create a finite (Galois) field for given modulus (prime number or irreducible polynomial)."""
    if isinstance(modulus, gfpx.Polynomial):
        return xGF(modulus)

    if isinstance(modulus, tuple):
        p, n, w = modulus
    else:
        p = modulus
        if p == 2:
            n, w = 1, 1
        else:
            n, w = 2, p-1
    return pGF(p, n, w)


class FiniteFieldElement:
    """Abstract base class for finite field elements.

    Invariant: attribute 'value' nonnegative and below modulus.
    """

    __slots__ = 'value'

    modulus = None
    order = None
    characteristic = None
    ext_deg = None
    byte_length = None
    is_signed = None
    _mix_types: type  # or, a tuple of types

    def __init__(self, value):
        self.value = value % self.modulus  # TODO: make this more direct for efficiency

    def __int__(self):
        """Extract field element as an integer value."""
        raise NotImplementedError('abstract method')

    @classmethod
    def to_bytes(cls, x):
        """Return byte string representing the given list of integers x."""
        byte_order = 'little'
        r = cls.byte_length
        return r.to_bytes(2, byte_order) + b''.join(v.to_bytes(r, byte_order) for v in x)

    @staticmethod
    def from_bytes(data):
        """Return the list of integers represented by the given byte string."""
        byte_order = 'little'
        from_bytes = int.from_bytes  # cache
        r = from_bytes(data[:2], byte_order)
        return [from_bytes(data[i:i+r], byte_order) for i in range(2, len(data), r)]

    def __add__(self, other):
        """Addition."""
        if isinstance(other, type(self)):
            return type(self)(self.value + other.value)

        if isinstance(other, self._mix_types):
            return type(self)(self.value + other)

        return NotImplemented

    def __radd__(self, other):
        """Addition (with reflected arguments)."""
        if isinstance(other, self._mix_types):
            return type(self)(self.value + other)

        return NotImplemented

    def __iadd__(self, other):
        """In-place addition."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        self.value += other
        self.value %= self.modulus
        return self

    def __sub__(self, other):
        """Subtraction."""
        if isinstance(other, type(self)):
            return type(self)(self.value - other.value)

        if isinstance(other, self._mix_types):
            return type(self)(self.value - other)

        return NotImplemented

    def __rsub__(self, other):
        """Subtraction (with reflected arguments)."""
        if isinstance(other, self._mix_types):
            return type(self)(other - self.value)

        return NotImplemented

    def __isub__(self, other):
        """In-place subtraction."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        self.value -= other
        self.value %= self.modulus
        return self

    def __neg__(self):
        """Negation."""
        return type(self)(-self.value)

    def __pos__(self):
        """Unary +."""
        return type(self)(+self.value)

    def __mul__(self, other):
        """Multiplication."""
        if isinstance(other, type(self)):
            return type(self)(self.value * other.value)

        if isinstance(other, self._mix_types):
            return type(self)(self.value * other)

        return NotImplemented

    def __rmul__(self, other):
        """Multiplication (with reflected arguments)."""
        if isinstance(other, self._mix_types):
            return type(self)(self.value * other)

        return NotImplemented

    def __imul__(self, other):
        """In-place multiplication."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        self.value *= other
        self.value %= self.modulus
        return self

    def __truediv__(self, other):
        """Division."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        return self * type(self)._reciprocal(other)

    def __rtruediv__(self, other):
        """Division (with reflected arguments)."""
        if isinstance(other, self._mix_types):
            return self.reciprocal() * other

        return NotImplemented

    def __itruediv__(self, other):
        """In-place division."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        self.value *= type(self)._reciprocal(other)
        self.value %= self.modulus
        return self

    def __pow__(self, other):
        """Exponentiation."""
        raise NotImplementedError('abstract method')

    @classmethod
    def _reciprocal(cls, a):
        """Multiplicative inverse."""
        raise NotImplementedError('abstract method')

    def reciprocal(self):
        """Multiplicative inverse."""
        cls = type(self)
        return cls(cls._reciprocal(self.value))

    def __lshift__(self, other):
        """Left shift."""
        if not isinstance(other, int):
            return NotImplemented

        return type(self)(self.value << other)

    def __rlshift__(self, other):
        """Left shift (with reflected arguments)."""
        return NotImplemented

    def __ilshift__(self, other):
        """In-place left shift."""
        if not isinstance(other, int):
            return NotImplemented

        self.value <<= other
        self.value %= self.modulus
        return self

    def __rshift__(self, other):
        """Right shift."""
        raise NotImplementedError('abstract method')

    def __rrshift__(self, other):
        """Right shift (with reflected arguments)."""
        return NotImplemented

    def __irshift__(self, other):
        """In-place right shift."""
        raise NotImplementedError('abstract method')

    @classmethod
    def _is_sqr(cls, a):
        """Test for quadratic residuosity (0 is also square)."""
        raise NotImplementedError('abstract method')

    def is_sqr(self):
        """Test for quadratic residuosity (0 is also square)."""
        return self._is_sqr(self.value)

    @classmethod
    def _sqrt(cls, a, INV=False):
        """Modular (inverse) square roots."""
        raise NotImplementedError('abstract method')

    def sqrt(self, INV=False):
        """Modular (inverse) square roots."""
        cls = type(self)
        return cls(cls._sqrt(self.value, INV=INV))

    def __eq__(self, other):
        """Equality test."""
        if isinstance(other, type(self)):
            return self.value == other.value

        if isinstance(other, self._mix_types):
            return not (self.value - other) % self.modulus

        return NotImplemented

    def __hash__(self):
        """Make finite field elements hashable (e.g., for LRU caching)."""
        return hash((type(self).__name__, self.value))

    def __bool__(self):
        """Truth value testing.

        Return False if this field element is zero, True otherwise.
        Field elements can thus be used directly in Boolean formulas.
        """
        return bool(self.value)


def find_prime_root(l, blum=True, n=1):
    """Find smallest prime of bit length at least l satisfying given constraints.

    Default is to return Blum primes (primes p with p % 4 == 3).
    Also, a primitive root w is returned of prime order at least n (0 < w < p).
    """
    if l <= 2:
        if not blum:
            p = 2
            assert n == 1
            w = 1
        else:
            p = 3
            n, w = 2, p-1
    elif n <= 2:
        p = gmpy2.next_prime(1 << l-1)
        if blum:
            while p%4 != 3:
                p = gmpy2.next_prime(p)
        p = int(p)
        w = p-1 if n == 2 else 1
    else:
        assert blum
        if not gmpy2.is_prime(n):
            n = gmpy2.next_prime(n)
        p = 1 + 2*n * (3 + 2*((1 << l-3) // n))
        while not gmpy2.is_prime(p):
            p += 4*n

        a = 2
        while (w := gmpy2.powmod(a, (p-1) // n, p)) == 1:
            a += 1
        p, w = int(p), int(w)
    return p, n, w


@functools.lru_cache(maxsize=None)
def pGF(p, n, w):
    """Create a finite field for given prime modulus p."""
    if not gmpy2.is_prime(p):
        raise ValueError('modulus is not a prime')

    GFp = type(f'GF({p})', (PrimeFieldElement,), {'__slots__': ()})
    GFp.__doc__ = 'Class of prime field elements.'
    GFp.modulus = p
    GFp.order = p
    GFp.characteristic = p
    GFp.ext_deg = 1
    GFp.byte_length = (GFp.order.bit_length() + 7) >> 3
    GFp.is_signed = True
    GFp.nth = n
    GFp.root = w % p
    return GFp


class PrimeFieldElement(FiniteFieldElement):
    """Common base class for prime field elements."""

    __slots__ = ()

    is_signed = None
    nth = None
    root = None
    _mix_types = int

    @staticmethod
    def createGF(p, n, w):
        """Create new object for use by pickle module."""
        obj = pGF(p, n, w)
        return PrimeFieldElement.__new__(obj)

    def __reduce__(self):
        return (PrimeFieldElement.createGF, (self.modulus, self.nth, self.root),
                (None, {'value': self.value}))

    def __int__(self):
        """Extract field element as a (signed) integer value."""
        if self.is_signed:
            v = self.signed_()
        else:
            v = self.unsigned_()
        return v

    def __abs__(self):
        """Absolute value of (signed) value."""
        return abs(self.__int__())

    def __pow__(self, other):
        """Exponentiation."""
        if not isinstance(other, int):
            return NotImplemented

        return type(self)(int(gmpy2.powmod(self.value, other, self.modulus)))

    @classmethod
    def _reciprocal(cls, a):
        return int(gmpy2.invert(a, cls.modulus))

    @classmethod
    @functools.lru_cache(maxsize=1)  # NB: 1-place cache to speed up rshift for secure trunc() etc.
    def _reciprocal2(cls, n):
        """Return multiplicative inverse of 2**n, n>=0."""
        return cls._reciprocal(1 << n)

    def __rshift__(self, other):
        """Right shift."""
        if not isinstance(other, int):
            return NotImplemented

        cls = type(self)
        return cls(self.value * cls._reciprocal2(other))

    def __irshift__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        self.value *= self._reciprocal2(other)
        self.value %= self.modulus
        return self

    @classmethod
    def _is_sqr(cls, a):
        p = cls.modulus
        if p == 2:
            return True

        return gmpy2.legendre(a, p) != -1

    @classmethod
    def _sqrt(cls, a, INV=False):
        p = cls.modulus
        if a == 0:
            if INV:
                raise ZeroDivisionError('no inverse sqrt of 0')

            return a

        if p == 2:
            return a

        if p&3 == 3:
            if INV:
                p4 = (p*3 - 5) >> 2  # a**p4 == a**(-1/2) == 1/sqrt(a) mod p
            else:
                p4 = (p+1) >> 2
            return int(gmpy2.powmod(a, p4, p))

        # 1 (mod 4) primes are covered using Cipolla-Lehmer's algorithm.
        # find b s.t. b^2 - 4*a is not a square
        b = 1
        while gmpy2.legendre(b * b - 4*a, p) != -1:
            b += 1

        # compute u*X + v = X^{(p+1)/2} mod f, for f = X^2 - b*X + a
        u, v = 0, 1
        e = (p+1) >> 1
        for i in range(e.bit_length() - 1, -1, -1):
            u2 = (u * u) % p
            u = ((u<<1) * v + b * u2) % p
            v = (v * v - a * u2) % p
            if (e >> i) & 1:
                u, v = (v + b * u) % p, (-a * u) % p
        if INV:
            v = cls._reciprocal(v)

        return v

    def signed_(self):
        """Return signed integer representation, symmetric around zero."""
        v = self.value
        if v > self.modulus >> 1:
            v -= self.modulus
        return v

    def unsigned_(self):
        """Return unsigned integer representation."""
        return self.value

    def __repr__(self):
        return f'{self.__int__()}'


def find_irreducible(p, d):
    """Find smallest irreducible polynomial of degree d over GF(p)."""
    # TODO: implement constraints, e.g., low weight, primitive
    return gfpx.GFpX(p).next_irreducible(p**d - 1)


@functools.lru_cache(maxsize=None)
def xGF(modulus):
    """Create a finite field for given irreducible polynomial."""
    p = modulus.p
    poly = gfpx.GFpX(p)
    if not poly.is_irreducible(modulus):
        raise ValueError('modulus is not irreducible')

    d = poly.deg(modulus)
    BaseFieldElement = BinaryFieldElement if p == 2 else ExtensionFieldElement
    GFq = type(f'GF({p}^{d})', (BaseFieldElement,), {'__slots__': ()})
    GFq.__doc__ = f'Class of {"binary" if p == 2 else "extension"} field elements.'
    GFq.modulus = modulus
    GFq.order = p**d
    GFq.characteristic = p
    GFq.ext_deg = d
    GFq.byte_length = (GFq.order.bit_length() + 7) >> 3
    return GFq


class ExtensionFieldElement(FiniteFieldElement):
    """Common base class for extension field elements."""

    __slots__ = ()

    _least_qnr = None
    _mix_types = (int, gfpx.Polynomial)

    def __init__(self, value):
        if isinstance(value, str):
            value = type(self.modulus)(value)  # to prevent % is used for string formatting
        super().__init__(value)

    @staticmethod
    def createGF(modulus):
        """Create new object for use by pickle module."""
        obj = xGF(modulus)
        return ExtensionFieldElement.__new__(obj)

    def __reduce__(self):
        return ExtensionFieldElement.createGF, (self.modulus,), (None, {'value': self.value})

    def __int__(self):
        return int(self.value)

    def __pow__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        poly = type(self.value)
        return type(self)(poly.powmod(self.value, other, self.modulus))

    @classmethod
    def _reciprocal(cls, a):
        return type(cls.modulus).invert(a, cls.modulus)

    def __rshift__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        return self * self._reciprocal(1 << other)

    def __irshift__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        self.value *= self._reciprocal(1 << other)
        self.value %= self.modulus
        return self

    @classmethod
    def _is_sqr(cls, a):
        poly = type(a)
        q = cls.order
        if q%2 == 0:
            return True

        return poly.powmod(a, (q-1) >> 1, cls.modulus) != [poly.p - 1]

    @classmethod
    def _sqrt(cls, a, INV=False):
        poly = type(a)
        q = cls.order
        if a == []:
            if INV:
                raise ZeroDivisionError('no inverse sqrt of 0')

            return a

        if q%2 == 0:
            return poly.powmod(a, q>>1, cls.modulus)

        if q&3 == 3:
            if INV:
                q4 = (q*3 - 5) >> 2  # a**q4 == a**(-1/2) == 1/sqrt(a) in GF(q)
            else:
                q4 = (q+1) >> 2
            return poly.powmod(a, q4, cls.modulus)

        # Tonelli-Shanks
        n = q-1
        s = (n & -n).bit_length() - 1  # number of times 2 divides n
        t = n >> s
        # q - 1 = t 2^s, t odd
        z = cls._least_qnr
        if z is None:
            i = 2
            while True:
                z = poly.powmod(i, t, cls.modulus)
                if poly.powmod(z, 1 << s-1, cls.modulus) != 1:
                    break
                i += 1
            cls._least_qnr = z  # cache least QNR raised to power t

        # TODO: improve following code a bit
        w = poly.powmod(a, t>>1, cls.modulus)
        x = a * w % cls.modulus
        b = x * w % cls.modulus
        v = s
        while b != 1:
            b2 = b
            k = 0
            while b2 != 1:
                b2 = b2 * b2 % cls.modulus
                k += 1
            w = poly.powmod(z, 1 << v - k - 1, cls.modulus)
            z = w * w % cls.modulus
            b = b * z % cls.modulus
            x = x * w % cls.modulus
            v = k
        if INV:
            x = cls._reciprocal(x)

        return x

    def __repr__(self):
        return f'{self.value}'


class BinaryFieldElement(ExtensionFieldElement):
    """Common base class for binary field elements."""

    __slots__ = ()

    characteristic = 2
    _mix_types = (int, gfpx.BinaryPolynomial)

    @classmethod
    def _is_sqr(cls, a):
        return True

    @classmethod
    def _sqrt(cls, a, INV=False):
        poly = type(a)
        q = cls.order
        if a == 0:
            if INV:
                raise ZeroDivisionError('no inverse sqrt of 0')

            return a

        return poly.powmod(a, q>>1, cls.modulus)

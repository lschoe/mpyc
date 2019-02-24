"""This module collects the secure (secret-shared) types for MPyC.

Secure (secret-shared) number types all use a common base class, which
ensures that operators such as +, *, >= are defined by operator overloading.
"""

import functools
import asyncio
from mpyc import gmpy
from mpyc import gf2x
from mpyc import bfield
from mpyc import pfield

runtime = None


class Share:
    """A secret-shared value.

    An MPC protocol operates on secret-shared values, represented by Share
    objects. The basic Python operators are overloaded for Share objects.
    An expression like a * b will create a new Share object, which will
    eventually contain the product of a and b. The product is computed
    asynchronously, using an instance of a specific cryptographic protocol.
    """

    __slots__ = 'df'

    field = None

    def __init__(self, value=None):
        """Initialize a share."""
        if value is not None:
            if isinstance(value, int):
                value = self.field(value)
            self.df = value
        else:
            self.df = asyncio.Future(loop=runtime._loop)

    def __bool__(self):
        """Use of secret-shared values in Boolean expressions makes no sense."""
        raise TypeError('cannot use secure type in Boolean expressions')

    def _coerce(self, other):
        if isinstance(other, Share):
            if not isinstance(other, type(self)):
                return NotImplemented
        elif isinstance(other, int):
            other = type(self)(other)
        elif isinstance(other, float):
            if self.field.frac_length:
                other = type(self)(other)
            else:
                return NotImplemented
        elif not isinstance(other, self.field):
            return NotImplemented
        return other

    def _coerce2(self, other):
        if isinstance(other, Share):
            if not isinstance(other, type(self)):
                return NotImplemented
        elif isinstance(other, int):
            pass
        elif isinstance(other, float):
            if self.field.frac_length:
                if other.is_integer():
                    other = round(other)
            else:
                return NotImplemented
        elif not isinstance(other, self.field):
            return NotImplemented
        return other

    def __neg__(self):
        """Negation."""
        return runtime.neg(self)

    def __add__(self, other):
        """Addition."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return runtime.add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        """Subtraction."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return runtime.sub(self, other)

    def __rsub__(self, other):
        """Subtraction (with reflected arguments)."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return runtime.sub(other, self)

    def __mul__(self, other):
        """Multiplication."""
        other = self._coerce2(other)
        if other is NotImplemented:
            return NotImplemented
        return runtime.mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Division."""
        other = self._coerce2(other)
        if other is NotImplemented:
            return NotImplemented
        return runtime.div(self, other)

    def __rtruediv__(self, other):
        """Division (with reflected arguments)."""
        other = self._coerce2(other)
        if other is NotImplemented:
            return NotImplemented
        return runtime.div(other, self)

    def __mod__(self, other):
        """Integer remainder."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        # TODO: extend beyond powers of 2
        a = other.df.value
        assert not a & (a - 1), 'Powers of 2 only, for now!'
        if a == 2:
            r = runtime.lsb(self)
        else:
            r = runtime.from_bits(runtime.to_bits(self, a.bit_length() - 1))
        return r

    def __rmod__(self, other):
        """Integer remainder (with reflected arguments)."""
        # TODO: extend beyond powers of 2
        a = self.df.value
        assert not a & (a - 1), 'Powers of 2 only, for now!'
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        if a == 2:
            r = runtime.lsb(other)
        else:
            r = runtime.from_bits(runtime.to_bits(other, a.bit_length() - 1))
        return r

    def __floordiv__(self, other):
        """Integer quotient."""
        r = self.__mod__(other)
        if r is NotImplemented:
            return NotImplemented
        other = self._coerce(other)  # avoid coercing twice
        if other is NotImplemented:
            return NotImplemented
        q = (self - r) / other.df
        return q

    def __rfloordiv__(self, other):
        """Integer quotient (with reflected arguments)."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        r = other.__mod__(self)  # avoid coercing twice
        if r is NotImplemented:
            return NotImplemented
        q = (other - r) / self.df
        return q

    def __divmod__(self, other):
        """Integer division."""
        r = self.__mod__(other)
        if r is NotImplemented:
            return NotImplemented
        other = self._coerce(other)  # avoid coercing twice
        if other is NotImplemented:
            return NotImplemented
        q = (self - r) / other.df
        return q, r

    def __rdivmod__(self, other):
        """Integer division (with reflected arguments)."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        r = other.__mod__(self)  # avoid coercing twice
        if r is NotImplemented:
            return NotImplemented
        q = (other - r) / self.df
        return q, r

    def __pow__(self, other):
        """Exponentation for public integral exponent."""
        # TODO: extend to secret exponent
        if not isinstance(other, int):
            return NotImplemented
        return runtime.pow(self, other)

    def __lshift__(self, other):
        """Left shift with public integral offset."""
        # TODO: extend to secret offset
        if not isinstance(other, int):
            return NotImplemented
        return runtime.mul(self, 1 << other)

    def __rlshift__(self, other):
        """Left shift (with reflected arguments)."""
        return NotImplemented

    def __rshift__(self, other):
        """Right shift with public integral offset."""
        # TODO: extend to secret offset
        # TODO: extend beyond offset 1
        if not isinstance(other, int):
            return NotImplemented
        return self.__floordiv__(1 << other)

    def __rrshift__(self, other):
        """Right shift (with reflected arguments)."""
        return NotImplemented

    def __and__(self, other):
        """Bitwise and, for now 1-bit only."""
        return self * other

    __rand__ = __and__

    def __xor__(self, other):
        """Bitwise exclusive-or, for now 1-bit only."""
        return self + other - 2 * self * other

    __rxor__ = __xor__

    def __invert__(self):
        """Bitwise not (inversion), for now 1-bit only."""
        return 1 - self

    def __or__(self, other):
        """Bitwise or, for now 1-bit only."""
        return self + other - self * other

    __ror__ = __or__

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        # self >= other
        c = self - other
        return runtime.sgn(c, GE=True)

    def __gt__(self, other):
        """Strictly greater-than comparison."""
        # self > other <=> not (self <= other)
        c = other - self
        return 1 - runtime.sgn(c, GE=True)

    def __le__(self, other):
        """Less-than or equal comparison."""
        # self <= other <=> other >= self
        c = other - self
        return runtime.sgn(c, GE=True)

    def __lt__(self, other):
        """Strictly less-than comparison."""
        # self < other <=> not (self >= other)
        c = self - other
        return 1 - runtime.sgn(c, GE=True)

    def __eq__(self, other):
        """Equality testing."""
        # self == other
        c = self - other
        return runtime.is_zero(c)

    def __ne__(self, other):
        """Negated equality testing."""
        # self != other <=> not (self == other)
        c = self - other
        return 1 - runtime.is_zero(c)


class SecureFiniteField(Share):
    """Base class for secret-shared finite field values.

    NB: bit-oriented operations will be supported for prime fields.
    """

    __slots__ = ()

    def __mod__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __rmod__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __and__(self, other):
        """Bitwise and for binary fields (otherwise 1-bit only)."""
        if not isinstance(self.field.modulus, int):
            return runtime.and_(self, other)

        return super().__and__(other)

    def __xor__(self, other):
        """Bitwise exclusive-or for binary fields (otherwise 1-bit only)."""
        if not isinstance(self.field.modulus, int):
            return runtime.xor(self, other)

        return super().__xor__(other)

    def __invert__(self):
        """Bitwise not (inversion) for binary fields (otherwise 1-bit only)."""
        if not isinstance(self.field.modulus, int):
            return runtime.invert(self)

        return super().__invert__()

    def __or__(self, other):
        """Bitwise or for binary fields (otherwise 1-bit only)."""
        if not isinstance(self.field.modulus, int):
            return runtime.or_(self, other)

        return super().__or__(other)

    def __ge__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __gt__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __le__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __lt__(self, other):
        """Currently no support at all."""
        return NotImplemented


class SecureInteger(Share):
    """Base class for secret-shared integer values."""

    __slots__ = ()


class SecureFixedPoint(Share):
    """Base class for secret-shared fixed-point values."""

    __slots__ = ()


_sectypes = {}


def SecFld(order=None, modulus=None, char2=None, l=None):
    """Secure prime or binary field of (l+1)-bit order.

    Field is prime by default, and if order (or modulus) is prime.
    Field is binary if order is a power of 2, if modulus is a
    polynomial, or if char2 is True.
    """
    if isinstance(modulus, str):
        modulus = gf2x.Polynomial(modulus)
    if isinstance(modulus, gf2x.Polynomial):
        char2 = char2 or (char2 is None)
        assert char2  # binary field
        modulus = int(modulus)
    if order is not None:
        if order == 2:
            assert modulus is None or modulus == 2 or modulus == 3
            if modulus is None or modulus == 2:
                # default: prime field
                char2 = char2 or False
            else:
                char2 = char2 or (char2 is None)
                assert char2  # binary field
        elif gmpy.is_prime(order):
            modulus = modulus or order
            assert modulus == order
            char2 = char2 or False
            assert not char2  # prime field
        elif order % 2 == 0:
            assert modulus is None or modulus.bit_length() == order.bit_length()
            char2 = char2 or (char2 is None)
            assert char2  # binary field
        else:
            raise ValueError('only prime fields and binary fields supported')
        l = l or order.bit_length() - 1
        assert l == order.bit_length() - 1
    if modulus is None:
        l = l or 1
        if char2:
            modulus = int(bfield.find_irreducible(l))
        else:
            modulus = pfield.find_prime_root(l + 1, blum=False)[0]
    l = modulus.bit_length() - 1
    if char2:
        field = bfield.GF(modulus)
    else:
        field = pfield.GF(modulus)
    assert runtime.threshold == 0 or field.order > len(runtime.parties), \
        'Field order must exceed number of parties, unless threshold is 0.'
    # field.order >= number of parties for MDS
    field.is_signed = False
    return _SecFld(l, field)


@functools.lru_cache(maxsize=None)
def _SecFld(l, field):
    sectype = type(f'SecFld{l}({field})', (SecureFiniteField,), {'__slots__': ()})
    sectype.field = field
    sectype.bit_length = l
    return sectype


def _pfield(l, f, p, n):
    k = runtime.options.security_parameter
    if p is None:
        p = pfield.find_prime_root(l + max(f, k + 1) + 1, n=n)
    else:
        assert p.bit_length() > l + max(f, k + 1), f'Prime {p} too small.'
    return pfield.GF(p, f)


def SecInt(l=None, p=None, n=2):
    """Secure l-bit integers."""
    if l is None:
        l = runtime.options.bit_length
    return _SecInt(l, p, n)


@functools.lru_cache(maxsize=None)
def _SecInt(l, p, n):
    if p is None:
        name = f'SecInt{l}'
    else:
        name = f'SecInt{l}({p})'
    sectype = type(name, (SecureInteger,), {'__slots__': ()})
    sectype.field = _pfield(l, 0, p, n)
    sectype.bit_length = l
    return sectype


def SecFxp(l=None, f=None, p=None, n=2):
    """Secure l-bit fixed-point numbers with f-bit fractional part.

    NB: if dividing secure fixed-point numbers, make sure that l =~ 2f.
    """
    if l is None:
        l = runtime.options.bit_length
    if f is None:
        f = l // 2  # l =~ 2f enables division such that x =~ 1/(1/x)

    return _SecFxp(l, f, p, n)


@functools.lru_cache(maxsize=None)
def _SecFxp(l, f, p, n):
    if p is None:
        name = f'SecFxp{l}:{f}'
    else:
        name = f'SecFxp{l}:{f}({p})'

    def init(self, value=None, integral=False):
        if value is not None:
            if isinstance(value, int):
                self.integral = True
                value <<= f
            elif isinstance(value, float):
                self.integral = value.is_integer()
                value = round(value * (1 << f))
            else:
                self.integral = integral
        else:
            self.integral = integral
        super(sectype, self).__init__(value)
    sectype = type(name, (SecureFixedPoint,),
                   {'__slots__': 'integral', '__init__': init})
    sectype.field = _pfield(l, f, p, n)
    sectype.bit_length = l
    return sectype

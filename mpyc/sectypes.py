"""This module collects the secure (secret-shared) types for MPyC.

Secure (secret-shared) number types all use a common base class, which
ensures that operators such as +, *, >= are defined by operator overloading.
"""

import math
import functools
import asyncio
from mpyc import gmpy as gmpy2
from mpyc import gfpx
from mpyc import finfields

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

    def __init__(self, value=None):
        """Initialize a share."""
        if value is not None:
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
            if isinstance(self, SecureFixedPoint):
                other = type(self)(other)
            else:
                return NotImplemented
        return other

    def _coerce2(self, other):
        if isinstance(other, Share):
            if not isinstance(other, type(self)):
                return NotImplemented
        elif isinstance(other, int):
            pass
        elif isinstance(other, float):
            if isinstance(self, SecureFixedPoint):
                if other.is_integer():
                    other = round(other)
            else:
                return NotImplemented
        return other

    def __neg__(self):
        """Negation."""
        return runtime.neg(self)

    def __pos__(self):
        """Unary +."""
        return runtime.pos(self)

    def __abs__(self):
        """Absolute value."""
        return runtime.abs(self)

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
        """Integer remainder with public divisor."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return runtime.mod(self, other.df.value)

    def __rmod__(self, other):
        """Integer remainder (with reflected arguments)."""
        return NotImplemented

    def __floordiv__(self, other):
        """Integer quotient with public divisor."""
        return self.__divmod__(other)[0]

    def __rfloordiv__(self, other):
        """Integer quotient (with reflected arguments)."""
        return NotImplemented

    def __divmod__(self, other):
        """Integer division with public divisor."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        r = runtime.mod(self, other.df.value)
        q = (self - r) / other.df
        return q * 2**other.df.frac_length, r

    def __rdivmod__(self, other):
        """Integer division (with reflected arguments)."""
        return NotImplemented

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
        return runtime.ge(self, other)

    def __gt__(self, other):
        """Strictly greater-than comparison."""
        # self > other <=> not (self <= other)
        return 1 - runtime.ge(other, self)

    def __le__(self, other):
        """Less-than or equal comparison."""
        # self <= other <=> other >= self
        return runtime.ge(other, self)

    def __lt__(self, other):
        """Strictly less-than comparison."""
        # self < other <=> not (self >= other)
        return 1 - runtime.ge(self, other)

    def __eq__(self, other):
        """Equality testing."""
        # self == other
        return runtime.eq(self, other)

    def __ne__(self, other):
        """Negated equality testing."""
        # self != other <=> not (self == other)
        return 1 - runtime.eq(self, other)


class SecureFiniteField(Share):
    """Base class for secret-shared finite field values.

    NB: bit-oriented operations will be supported for prime fields.
    """

    __slots__ = ()
    field = None

    def __abs__(self):
        """Currently no support at all."""
        raise TypeError(f"bad operand type for abs(): '{type(self).__name__}'")

    def __mod__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __rmod__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __floordiv__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __rfloordiv__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __divmod__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __rdivmod__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __lshift__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __rlshift__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __rshift__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __rrshift__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __and__(self, other):
        """Bitwise and for binary fields (otherwise 1-bit only)."""
        if self.field.characteristic == 2:
            return runtime.and_(self, other)

        return super().__and__(other)

    def __xor__(self, other):
        """Bitwise exclusive-or for binary fields (otherwise 1-bit only)."""
        if self.field.characteristic == 2:
            return runtime.xor(self, other)

        return super().__xor__(other)

    def __invert__(self):
        """Bitwise not (inversion) for binary fields (otherwise 1-bit only)."""
        if self.field.characteristic == 2:
            return runtime.invert(self)

        return super().__invert__()

    def __or__(self, other):
        """Bitwise or for binary fields (otherwise 1-bit only)."""
        if self.field.characteristic == 2:
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


def SecFld(order=None, modulus=None, char=None, ext_deg=None, min_order=None, signed=False):
    """Secure finite field of order q = p**d.

    Order q >= min_order.
    Field is prime (d = 1) by default and if modulus is prime.
    Extension degree d > 1 if order is a prime power p**d with d > 1,
    if modulus is a polynomial or a string or an integer > char,
    or if ext_deg is an integer > 1, or if min_order > char.
    """
    # TODO: raise errors instead of assert statements
    if order is not None:
        p, d = gmpy2.factor_prime_power(order)
        char = char or p
        assert char == p
        ext_deg = ext_deg or d
        assert ext_deg == d
    # order now represented by (char, ext_deg)

    if isinstance(modulus, str):
        char = char or 2
        modulus = gfpx.GFpX(char)(modulus)
    if isinstance(modulus, int):
        if char and modulus > char:
            modulus = gfpx.GFpX(char)(modulus)
    if isinstance(modulus, gfpx.Polynomial):
        char = char or modulus.p
        assert char == modulus.p
        ext_deg = ext_deg or modulus.degree()
    elif isinstance(modulus, int):
        char = char or modulus
        assert char == modulus
        ext_deg = ext_deg or 1
        assert ext_deg == 1
    else:
        assert modulus is None
        if min_order is None:
            char = char or 2
            ext_deg = ext_deg or 1
            min_order = char**ext_deg
        else:
            if char is None:
                ext_deg = ext_deg or 1
                root, exact = gmpy2.iroot(min_order, ext_deg)
                min_char = root + (not exact)   # ceiling of min_order^(1/ext_deg)
                char = int(gmpy2.next_prime(min_char - 1))
            else:
                if ext_deg is None:
                    ext_deg = math.ceil(math.log(min_order, char))

        if ext_deg == 1:
            modulus = char
        else:
            modulus = finfields.find_irreducible(char, ext_deg)

    order = order or char**ext_deg
    min_order = min_order or order
    assert min_order <= order
    field = finfields.GF(modulus)
    assert runtime.threshold == 0 or field.order > len(runtime.parties), \
        'Field order must exceed number of parties, unless threshold is 0.'
    # TODO: field.order >= number of parties for MDS
    field.is_signed = signed
    return _SecFld(field)


@functools.lru_cache(maxsize=None)
def _SecFld(field):
    l = field.order.bit_length() - 1

    def init(self, value=None):
        if value is not None:
            if isinstance(value, int):
                value = sectype.field(value)
            elif not isinstance(value, sectype.field):
                if isinstance(value, finfields.FiniteFieldElement):
                    raise TypeError(f'incompatible finite field {type(value).__name__}'
                                    f' for {type(self).__name__}')

                raise TypeError('None, int, or finite field required')
        super(sectype, self).__init__(value)
    sectype = type(f'SecFld{l}({field.__name__})', (SecureFiniteField,),
                   {'__slots__': (), '__init__': init})
    sectype.field = field
    sectype.bit_length = l
    return sectype


def _pfield(l, f, p, n):
    k = runtime.options.sec_param
    if p is None:
        p = finfields.find_prime_root(l + max(f, k+1) + 1, n=n)
    elif p.bit_length() <= l + max(f, k+1):
        raise ValueError(f'Prime {p} too small.')

    return finfields.GF(p, f)


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

    def init(self, value=None):
        """Value must be None, int, or correct field type."""
        if value is not None:
            if isinstance(value, int):
                value = sectype.field(value)
            elif not isinstance(value, sectype.field):
                if isinstance(value, finfields.FiniteFieldElement):
                    raise TypeError(f'incompatible finite field {type(value).__name__}'
                                    f' for {type(self).__name__}')

                raise TypeError('None, int, or finite field required')

        super(sectype, self).__init__(value)
    sectype = type(name, (SecureInteger,), {'__slots__': (), '__init__': init})
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
        f = l//2  # l =~ 2f enables division such that x =~ 1/(1/x)

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
                value = sectype.field(value << f)
            elif isinstance(value, float):
                self.integral = value.is_integer()
                value = sectype.field(round(value * (1<<f)))
            else:  # TODO: check field type
                self.integral = integral
        else:
            self.integral = integral
        super(sectype, self).__init__(value)
    sectype = type(name, (SecureFixedPoint,),
                   {'__slots__': 'integral', '__init__': init})
    sectype.field = _pfield(l, f, p, n)
    sectype.bit_length = l
    return sectype

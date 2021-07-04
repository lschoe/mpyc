"""This module collects the secure (secret-shared) types for MPyC.

Secure (secret-shared) number types all use common base classes, which
ensures that operators such as +,*,>= are defined by operator overloading.
"""

import math
import functools
from asyncio import Future
from mpyc import gmpy as gmpy2
from mpyc import gfpx
from mpyc import finfields

runtime = None


class SecureObject:
    """A secret-shared object.

    An MPC protocol operates on secret-shared objects of type SecureObject.
    The basic Python operators are overloaded by SecureObject classes.
    An expression like a * b will create a new SecureObject, which will
    eventually contain the product of a and b. The product is computed
    asynchronously, using an instance of a specific cryptographic protocol.
    """

    __slots__ = 'share'

    def __init__(self, value=None):
        """Initialize a share.

        If value is None (default), the SecureObject starts out as an empty
        placeholder (implemented as a Future).
        """
        if value is None:
            value = Future(loop=runtime._loop)
        self.share = value

    def set_share(self, v):
        if isinstance(v, Future):
            if v.done():
                self.share.set_result(v.result())
            else:
                v.add_done_callback(lambda x: self.share.set_result(x.result()))
        else:
            self.share.set_result(v)

    def __deepcopy__(self, memo):
        """Let SecureObjects behave as immutable objects.

        Introduced for github.com/meilof/oblif.
        """
        return self

    def __bool__(self):
        """Use of secret-shared objects in Boolean expressions makes no sense."""
        raise TypeError('cannot use secure type in Boolean expressions')

    def if_else(self, x, y):
        """Use SecureObject as condition for secure selection between x and y."""
        return runtime.if_else(self, x, y)

    def if_swap(self, x, y):
        """Use SecureObject as condition for secure swap of x and y."""
        return runtime.if_swap(self, x, y)


class SecureNumber(SecureObject):
    """Base class for secure (secret-shared) numbers."""

    __slots__ = ()

    bit_length = None

    def _coerce(self, other):
        if isinstance(other, SecureObject):
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
        if isinstance(other, SecureObject):
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

        return runtime.mod(self, other)

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

        r = runtime.mod(self, other)
        q = (self - r) * runtime.reciprocal(other)
        return q * 2**self.frac_length, r

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

        return runtime.mul(self, 1<<other)

    def __rlshift__(self, other):
        """Left shift (with reflected arguments)."""
        return NotImplemented

    def __rshift__(self, other):
        """Right shift with public integral offset."""
        # TODO: extend to secret offset
        if not isinstance(other, int):
            return NotImplemented

        return self.__floordiv__(1<<other)

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

    def __lt__(self, other):
        """Strictly less-than comparison."""
        # self < other
        return runtime.lt(self, other)

    def __le__(self, other):
        """Less-than or equal comparison."""
        # self <= other <=> not (other < self)
        return 1 - runtime.lt(other, self)

    def __eq__(self, other):
        """Equality testing."""
        # self == other
        return runtime.eq(self, other)

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        # self >= other <=> not (self < other)
        return 1 - runtime.lt(self, other)

    def __gt__(self, other):
        """Strictly greater-than comparison."""
        # self > other <=> other < self
        return runtime.lt(other, self)

    def __ne__(self, other):
        """Negated equality testing."""
        # self != other <=> not (self == other)
        return 1 - runtime.eq(self, other)


class SecureFiniteField(SecureNumber):
    """Base class for secure (secret-shared) finite field elements.

    NB: bit-oriented operations will be supported for prime fields.
    """

    __slots__ = ()

    frac_length = 0
    field = None

    _output_conversion = None

    def __init__(self, value=None):
        """Initialize a secure finite field element.

        Value must be None, int, or correct field type.
        """
        if value is not None:
            if isinstance(value, int):
                value = self.field(value)
            elif isinstance(value, self.field):
                pass
#            elif isinstance(value, Future):
#                pass  # NB: for internal use in runtime only
            else:
                if isinstance(value, finfields.FiniteFieldElement):
                    raise TypeError(f'incompatible finite field {type(value).__name__} '
                                    f'for {type(self).__name__}')

                raise TypeError('None, int, or finite field element required')

        super().__init__(value)

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

    def __lt__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __le__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __ge__(self, other):
        """Currently no support at all."""
        return NotImplemented

    def __gt__(self, other):
        """Currently no support at all."""
        return NotImplemented


class SecureInteger(SecureNumber):
    """Base class for secure (secret-shared) integers."""

    __slots__ = ()

    frac_length = 0
    field = None

    _output_conversion = int

    def __init__(self, value=None):
        """Initialize a secure integer.

        Value must be None, int, or correct field type.
        """
        if value is not None:
            if isinstance(value, int):
                value = self.field(value)
            elif isinstance(value, self.field):
                pass
            elif isinstance(value, Future):
                pass  # NB: for internal use in runtime only
            else:
                if isinstance(value, finfields.FiniteFieldElement):
                    raise TypeError(f'incompatible finite field {type(value).__name__} '
                                    f'for {type(self).__name__}')

                raise TypeError('None, int, or finite field element required')

        super().__init__(value)


class SecureFixedPoint(SecureNumber):
    """Base class for secure (secret-shared) fixed-point numbers."""

    __slots__ = 'integral'

    frac_length = 0
    field = None

    @classmethod
    def _output_conversion(cls, a):
        return int(a) / 2**cls.frac_length

    def __init__(self, value=None, integral=None):
        """Initialize a secure fixed-point number.

        Value must be None, int, float, or correct field type.

        Boolean flag integral sets the attribute integral of the secure fixed-point number.
        If integral=True or integral=False, the attribute integral is set accordingly.
        Otherwise, if integral=None (not set), the attribute integral is inferred from value
        (set to True if value is a whole number of type int or float).
        """
        if value is not None:
            if isinstance(value, int):
                if integral is None:
                    integral = True
                value = self.field(value << self.frac_length)
            elif isinstance(value, float):
                if integral is None:
                    integral = value.is_integer()
                value = self.field(round(value * (1<<self.frac_length)))
            elif isinstance(value, self.field):
                pass
            elif isinstance(value, Future):
                pass  # NB: for internal use in runtime only
            else:
                if isinstance(value, finfields.FiniteFieldElement):
                    raise TypeError(f'incompatible finite field {type(value).__name__} '
                                    f'for {type(self).__name__}')

                raise TypeError('None, int, float, or finite field element required')

        self.integral = integral
        super().__init__(value)


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
    name = f'SecFld{l}({field.__name__})'
    sectype = type(name, (SecureFiniteField,), {'__slots__': ()})
    sectype.__doc__ = 'Class of secret-shared finite field elements.'
    sectype.field = field
    sectype.bit_length = l
    globals()[name] = sectype  # TODO: check name dynamic SecureFiniteField type sufficiently unique
    return sectype


def _pfield(l, f, p, n):
    k = runtime.options.sec_param
    if p is None:
        p = finfields.find_prime_root(l + f + k + 2, n=n)
    elif p.bit_length() <= l + f + k + 1:
        raise ValueError(f'Prime {p} too small.')

    return finfields.GF(p, f)


def SecInt(l=None, p=None, n=2):
    """Secure l-bit integers."""
    if l is None:
        l = runtime.options.bit_length
    return _SecInt(l, p, n)


@functools.lru_cache(maxsize=None)
def _SecInt(l, p, n):
    name = f'SecInt{l}' if p is None else f'SecInt{l}({p})'
    sectype = type(name, (SecureInteger,), {'__slots__': ()})
    sectype.__doc__ = 'Class of secret-shared integers.'
    sectype.field = _pfield(l, 0, p, n)
    sectype.bit_length = l
    globals()[name] = sectype  # NB: exploit (almost) unique name dynamic SecureInteger type
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
    name = f'SecFxp{l}:{f}' if p is None else f'SecFxp{l}:{f}({p})'
    sectype = type(name, (SecureFixedPoint,), {'__slots__': ()})
    sectype.__doc__ = 'Class of secret-shared fixed-point numbers.'
    sectype.field = _pfield(l, f, p, n)
    sectype.bit_length = l
    sectype.frac_length = f
    globals()[name] = sectype  # NB: exploit (almost) unique name dynamic SecureFixedPoint type
    return sectype


class SecureFloat(SecureNumber):
    """Base class for secure (secret-shared) floating-point numbers.

    Basic arithmetic +,-,*,/ and comparisons <,<=,--,>,>=,!= are supported for secure floats,
    as well as input()/output() and sorting operations like min()/argmax()/sorted().
    Other operations like sum()/prod()/all()/any()/in_prod() are currently not supported for
    secure floats.

    Implementation is kept simple, representing a secure float as a pair consisting of
    a secure fixed-point number for the significand and a secure integer for the exponent.
    Note, however, that even basic arithmetic +,-,*,/ with secure floats is very
    demanding performance-wise (due to dependence on secure bitwise operations).
    """

    __slots__ = ()

    significand_type = None
    exponent_type = None

    def __init__(self, value=None):
        """Initialize a secure floating-point number.

        Value must be None, int, or float.
        """
        if value is None:
            value = (self.significand_type(None), self.exponent_type(None))
        else:
            if isinstance(value, (int, float)):
                e = math.ceil(math.log(abs(value), 2)) if value else 0
                s = value / 2**e
                assert s == 0 or 0.5 <= abs(s) <= 1, (value, s, e)
                value = (self.significand_type(s, integral=False), self.exponent_type(e))
            elif isinstance(value, tuple):
                if len(value) != 2 or \
                   not isinstance(value[0], self.significand_type) or \
                   not isinstance(value[1], self.exponent_type):
                    raise TypeError('Significand/exponent pair required')

            else:
                raise TypeError('None, int, float, or significand/exponent pair required')

        super().__init__(value)

    def set_share(self, v):
        self.share[0].set_share(v[0].share)
        self.share[1].set_share(v[1].share)

    def __neg__(self):
        """Negation."""
        s, e = self.share
        return type(self)((-s, e))

    def __pos__(self):
        """Unary +."""
        return self

    def __abs__(self):
        """Absolute value."""
        s, e = self.share
        return type(self)((abs(s), e))

    def __add__(self, other):
        """Addition."""
        secflt = type(self)
        if isinstance(other, (int, float)):
            other = secflt(other)
        s1, e1 = self.share
        s2, e2 = other.share
        secfxp = type(s1)
        secint = type(e1)
        l = secfxp.bit_length
        f = secfxp.frac_length

        c_e = e1 < e2
        c_s = runtime.convert(c_e, secfxp)
#        e1, e2 = runtime.if_else(c_e, [e2, e1], [e1, e2])
        e1, e2 = runtime.if_swap(c_e, e1, e2)
#        s1, s2 = runtime.if_else(c_s, [s2, s1], [s1, s2])
        s1, s2 = runtime.if_swap(c_s, s1, s2)
        # e1 >= e2
        d = runtime.min(e1 - e2, f)
        d = runtime.convert(d, secfxp)  # NB: 0 <= d <= f fits in headroom secfxp
        d_u = runtime.unit_vector(d, f+1)
        d2 = runtime.in_prod(d_u, [secfxp(2**-i) for i in range(f+1)])  # TODO: avoid reshare
        s = s1 + s2 * d2
        # Normalize s, see also runtime._norm():
        x = runtime.to_bits(s)
        b = x[-1]  # sign bit
        del x[-1]
        x.reverse()
        N, n = runtime.find(x, 1-b, cs_f=lambda b, i: ((b+1) << i, b + i))
        N, n = N * (2**(f - (l-1))), n + (f - (l-1))  # NB: f <= l
        n = runtime.convert(n, secint)
        return secflt((s * N, e1 - n))

    __radd__ = __add__

    def __sub__(self, other):
        """Subtraction."""
        return self + (-other)

    def __rsub__(self, other):
        """Subtraction (with reflected arguments)."""
        return other + (-self)

    def __mul__(self, other):
        """Multiplication."""
        secflt = type(self)
        if isinstance(other, (int, float)):
            other = secflt(other)
        s1, e1 = self.share
        s2, e2 = other.share
        s = s1 * s2  # 1/4 <= abs(s) <= 1
        e = e1 + e2
        x = runtime.to_bits(s)
        # -1   = 1_1.0_00000000
        # -3/4 = 1_1.0_10000000
        # -1/2 = 1_1.1_00000000   <-- x[-2] == x[-3]
        # -1/4 = 1_1.1_10000000   <-- x[-2] == x[-3]
        #  1/4 = 0_0.0_10000000   <-- x[-2] == x[-3]
        #  1/2 = 0_0.1_00000000
        #  3/4 = 0_0.1_10000000
        #  1   = 0_1.0_00000000
        c_s = (x[-2] - x[-3])**2  # x[-2] ^ x[-3]
        c_e = runtime.convert(c_s, type(e))
        s = runtime.if_else(c_s, s, s*2)
        e = runtime.if_else(c_e, e, e-1)
        return secflt((s, e))

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Division."""
        return self * (1/other)

    def __rtruediv__(self, other):
        """Division (with reflected arguments)."""
        return other * self.reciprocal()

    def reciprocal(self):
        """Secure reciprocal (multiplicative inverse)."""
        s, e = self.share
        s = 0.5*(1/s)  # TODO: no normalization for 1/s as 1/2<=abs(s)<=1 (s<0 test still needed)
        return type(self)((s, 1-e))

    def __lt__(self, other):
        """Strictly less-than comparison."""
        # self < other
        s = (self - other).share[0]
        return type(self)((s < 0, self.exponent_type(0)))

    def __le__(self, other):
        """Less-than or equal comparison."""
        # self <= other
        s = (self - other).share[0]
        return type(self)((s <= 0, self.exponent_type(0)))

    def __eq__(self, other):
        """Equality testing."""
        # self == other
        s = (self - other).share[0]
        return type(self)((s == 0, self.exponent_type(0)))

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        # self >= other
        s = (self - other).share[0]
        return type(self)((s >= 0, self.exponent_type(0)))

    def __gt__(self, other):
        """Strictly greater-than comparison."""
        # self > other
        s = (self - other).share[0]
        return type(self)((s > 0, self.exponent_type(0)))

    def __ne__(self, other):
        """Negated equality testing."""
        # self != other
        s = (self - other).share[0]
        return type(self)((s != 0, self.exponent_type(0)))

    @staticmethod
    def is_zero_public(a):
        """Called by runtime.is_zero_public()."""
        return runtime.is_zero_public(a.share[0])

    @classmethod
    def _input(cls, x, senders):
        """Called by runtime.input()."""
        x_s = [a.share[0] for a in x]
        x_e = [a.share[1] for a in x]
        shares_s = runtime.input(x_s, senders)
        shares_e = runtime.input(x_e, senders)
        return [[cls(a) for a in zip(x_s, x_e)] for x_s, x_e in zip(shares_s, shares_e)]

    @classmethod
    async def _output(cls, x, receivers, threshold):
        """Called by runtime.output()."""
        x_s = [a.share[0] for a in x]
        x_s = await runtime.output(x_s, receivers, threshold)
        e_0 = cls.exponent_type(0)
        x_e = [x[i].share[1] if x_s[i] else e_0 for i in range(len(x))]
        x_e = await runtime.output(x_e, receivers, threshold)
        # TODO: consider normalization to eliminate case abs(s)=1 (or case abs(s)=0.5)
        assert all(s == 0 or 0.5 <= abs(s) <= 1 for s in x_s), (x_s, x_e)
        assert all(s != 0 or e == 0 for s, e in zip(x_s, x_e)), (x_s, x_e)
        return [s * 2**e for s, e in zip(x_s, x_e)]


def SecFlt(l=None, s=None, e=None):
    """Secure l-bit floating-point number with s-bit significand and e-bit exponent, where l=s+e.

    The significand is an (s+1)-bit secure (signed) fixed-point number. The absolute value
    of a nonzero significand is normalized between 0.5 and 1.0. Here, both 0.5 and 1.0 are
    included and therefore one extra bit is used.
    The exponent is an e-bit secure (signed) integer.
    """
    if l is None:
        if s is None or e is None:
            l = runtime.options.bit_length
        else:
            l = s + e
    if s is None:
        if e is None:
            s = round(6.2 + 0.99*l - 4.1*math.log(l))  # yields IEEE 754 precisions
            # 1-bit float -> s-bit significand: 16->11, 32->24, 64->53, 128->113, 256->237
        else:
            s = l - e
    if e is None:
        e = l - s
    if not l == s + e:
        raise ValueError(f'Inconsistent bit lengths: l={l} not equal to s+e={s}+{e}={s+e}.')

    return _SecFlt(s, e)


@functools.lru_cache(maxsize=None)
def _SecFlt(s, e):
    name = f'SecFlt{s + e}:{s}:{e}'
    sectype = type(name, (SecureFloat,), {'__slots__': ()})
    sectype.__doc__ = 'Class of secret-shared floating-point numbers.'
    sectype.bit_length = s + e
    sectype.significand_type = SecFxp(s+1, s-1)  # NB: 1 sign bit, 1 extra bit, s-1 fractional bits
    sectype.exponent_type = SecInt(e)
    globals()[name] = sectype  # NB: exploit (almost) unique name dynamic SecureFloat type
    return sectype

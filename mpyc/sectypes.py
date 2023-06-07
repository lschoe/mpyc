"""This module collects basic secure (secret-shared) types for MPyC.

Secure number (array) types all use common base classes, which
ensures that operators such as +,*,>= are defined by operator overloading.
"""

import math
import functools
from asyncio import Future
from mpyc.numpy import np
from mpyc import gmpy as gmpy2
from mpyc import gfpx
from mpyc import finfields
from mpyc import fingroups

runtime = None


if np:
    import operator
    from numpy.core import umath as um


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
        """Initialize share.

        If value is None (default), the SecureObject starts out as an empty
        placeholder (implemented as a Future).
        """
        if value is None:
            value = Future(loop=runtime._loop)
        self.share = value

    def set_share(self, value):
        """Set share to the given value.

        The share is set directly (or recursively, for a composite SecureObject),
        using callbacks if value contains Futures that are not yet done.
        """
        if isinstance(value, Future):
            if value.done():
                self.share.set_result(value.result())
            else:
                value.add_done_callback(lambda x: self.share.set_result(x.result()))
        else:
            self.share.set_result(value)

    def __deepcopy__(self, memo):
        """Let SecureObjects behave as immutable objects.

        Introduced for github.com/meilof/oblif.
        """
        return self

    def __bool__(self):
        """Use of secret-shared objects in Boolean expressions makes no sense."""
        raise TypeError('cannot use secure type in Boolean expressions')

    if np:
        binary_ops = {um.less: operator.lt, um.less_equal: operator.le,
                      um.equal: operator.eq, um.not_equal: operator.ne,
                      um.greater: operator.gt, um.greater_equal: operator.ge,
                      um.add: operator.add, um.subtract: operator.sub,
                      um.multiply: operator.mul, um.divide: operator.truediv,
                      um.floor_divide: operator.floordiv, um.remainder: operator.mod,
                      um.divmod: divmod, um.power: operator.pow,
                      um.left_shift: operator.lshift, um.right_shift: operator.rshift}
        # um.bitwise_and: operator.and_, um.bitwise_xor, operator.xor, um.bitwise_or, operator.or_}
        unary_ops = {um.negative: operator.neg, um.positive: operator.pos,
                     um.absolute: operator.abs}
        # um.invert, operator.invert}

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            """Delegate __array_ufunc__ call to corresponding operator call.

            Provisional support for calls like np.less(secint(9), 10).
            """
            # TODO: handle method and kwargs
            # TODO: handle np.arrays in inputs
            inputs = list(inputs)
            for i in range(len(inputs)):  # NB: also supports things like secint(7) + np.int32(4)
                if isinstance(inputs[i], np.integer):
                    inputs[i] = int(inputs[i])
                elif isinstance(inputs[i], np.floating):
                    inputs[i] = float(inputs[i])
            if op := SecureObject.binary_ops.get(ufunc):
                if isinstance(inputs[0], SecureObject):
                    return op(inputs[0], inputs[1])

                if op == operator.sub:
                    return inputs[1].__rsub__(inputs[0])

                return op(inputs[1], inputs[0])

            if op := SecureObject.unary_ops.get(ufunc):
                return op(inputs[0])

            try:
                func = eval(f'runtime.np_{ufunc.__name__}')
            except AttributeError:
                raise TypeError(f'np.{ufunc.__name__} not supported for {type(self).__name__}')

            return func(*inputs, **kwargs)

        def __array_function__(self, func, types, args, kwargs):
            """Redirect __array_function__ call to array class, if any.

            To support calls like np.block([[secint(9), -1], [1, secint(7)]]).
            """
            return self.array.__array_function__(self, func, types, args, kwargs)


class SecureNumber(SecureObject):
    """Base class for secure (secret-shared) numbers."""

    __slots__ = ()

    bit_length = None

    def if_else(self, x, y):
        """Use SecureNumber as condition for secure selection between x and y."""
        return runtime.if_else(self, x, y)

    def if_swap(self, x, y):
        """Use SecureNumber as condition for secure swap of x and y."""
        return runtime.if_swap(self, x, y)

    def _coerce(self, other):
        if isinstance(other, SecureObject):
            if not isinstance(other, type(self)):
                return NotImplemented

        elif isinstance(other, int):
            other = type(self)(other)
        elif isinstance(other, type(self).field):
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
        elif isinstance(other, type(self).field):
            # TODO: reconsider need for this case (see seclists._norm())
            pass
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
        """Multiplication.

        Special case: repeat of additive group operation.
        """
        if isinstance(other, fingroups.FiniteGroupElement):
            group = type(other)
            if group.is_additive:
                return runtime.SecGrp(type(other)).repeat(other, self)

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
        """Exponentiation for public integral exponent."""
        # TODO: extend to secret exponent
        if not isinstance(other, int):
            return NotImplemented

        return runtime.pow(self, other)

    def __rpow__(self, other):
        """Exponentiation (with reflected arguments) for secret exponent.

        Special case: repeat of multiplicative group operation.
        """
        if isinstance(other, fingroups.FiniteGroupElement):
            group = type(other)
            if group.is_multiplicative:
                return runtime.SecGrp(type(other)).repeat(other, self)

        return NotImplemented

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

    def __rxor__(self, other):
        """Bitwise exclusive-or, for now 1-bit only.

        Special case: repeat of finite group operation.
        """
        if isinstance(other, fingroups.FiniteGroupElement):
            return runtime.SecGrp(type(other)).repeat(other, self)

        return self + other - 2 * self * other

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
    field: type
    subfield: type

    _output_conversion = None

    def __init__(self, value=None):
        """Initialize a secure finite field element.

        Value must be None, int, or correct field type.
        """
        if value is not None:
            if isinstance(value, int):
                if self.subfield is not None:
                    value %= self.subfield.modulus
                value = self.field(value)
            elif isinstance(value, self.field):
                pass
            elif self.subfield is not None and isinstance(value, self.subfield):
                value = self.field(value.value)
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
    field: type

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
    field: type

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

    def _coerce(self, other):
        if isinstance(other, float):
            return type(self)(other)

        return super()._coerce(other)

    def _coerce2(self, other):
        if isinstance(other, float):
            return other

        return super()._coerce2(other)


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
    field.is_signed = signed
    return _SecFld(field)


@functools.cache
def _SecFld(field):
    l = (field.order - 1).bit_length()
    name = f'SecFld{l}({field.__name__})'
    secfld = type(name, (SecureFiniteField,), {'__slots__': ()})
    secfld.__doc__ = 'Class of secret-shared finite field elements.'
    t = runtime.threshold
    m = len(runtime.parties)
    q = field.order
    if t == 0 or m < q:  # TODO: cover case m=q using MDS codes
        secfld.subfield = None
        secfld.field = field
    else:
        secfld.subfield = field
        assert field.ext_deg == 1  # TODO: cover case ext_deg > 1
        e = math.ceil(math.log(m+1, q))  # ensure q**e > m with e>=2
        modulus = finfields.find_irreducible(field.characteristic, e)
        secfld.field = finfields.GF(modulus)

        @classmethod
        def out_conv(cls, a):  # field -> subfield
            assert a.value.degree() <= 0
            return cls.subfield(int(a))

        secfld._output_conversion = out_conv
    secfld.bit_length = l
    globals()[name] = secfld  # TODO: check name dynamic SecureFiniteField type sufficiently unique

    name = f'Array{secfld.__name__}'
    secarray = type(name, (SecureFiniteFieldArray,), {'__slots__': ()})
    secarray.sectype = secfld
    globals()[name] = secarray  # TODO: check name dynamic type sufficiently unique
    secfld.array = secarray
    return secfld


def _pfield(l, f, p, n):
    k = runtime.options.sec_param
    if p is None:
        p = finfields.find_prime_root(l + f + k + 2, n=n)
    elif p.bit_length() <= l + f + k + 1:
        raise ValueError(f'Prime {p} too small.')

    field = finfields.GF(p)
    assert runtime.threshold == 0 or len(runtime.parties) < field.order  # for Shamir secret sharing
    return field


def SecInt(l=None, p=None, n=2):
    """Secure l-bit integers."""
    if l is None:
        l = runtime.options.bit_length
    return _SecInt(l, p, n)


@functools.cache
def _SecInt(l, p, n):
    name = f'SecInt{l}' if p is None else f'SecInt{l}({p})'
    secint = type(name, (SecureInteger,), {'__slots__': ()})
    secint.__doc__ = 'Class of secret-shared integers.'
    secint.field = _pfield(l, 0, p, n)
    secint.bit_length = l
    globals()[name] = secint  # NB: exploit (almost) unique name dynamic SecureInteger type

    name = f'Array{secint.__name__}'
    secarray = type(name, (SecureIntegerArray,), {'__slots__': ()})
    secarray.sectype = secint
    globals()[name] = secarray  # TODO: check name dynamic type sufficiently unique
    secint.array = secarray
    return secint


def SecFxp(l=None, f=None, p=None, n=2):
    """Secure l-bit fixed-point numbers with f-bit fractional part.

    NB: if dividing secure fixed-point numbers, make sure that l =~ 2f.
    """
    if l is None:
        l = runtime.options.bit_length
    if f is None:
        f = l//2  # l =~ 2f enables division such that x =~ 1/(1/x)
    return _SecFxp(l, f, p, n)


@functools.cache
def _SecFxp(l, f, p, n):
    name = f'SecFxp{l}:{f}' if p is None else f'SecFxp{l}:{f}({p})'
    secfxp = type(name, (SecureFixedPoint,), {'__slots__': ()})
    secfxp.__doc__ = 'Class of secret-shared fixed-point numbers.'
    secfxp.field = _pfield(l, f, p, n)
    secfxp.bit_length = l
    secfxp.frac_length = f
    globals()[name] = secfxp  # NB: exploit (almost) unique name dynamic SecureFixedPoint type

    name = f'Array{secfxp.__name__}'
    secarray = type(name, (SecureFixedPointArray,), {'__slots__': ()})
    secarray.frac_length = f  # TODO: consider use of secfxp.frac_length instead
    secarray.sectype = secfxp
    globals()[name] = secarray  # TODO: check name dynamic type sufficiently unique
    secfxp.array = secarray
    return secfxp


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

    significand_type: type
    exponent_type: type

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

    def set_share(self, value):
        self.share[0].set_share(value[0].share)
        self.share[1].set_share(value[1].share)

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
        e1, e2 = runtime.if_swap(c_e, e1, e2)
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


@functools.cache
def _SecFlt(s, e):
    name = f'SecFlt{s + e}:{s}:{e}'
    secflt = type(name, (SecureFloat,), {'__slots__': ()})
    secflt.__doc__ = 'Class of secret-shared floating-point numbers.'
    secflt.bit_length = s + e
    secflt.significand_type = SecFxp(s+1, s-1)  # NB: 1 sign bit, 1 extra bit, s-1 fractional bits
    secflt.exponent_type = SecInt(e)
    globals()[name] = secflt  # NB: exploit (almost) unique name dynamic SecureFloat type
    return secflt


class SecureArray(SecureObject):
    """Base class for secure (secret-shared) number arrays."""

    __slots__ = 'shape'

    sectype: type

    def __init__(self, value=None, shape=None):
        """Initialize a secure array.

        The given value must be None, a Future, or a finite field array of correct type.
        If value is None (default) or a Future, shape must not be None.

        The given shape must be a (possibly empty) tuple of nonnegative integers.
        If shape is None (default), value should be a finite field array.
        """
        if isinstance(value, Future):
            pass
        elif value is not None:
            shape = value.value.shape
        assert shape is not None
        self.shape = shape
        super().__init__(value)

    def __bool__(self):
        """Return True if secure array is nonempty, False otherwise."""
        return bool(self.size)

    def __array_function__(self, func, types, args, kwargs):
        # minimal redirect for now
        return eval(f'runtime.np_{func.__name__}')(*args, **kwargs)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return math.prod(self.shape)

    def _coerce(self, other):
        if isinstance(other, SecureArray):
            if not isinstance(other, type(self)):
                return NotImplemented

        elif isinstance(other, SecureObject):
            if not isinstance(other, type(self).sectype):
                return NotImplemented

        elif isinstance(other, int):
            other = type(self)(np.array(other))
        elif isinstance(other, np.ndarray):
            other = type(self)(other)
        elif isinstance(other, type(self).sectype.field.array):
            other = type(self)(other)
        elif isinstance(other, type(self).sectype.field):
            other = type(self).sectype(other)
        else:
            return NotImplemented

        return other

    def _coerce2(self, other):
        if isinstance(other, SecureArray):
            if not isinstance(other, type(self)):
                return NotImplemented

        elif isinstance(other, SecureObject):
            if not isinstance(other, type(self).sectype):
                return NotImplemented

        elif isinstance(other, int):
            pass
        elif isinstance(other, np.ndarray):
            pass
        elif isinstance(other, type(self).sectype.field.array):
            pass
        elif isinstance(other, type(self).sectype.field):
            pass
        else:
            return NotImplemented

        return other

    def __neg__(self):
        """Matrix negation."""
        return runtime.np_negative(self)

    def __abs__(self):
        """Matrix absolute value."""
        return runtime.np_absolute(self)

    def __add__(self, other):
        """Matrix addition."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return runtime.np_add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        """Matrix subtraction."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return runtime.np_subtract(self, other)

    def __rsub__(self, other):
        """Matrix subtraction."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return runtime.np_subtract(other, self)

    def __mul__(self, other):
        """Multiplication."""
        other = self._coerce2(other)
        if other is NotImplemented:
            return NotImplemented

        return runtime.np_multiply(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Division."""
        other = self._coerce2(other)
        if other is NotImplemented:
            return NotImplemented

        return runtime.np_divide(self, other)

    def __rtruediv__(self, other):
        """Division (with reflected arguments)."""
        other = self._coerce2(other)
        if other is NotImplemented:
            return NotImplemented

        return runtime.np_divide(other, self)

    def __pow__(self, other):
        """Exponentiation for public integral exponent."""
        # TODO: extend to secret exponent
        if not isinstance(other, int):  # TODO: extend to np.array
            return NotImplemented

        return runtime.np_pow(self, other)

    def __matmul__(self, other):
        """Matrix multiplication."""
        return runtime.np_matmul(self, other)

    def __rmatmul__(self, other):
        """Matrix multiplication (with reflected arguments)."""
        return runtime.np_matmul(other, self)

    def __lt__(self, other):
        """Strictly less-than comparison."""
        # self < other
        return runtime.np_less(self, other)

    def __le__(self, other):
        """Less-than or equal comparison."""
        # self <= other <=> not (other < self)
        return 1 - runtime.np_less(other, self)

    def __eq__(self, other):
        """Equality testing."""
        # self == other
        return runtime.np_equal(self, other)

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        # self >= other <=> not (self < other)
        return 1 - runtime.np_less(self, other)

    def __gt__(self, other):
        """Strictly greater-than comparison."""
        # self > other <=> other < self
        return runtime.np_less(other, self)

    def __ne__(self, other):
        """Negated equality testing."""
        # self != other <=> not (self == other)
        return 1 - runtime.np_equal(self, other)

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    @property
    def T(self):
        return self.transpose()

    @property
    def flat(self):
        # via flatten(), no MPyC coroutine for generators yet
        yield from self.flatten()

    def __len__(self):
        if self.shape == ():
            # Let NumPy generate error message by calling len(a) for dummy shape-() array a:
            return len(np.array(0))

        return self.shape[0]

    def __getitem__(self, i):
        return runtime.np_getitem(self, i)

    # TODO: __setitem__(self, i, x) or runtime.np_update(a, i, x) instead

    def flatten(self, order='C'):
        return runtime.np_flatten(self, order=order)

    def tolist(self):
        return runtime.np_tolist(self)

    def reshape(self, *shape, order='C'):
        # NB: numpy.reshape only accepts a tuple (or a single int) as shape
        return runtime.np_reshape(self, shape, order=order)

    def copy(self, order='C'):
        return runtime.np_copy(self, order=order)

    def transpose(self, *axes):
        if axes == ():
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = axes[0]
        return runtime.np_transpose(self, axes=axes)

    def swapaxes(self, axis1, axis2):
        return runtime.np_swapaxes(self, axis1, axis2)

    def sum(self, *args, **kwargs):
        return runtime.np_sum(self, *args, **kwargs)

    def sort(self, *args, **kwargs):
        """Returns new array sorted along an axis.

        By default, axis=-1.
        If axis is None, the array is flattened.
        """
        return runtime.np_sort(self, *args, **kwargs)

    def argmin(self, *args, **kwargs):
        """Returns the indices of the minimum values along an axis.

        If no axis is given (default), array is flattened first.

        By default, the indices are returned as unit vectors.
        Also, by default, the minimum values are returned (next to the indices).

        NB: Different defaults than for np_argmin(). Latter behaves like np.argmin()
        for NumPy arrays, returning the indices as numbers and omitting the minimum values.
        """
        if 'arg_unary' not in kwargs:
            kwargs['arg_unary'] = True
        if 'arg_only' not in kwargs:
            kwargs['arg_only'] = False
        return runtime.np_argmin(self, *args, **kwargs)

    def argmax(self, *args, **kwargs):
        """Returns the indices of the maximum values along an axis.

        If no axis is given (default), array is flattened first.

        By default, the indices are returned as unit vectors.
        Also, by default, the maximum values are returned (next to the indices).

        NB: Different defaults than for np_argmax(). Latter behaves like np.argmax()
        for NumPy arrays, returning the indices as numbers and omitting the maximum values.
        """
        if 'arg_unary' not in kwargs:
            kwargs['arg_unary'] = True
        if 'arg_only' not in kwargs:
            kwargs['arg_only'] = False
        return runtime.np_argmax(self, *args, **kwargs)


class SecureFiniteFieldArray(SecureArray):
    """Base class for secure (secret-shared) arrays of finite field elements."""

    __slots__ = ()

    frac_length = 0

    _output_conversion = None

    def __init__(self, value=None, shape=None):
        """Initialize a secure finite field array to the given value.

        If value is None (default), shape must not be None.
        The given value must be array of the appropriate type (int/polynomial/finite field array).
        The given shape must be a (possibly empty) tuple of nonnegative integers.
        """
        if value is not None:
            if isinstance(value, np.ndarray):
                value = self.sectype.field.array(value)
            elif isinstance(value, self.sectype.field.array):
                pass
            elif isinstance(value, Future):
                pass  # NB: for internal use in runtime only
            else:
                # TODO: allow nested lists/tuples over int/poly/field-elt, possibly secfld
                if isinstance(value, finfields.FiniteFieldArray):
                    raise TypeError(f'incompatible finite field array {type(value).__name__} '
                                    f'for {type(self).__name__}')

                raise TypeError('None, int/polynomial array, or finite field array required')

        super().__init__(value, shape)


class SecureIntegerArray(SecureArray):
    """Base class for secure (secret-shared) integer arrays."""

    __slots__ = ()

    frac_length = 0

    @classmethod
    def _output_conversion(cls, a):
        return cls.sectype.field.array.intarray(a)
        # NB: returns dtype=object array b, say
        # Convert to NumPy int array, if desired, using calls like b.astype(np.int32).

    def __init__(self, value=None, shape=None):
        """Initialize a secure integer array to the given value.

        If value is None (default), shape must not be None.
        The given value must be array of the appropriate type (int/finite field array).
        The given shape must be a (possibly empty) tuple of nonnegative integers.
        """
        if value is not None:
            if isinstance(value, np.ndarray):
                value = self.sectype.field.array(value)
            elif isinstance(value, self.sectype.field.array):
                pass
            elif isinstance(value, Future):
                pass  # NB: for internal use in runtime only
            else:
                # TODO: allow nested lists/tuples over int, possibly secint
                if isinstance(value, finfields.FiniteFieldArray):
                    raise TypeError(f'incompatible finite field array {type(value).__name__} '
                                    f'for {type(self).__name__}')

                raise TypeError('None, int array, or finite field array required')

        super().__init__(value, shape)


class SecureFixedPointArray(SecureArray):
    """Base class for secure (secret-shared) arrays of fixed-point numbers."""

    __slots__ = 'integral'

    frac_length = 0

    @classmethod
    def _output_conversion(cls, a):
        a = cls.sectype.field.array.intarray(a) / 2**cls.frac_length
        if isinstance(a, np.ndarray):
            a = a.astype(float)
        return a

    def __init__(self, value=None, shape=None, integral=None):
        """Initialize a secure fixed-point array to the given value.

        If value is None (default), shape must not be None.
        The given value must be an array of the appropriate type (int/float/finite field array).
        The given shape must be a (possibly empty) tuple of nonnegative integers.
        """
        if value is not None:
            if isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.floating):
                    if integral is None:
                        integral = np.vectorize(lambda a: a.is_integer())(value).all()
                        integral = bool(integral)  # NB: from np.bool_ to bool
                    f2 = 1 << self.frac_length
                    # Scale to Python int entries (by setting otypes='O', prevents overflow):
                    value = np.vectorize(round, otypes='O')(value * f2)
                    value = self.sectype.field.array(value)
                elif np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, object):
                    if integral is None:
                        integral = True
                    if np.issubdtype(value.dtype, np.integer):
                        # Convert np.integer to Python int entries, to prevent overflow:
                        value = value.astype(object)
                    value <<= self.frac_length  # NB: fails for np.floating entries in array value
                    value = self.sectype.field.array(value)
                else:
                    raise TypeError(f'Invalid dtype {value.dtype}')

            elif isinstance(value, self.sectype.field.array):
                pass
            elif isinstance(value, Future):
                pass  # NB: for internal use in runtime only
            else:
                # TODO: allow nested lists/tuples over int/float, possibly secfxp
                if isinstance(value, finfields.FiniteFieldArray):
                    raise TypeError(f'incompatible finite field array {type(value).__name__} '
                                    f'for {type(self).__name__}')

                raise TypeError('None, int/float array, or finite field array required')

        self.integral = integral
        super().__init__(value, shape)

    def _coerce(self, other):
        if isinstance(other, float):
            return type(self)(np.array(other))

        return super()._coerce(other)

    def _coerce2(self, other):
        if isinstance(other, float):
            return other  # TODO: consider returning np.array(other) here

        return super()._coerce2(other)

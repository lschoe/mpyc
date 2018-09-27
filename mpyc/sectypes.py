"""This module collects the secure (secret-shared) types for MPyC.

Secure (secret-shared) number types all use a common base class, which
ensures that operators such as +, *, >= are defined by operator overloading.
"""

import asyncio
from mpyc import pfield

class Share:
    """A secret-shared value.

    An MPC protocol operates on secret-shared values, represented by Share
    objects. The basic Python operators are overloaded for Share objects.
    An expression like a * b will create a new Share object, which will
    eventually contain the product of a and b. The product is computed
    asynchronously, using an instance of a specific cryptographic protocol.
    """

    __slots__ = 'df'

    def __init__(self, field=None, value=None):
        """Initialize a share."""
        if value is not None:
            if isinstance(value, int):
                value = field(value)
            self.df = value
        else:
            self.df = asyncio.Future()

    def __bool__(self):
        """Use of secret-shared values in Boolean expressions makes no sense."""
        raise TypeError('cannot use secure type in Boolean expressions')

    def __neg__(self):
        """Negation."""
        return self.runtime.neg(self)

    def __add__(self, other):
        """Addition."""
        if isinstance(other, Share):
            if type(self) != type(other):
                return NotImplemented
        elif isinstance(other, float):
            if type(self).field.frac_length == 0:
                return NotImplemented
            else:
                other = type(self)(other)
        elif isinstance(other, int):
            other = type(self)(other)
        elif type(self).field != type(other):
            return NotImplemented
        return self.runtime.add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        """Subtraction."""
        if isinstance(other, Share):
            if type(self) != type(other):
                return NotImplemented
        elif isinstance(other, float):
            if type(self).field.frac_length == 0:
                return NotImplemented
            else:
                other = type(self)(other)
        elif isinstance(other, int):
            other = type(self)(other)
        elif type(self).field != type(other):
            return NotImplemented
        return self.runtime.sub(self, other)

    def __rsub__(self, other):
        """Subtraction (reflected argument version)."""
        if isinstance(other, Share):
            if type(self) != type(other):
                return NotImplemented
        elif isinstance(other, float):
            if type(self).field.frac_length == 0:
                return NotImplemented
            else:
                other = type(self)(other)
        elif isinstance(other, int):
            other = type(self)(other)
        elif type(self).field != type(other):
            return NotImplemented
        return self.runtime.sub(other, self)

    def __mul__(self, other):
        """Multiplication."""
        if isinstance(other, Share):
            if type(self) != type(other):
                return NotImplemented
        elif isinstance(other, float):
            if type(self).field.frac_length == 0:
                return NotImplemented
            else:
                other = type(self)(other)
        elif isinstance(other, int):
#            other <<= type(self).field.frac_length
            pass
        elif type(self).field != type(other):
            return NotImplemented
        return self.runtime.mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Division."""
        return self.runtime.div(self, other)

    __floordiv__ = __truediv__

    def __rtruediv__(self, other):
        """Division (reflected argument version)."""
        return self.runtime.div(other, self)

    __rfloordiv__ = __rtruediv__

    def __pow__(self, exponent):
        """Exponentation with publicly known integer exponent."""
        return self.runtime.pow(self, exponent)

    def __and__(self, other):
        """And 1-bit."""
        return self.runtime.and_(self, other)

    __rand__ = __and__

    def __xor__(self, other):
        """Exclusive-or 1-bit."""
        return self.runtime.xor(self, other)

    __rxor__ = __xor__

    def __invert__(self):
        """Inversion (not) 1-bit."""
        return self.runtime.invert(self)

    def __or__(self, other):
        """Or 1-bit."""
        return self.runtime.or_(self, other)

    __ror__ = __or__

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        # self >= other
        c = self - other
        if type(c).__name__.startswith('SecFld'):
            return NotImplemented
        return self.runtime.sgn(c, GE=True)

    def __gt__(self, other):
        """Strictly greater-than comparison."""
        # self > other <=> not (self <= other)
        c = other - self
        if type(c).__name__.startswith('SecFld'):
            return NotImplemented
        return 1 - self.runtime.sgn(c, GE=True)

    def __le__(self, other):
        """Less-than or equal comparison."""
        # self <= other <=> other >= self
        c = other - self
        if type(c).__name__.startswith('SecFld'):
            return NotImplemented
        return self.runtime.sgn(c, GE=True)

    def __lt__(self, other):
        """Strictly less-than comparison."""
        # self < other <=> not (self >= other)
        c = self - other
        if type(c).__name__.startswith('SecFld'):
            return NotImplemented
        return 1 - self.runtime.sgn(c, GE=True)

    def __eq__(self, other):
        """Equality testing."""
        # self == other
        c = self - other
        return self.runtime.is_zero(c)

    def __ne__(self, other):
        """Negated equality testing."""
        # self != other
        c = self - other
        return 1 - self.runtime.is_zero(c)


_sectypes = {}

def SecFld(p=None, l=None):
    """Secure prime field of l-bit order p."""
    if p is None:
        if l is None:
            l = 1
        p = pfield.find_prime_root(l, blum=False)[0]
    else:
        l = p.bit_length()
    assert p > len(Share.runtime.parties), 'Prime field order must exceed number of parties.'
    # p >= number of parties for MDS
    field = pfield.GF(p)
    field.is_signed = False

    if (l, p) not in _sectypes:
        class SecureFld(Share):
            __slots__ = ()
            def __init__(self, value=None):
                super().__init__(field, value)
        SecureFld.field = field
        SecureFld.bit_length = l
        name = f'SecFld{SecureFld.bit_length}({SecureFld.field.modulus})' 
        _sectypes[(l, p)] = type(name, (SecureFld,), {'__slots__':()})
    return _sectypes[(l, p)]

def _SecNum(l, f, p, n):
    k = Share.runtime.options.security_parameter
    if p is None:
        p = pfield.find_prime_root(l + max(f, k + 1) + 1, n=n)
    else:
        assert p.bit_length() > l + max(f, k + 1), f'Prime {p} too small.'
    field = pfield.GF(p, f)

    class SecureNum(Share):
        __slots__ = ()
        def __init__(self, value=None):
            super().__init__(field, value)
    SecureNum.field = field
    SecureNum.bit_length = l
    return SecureNum

def SecInt(l=None, p=None, n=2):
    """Secure l-bit integers."""
    if l is None:
        l = Share.runtime.options.bit_length

    if (l, 0, p, n) not in _sectypes:
        SecureInt = _SecNum(l, 0, p, n)
        if p is None:
            name = f'SecInt{l}'
        else:
            name = f'SecInt{l}({p})'
        _sectypes[(l, 0, p, n)] = type(name, (SecureInt,), {'__slots__':()})
    return _sectypes[(l, 0, p, n)]

def SecFxp(l=None, f=None, p=None, n=2):
    """Secure l-bit fixed-point numbers with f-bit fractional part.
    
    NB: if dividing secure fixed-point numbers, make sure that l =~ 2f.
    """
    if l is None:
        l = Share.runtime.options.bit_length
    if f is None:
        f = l // 2 # l =~ 2f enables division such that x =~ 1/(1/x)

    if (l, f, p, n) not in _sectypes:
        SecureFxp = _SecNum(l, f, p, n)
        if p is None:
            name = f'SecFxp{l}:{f}' 
        else:
            name = f'SecFxp{l}:{f}({p})'
        def init(self, value=None, integral=False):
            if value is not None:
                self.integral = isinstance(value, int)
                if self.integral:
                    value <<= f
                elif isinstance(value, float):
                    value = round(value * (1 << f))
                else:
                    self.integral = integral
            else:
                self.integral = integral
            super(_sectypes[(l, f, p, n)], self).__init__(value)
        _sectypes[(l, f, p, n)] = type(name, (SecureFxp,), {'__slots__':'integral', '__init__':init})
    return _sectypes[(l, f, p, n)]

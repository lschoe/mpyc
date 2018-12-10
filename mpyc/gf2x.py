"""This module supports arithmetic with polynomials over GF(2).

Polynomials over GF(2) are represented as nonnegative integers.
The polynomial b_n x^n + ... + b_1 x + b_0 corresponds
to the integer b_n 2^n + ... + b_1 2 + b_0, for bits b_n,...,b_0.

The operators +, -, *, //, %, and function divmod are overloaded.
Using the direct correspondence between polynomials and integers,
the operators <, <=, >, >=, ==, != are overloaded as well, where
the zero polynomial is the smallest polynomial.

GCD, extended GCD, modular inverse and powers are all supported.
A simple irreducibility test is provided as well as a basic
routine to find the next largest irreducible polynomial.
"""

def add(a, b):
    """Add polynomials a and b."""
    if isinstance(a, Polynomial):
        a = a.value
    if isinstance(b, Polynomial):
        b = b.value
    return Polynomial(a ^ b)

def mul(a, b):
    """Multiply polynomials a and b."""
    if isinstance(a, Polynomial):
        a = a.value
    if isinstance(b, Polynomial):
        b = b.value
    return Polynomial(_mul(a, b))

def _mul(a, b):
    if a < b:
        a, b = b, a
    # a >= b
    c = 0
    while b:
        if b & 1:
            c ^= a
        a <<= 1
        b >>= 1
    return c

def mod(a, b):
    """Reduce polynomial a modulo polynomial b, for nonzero b."""
    if isinstance(a, Polynomial):
        a = a.value
    if isinstance(b, Polynomial):
        b = b.value
    return Polynomial(_mod(a, b))

def _mod(a, b):
    if b is None: # see _powmod()
        return a
    if b == 0:
        raise ZeroDivisionError('division by zero polynomial')
    m = _degree(a)
    n = _degree(b)
    if m < n:
        return a
    b <<= m - n
    for i in range(m - n + 1):
        if (a >> m - i) & 1:
            a ^= b
        b >>= 1
    return a

def divmod_(a, b):
    """Divide polynomial a by polynomial b with remainder, for nonzero b."""
    if isinstance(a, Polynomial):
        a = a.value
    if isinstance(b, Polynomial):
        b = b.value
    q, r = _divmod(a, b)
    return Polynomial(q), Polynomial(r)

def _divmod(a, b):
    if b == 0:
        raise ZeroDivisionError('division by zero polynomial')
    m = _degree(a)
    n = _degree(b)
    if m < n:
        return 0, a
    b <<= m - n
    q = 0
    for i in range(m - n + 1):
        q <<= 1
        if (a >> m - i) & 1:
            a ^= b
            q ^= 1
        b >>= 1
    return q, a

def gcd(a, b):
    """Greatest common divisor of polynomials a and b."""
    if isinstance(a, Polynomial):
        a = a.value
    if isinstance(b, Polynomial):
        b = b.value
    return Polynomial(_gcd(a, b))

def _gcd(a, b):
    while b:
        a, b = b, _mod(a, b)
    return a

def gcdext(a, b):
    """Extended GCD for polynomials a and b.

    Return d, s, t satisfying s a + t b = d = gcd(a,b).
    """
    if isinstance(a, Polynomial):
        a = a.value
    if isinstance(b, Polynomial):
        b = b.value
    d, s, t = _gcdext(a, b)
    return Polynomial(d), Polynomial(s), Polynomial(t)

def _gcdext(a, b):
    s, s1 = 1, 0
    t, t1 = 0, 1
    while b:
        q, r = _divmod(a, b)
        a, b = b, r
        s, s1 = s1, s ^ _mul(q, s1)
        t, t1 = t1, t ^ _mul(q, t1)
    return a, s, t

def invert(a, b):
    """Inverse of polynomial a modulo polynomial b, for nonzero b."""
    if isinstance(a, Polynomial):
        a = a.value
    if isinstance(b, Polynomial):
        b = b.value
    return Polynomial(_invert(a, b))

def _invert(a, b):
    if b == 0:
        raise ZeroDivisionError('division by zero polynomial')
    s, s1 = 1, 0
    while b:
        q, r = _divmod(a, b)
        a, b = b, r
        s, s1 = s1, s ^ _mul(q, s1)
    if a != 1:
        raise ZeroDivisionError('inverse does not exist')
    return s

def to_terms(a, x='x'):
    """Convert polynomial a to a string with sum of powers of x."""
    if isinstance(a, Polynomial):
        a = a.value
    return _to_terms(a, x)

def _to_terms(a, x='x'):
    if a == 0:
        return '0'
    p = ''
    for i in range(a.bit_length(), -1, -1):
        if (a >> i) & 1:
            if i == 0:
                p += '+1'    # x^0 = 1
            elif i == 1:
                p += f'+{x}' # x^1 = x
            else:
                p += f'+{x}^{i}'
    return p[1:]

def from_terms(s, x='x'):
    """Convert string s with sum of powers of x to a polynomial."""
    return Polynomial(_from_terms(s, x))

def _from_terms(s, x='x'):
    s = "".join(s.split()) # remove all whitespace
    a = 0
    for term in s.split('+'):
        if term == '0':
            t = 0
        elif term == '1':
            t = 1 # 2^0
        elif term == x:
            t = 2 # 2^1
        elif term.startswith(f'{x}^'):
            t = 1 << int(term[2:], base=0)
        else: # illegal term
            raise ValueError('ill formatted polynomial')
        if a & t: # repeated term
            raise ValueError('ill formatted polynomial')
        else:
            a ^= t
    return a

def degree(a):
    """Degree of polynomial a (-1 if a is zero)."""
    if isinstance(a, Polynomial):
        a = a.value
    return _degree(a)

def _degree(a):
    return a.bit_length() - 1

def powmod(a, n, b):
    """Raise polynomial a to the power of n modulo polynomial b, for nonzero b."""
    if isinstance(a, Polynomial):
        a = a.value
    if isinstance(b, Polynomial):
        b = b.value
    return Polynomial(_powmod(a, n, b))

def _powmod(a, n, b=None):
    if n == 0:
        return 1
    if n < 0:
        if b is None:
            raise ValueError('negative exponent')
        a = _invert(a, b)
        n = -n
    d = a
    c = 1
    for i in range(n.bit_length() - 1):
        # d = a ** (1 << i) holds
        if n & (1 << i):
            c = _mul(c, d)
            c = _mod(c, b)
        d = _mul(d, d)
        d = _mod(d, b)
    c = _mul(c, d)
    c = _mod(c, b)
    return c

def is_irreducible(a):
    """Test polynomial a for irreducibility."""
    if isinstance(a, Polynomial):
        a = a.value
    return _is_irreducible(a)

def _is_irreducible(a):
    if a <= 1:
        return False
    b = 2
    for _ in range(_degree(a) // 2):
        b = _mul(b, b)
        b = _mod(b, a)
        if _gcd(b ^ 2, a) != 1:
            return False
    return True

def next_irreducible(a):
    """Return next irreducible polynomial > a.

    NB: 'x' < 'x+1' < 'x^2+x+1' < 'x^3+x+1' < 'x^3+x^2+1' < ...
    """
    if isinstance(a, Polynomial):
        a = a.value
    return Polynomial(_next_irreducible(a))

def _next_irreducible(a):
    if a <= 2:
        a += 1
    else:
        a += 1 + (a % 2)
        while not _is_irreducible(a):
            a += 2
    return a

class Polynomial:
    """Polynomials over GF(2) represented as nonnegative integers."""
    __slots__ = 'value'

    def __init__(self, value, x='x'):
        if isinstance(value, str):
            value = _from_terms(value, x)
        self.value = value

    def degree(self):
        """Degree of polynomial (-1 for zero polynomial)."""
        return _degree(self.value)

    def is_irreducible(self):
        """Test polynomial for irreducibility."""
        return _is_irreducible(self.value)

    def __int__(self):
        return self.value

    def __add__(self, other):
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return Polynomial(self.value ^ other)

    def __radd__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return Polynomial(self.value ^ other)

    def __iadd__(self, other):
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        self.value ^= other
        return self

    __sub__ = __add__
    __rsub__ = __radd__
    __isub__ = __iadd__

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return Polynomial(_mul(self.value, other))

    def __rmul__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return Polynomial(_mul(self.value, other))

    def __imul__(self, other):
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        self.value = _mul(self.value, other)
        return self

    def __floordiv__(self, other):
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return Polynomial(_divmod(self.value, other)[0])

    def __rfloordiv__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return Polynomial(_divmod(self.value, other)[0])

    def __ifloordiv__(self, other):
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        self.value = _divmod(self.value, other)[0]
        return self

    def __mod__(self, other):
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return Polynomial(_mod(self.value, other))

    def __rmod__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return Polynomial(_mod(self.value, other))

    def __imod__(self, other):
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        self.value = _mod(self.value, other)
        return self

    def __divmod__(self, other):
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return Polynomial(_divmod(self.value, other))

    def __rdivmod__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return Polynomial(_divmod(self.value, other))

    def __pow__(self, exponent):
        return Polynomial(_powmod(self.value, exponent))

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return self.value >= other

    def __gt__(self, other):
        """Strictly greater-than comparison."""
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return self.value > other

    def __le__(self, other):
        """Less-than or equal comparison."""
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return self.value <= other

    def __lt__(self, other):
        """Strictly less-than comparison."""
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return self.value < other

    def __repr__(self):
        return _to_terms(self.value)

    def __eq__(self, other):
        """Equality test."""
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return self.value == other

    def __ne__(self, other):
        """Negated equality testing."""
        if isinstance(other, Polynomial):
            other = other.value
        elif not isinstance(other, int):
            return NotImplemented
        return self.value != other

    def __hash__(self):
        """Hash value."""
        return hash((type(self), self.value))

    def __bool__(self):
        """Truth value testing.

        Return False if this field element is zero, True otherwise.
        Field elements can thus be used directly in Boolean formulas.
        """
        return bool(self.value)

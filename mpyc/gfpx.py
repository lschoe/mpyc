"""This module supports arithmetic with polynomials over GF(p).

Polynomials over GF(p) are represented as coefficient lists.
The polynomial a_0 + a_1 X + ... + a_n X^n corresponds
to the list [a_0, a_1, ... , a_n] of integers in {0, ... , p-1}.
Leading coefficient a_n is nonzero, using [] for the zero polynomial.

However, binary polynomials (over GF(2)) are represented as integers.
The polynomial a_0 + a_1 X + ... + a_n X^n corresponds
to the integer a_0 + a_1 2 + ... + a_n 2^n.
Leading coefficient a_n is 1, using 0 for the zero polynomial.

The operators +,-,*,<<,//,%, and function divmod are overloaded.
The operators <,<=,>,>=,==,!= are overloaded as well, using the
lexicographic order for polynomials (zero polynomial is the smallest).
Plus some SageMath-style functionality, for instance, to access
coefficients using Python indexing and to reverse a polynomial.

GCD, extended GCD, modular inverse and powers are all supported.
A simple irreducibility test is provided as well as a basic
routine to find the next largest irreducible polynomial.
"""

import functools
from mpyc import gmpy as gmpy2

X = 'x'  # symbol for indeterminate in polynomials


@functools.cache
def GFpX(p):
    """Create type for polynomials over GF(p)."""
    if not gmpy2.is_prime(p):
        raise ValueError('number is not prime')

    BasePolynomial = BinaryPolynomial if p == 2 else Polynomial
    GFpPolynomial = type(f'GF({p})[{X}]', (BasePolynomial,), {'__slots__': ()})
    GFpPolynomial.p = p
    globals()[f'GF({p})[{X}]'] = GFpPolynomial  # NB: exploit unique name dynamic Polynomial type
    return GFpPolynomial


class Polynomial:
    """Polynomials over GF(p) represented as lists of integers in {0, ... , p-1}.

    Invariant: last element of attribute 'value' is a nonzero integer (if 'value' nonempty).
    """

    __slots__ = 'value'

    p = None

    def __init__(self, value=0, check=True):
        """Initialize polynomial to given value (zero polynomial, by default)."""
        if check:
            value = self._intern(value)
        self.value = value

    @classmethod
    def _intern(cls, a):
        # convert a to cls internal format, if possible
        a = cls._coerce(a)
        if a is NotImplemented:
            raise TypeError(f'polynomial over GF({cls.p}) expected')

        return a

    @classmethod
    def _coerce(cls, a):
        if isinstance(a, Polynomial):
            if not isinstance(a, cls):
                raise TypeError(f'polynomial of type {cls.__name__} expected')

            return a.value

        if isinstance(a, int):
            return cls._from_int(a)

        if isinstance(a, str):
            return cls._from_terms(a)

        if isinstance(a, tuple):
            a = list(a)
        if isinstance(a, list):
            p = cls.p
            if not all(isinstance(a_i, int) and 0 <= a_i < p for a_i in a):
                raise ValueError('polynomial coefficients invalid or out of range')

            return cls._from_list(a)

        return NotImplemented

    def __int__(self):
        return self._to_int(self.value)

    def __getitem__(self, key):  # NB: no set_item to prevent mutability
        if isinstance(key, slice):
            # TODO: consider SageMath-style slicing for prefixes
            raise IndexError('slicing of polynomials not supported, use list() or similar')

        if not isinstance(key, int):
            # TODO: consider use of key.__index__() for more generality
            raise IndexError('use int for indexing polynomials')

        if key == -1 and not self.value:
            return 0  # e.g., for zero polynomial z we get z[z.degree()] == 0

        if key < 0:
            raise IndexError('negative index not allowed for nonzero polynomials')

        return self._getitem(key)

    def _getitem(self, key):
        try:
            v = self.value[key]
        except IndexError:
            v = 0
        return v

    def __iter__(self):
        yield from self.value

    def __call__(self, x):
        """Evaluate polynomial at given x."""
        p = type(self).p
        x = x % p
        y = 0
        for c in reversed(self.value):
            y *= x
            y += c
            y %= p
        return y

    def to_bytes(self, length, byteorder):
        """Return a bytes object representing a polynomial."""
        # TODO: consider coefficient-wise serialization (via numpy arrays)
        return self._to_int(self.value).to_bytes(length, byteorder)

    @classmethod
    def _from_int(cls, a):
        p = cls.p
        neg = a < 0
        if neg:
            a = -a
        c = []
        while a:
            a, r = divmod(a, p)
            c.append(p - r if neg and r else r)
        return c

    @classmethod
    def _to_int(cls, a):
        p = cls.p
        s = 0
        for ai in reversed(a):
            s *= p
            s += ai
        return s

    @staticmethod
    def _from_list(a):
        return a

    @staticmethod
    def _to_list(a):
        return a.value

    @classmethod
    def _from_terms(cls, s, x=X):
        p = cls.p
        d = {}
        s = ''.join(s.split())  # remove all whitespace
        for term in s.split('+'):
            try:
                if term.find(x) == -1:
                    c = int(term)
                    i = 0
                elif term.endswith(x):
                    c = term[:-1]
                    c = 1 if c == '' else int(c)
                    i = 1
                else:
                    c, i = term.split(f'{x}^')
                    c = 1 if c == '' else int(c)
                    i = int(i)
            except Exception as exc:
                raise ValueError('ill formatted polynomial') from exc

            d[i] = d.get(i, 0) + c

        m = max(d.keys(), default=-1)
        a = [0] * (m+1)
        for i, c in d.items():
            a[i] = c % p
        while a and not a[-1]:
            a.pop()
        return a

    @staticmethod
    def _to_terms(a, x=X):
        if a == []:
            return '0'

        s = ''
        for i in range(len(a) - 1, -1, -1):
            if a[i]:
                c = '' if a[i] == 1 else a[i]
                if i == 0:
                    s += f'+{a[i]}'  # x^0 = 1
                elif i == 1:
                    s += f'+{c}{x}'  # x^1 = x
                else:
                    s += f'+{c}{x}^{i}'
        return s[1:]

    @staticmethod
    def _deg(a):
        return len(a) - 1

    @classmethod
    def _monic(cls, a, lc_pinv=False):
        a1 = a[-1] if a else 0
        if a and a1 != 1:
            p = cls.p
            a = a[:]
            a1 = int(gmpy2.invert(a1, p))
            for i in range(len(a) - 1):
                a[i] *= a1
                a[i] %= p
            a[-1] = 1
        if lc_pinv:
            return a, a1  # attach pseudoinverse of leading coefficient

        return a

    @staticmethod
    def _reverse(a, d=None):
        if d is None:
            d = len(a) - 1
        # d >= -1
        a = a[:d+1]
        a.extend([0] * (d + 1 - len(a)))
        a.reverse()
        while a and not a[-1]:
            a.pop()
        return a

    @classmethod
    def _neg(cls, a):
        p = cls.p
        return [0 if a_i == 0 else p - a_i for a_i in a]

    @classmethod
    def _pos(cls, a):
        return a

    @classmethod
    def _add(cls, a, b):
        p = cls.p
        if len(a) < len(b):
            a, b = b, a
        # len(a) >= len(b)
        c = a[:]
        for i, b_i in enumerate(b):
            c[i] += b_i
            if c[i] >= p:
                c[i] -= p
        while c and not c[-1]:
            c.pop()
        return c

    @classmethod
    def _sub(cls, a, b):
        p = cls.p
        c = a + [0] * (len(b) - len(a))
        for i, b_i in enumerate(b):
            c[i] -= b_i
            if c[i] < 0:
                c[i] += p
        while c and not c[-1]:
            c.pop()
        return c

    @classmethod
    def _mul(cls, a, b):
        p = cls.p
        if len(a) > len(b):
            a, b = b, a
        # len(a) <= len(b)
        if not a:
            return []

        c = [0] * (len(a) + len(b) - 1)
        for i, a_i in enumerate(a):
            if a_i:
                for j, b_j in enumerate(b):
                    c[i + j] += a_i * b_j
        for i in range(len(c)):
            c[i] %= p
        return c

    @classmethod
    def _lshift(cls, a, n):
        return [0] * n + a

    @classmethod
    def _rshift(cls, a, n):
        return a[n:]

    @classmethod
    def _mod(cls, a, b):
        p = cls.p
        if b is None:  # see _powmod()
            return a  # NB: in-place

        if b == []:
            raise ZeroDivisionError('division by zero polynomial')

        m = len(a)
        n = len(b)
        if m < n:
            return a

        b1 = int(gmpy2.invert(b[-1], p))
        r = a[:]
        for i in range(m - n, -1, -1):
            if len(r) >= i + n:
                q_i = (r[-1] * b1) % p
                for j in range(n):
                    r[i + j] -= q_i * b[j]
                    r[i + j] %= p
                while r and not r[-1]:
                    r.pop()
        return r

    @classmethod
    def _divmod(cls, a, b):
        p = cls.p
        if b == []:
            raise ZeroDivisionError('division by zero polynomial')

        m = len(a)
        n = len(b)
        if m < n:
            return [], a

        b1 = int(gmpy2.invert(b[-1], p))
        q, r = [0] * (m - n + 1), a[:]
        for i in range(m - n, -1, -1):
            if len(r) >= i + n:
                q[i] = q_i = (r[-1] * b1) % p
                for j in range(n):
                    r[i + j] -= q_i * b[j]
                    r[i + j] %= p
                while r and not r[-1]:
                    r.pop()
        return q, r

    @classmethod
    def _powmod(cls, a, n, modulus=None):
        if n == 0:
            return cls._intern(1)

        if n < 0:
            if modulus is None:
                raise ValueError('negative exponent')

            a = cls._invert(a, modulus)
            n = -n
        b = a
        for i in range(n.bit_length()-2, -1, -1):
            b = cls._mul(b, b)
            b = cls._mod(b, modulus)
            if (n >> i) & 1:
                b = cls._mul(b, a)
                b = cls._mod(b, modulus)
        return b

    @classmethod
    def _gcd(cls, a, b):
        while b:
            a, b = b, cls._mod(a, b)
        a = cls._monic(a)
        return a

    @classmethod
    def _gcdext(cls, a, b):
        p = cls.p
        s, s1 = [1], []
        t, t1 = [], [1]
        while b:
            a, (q, b) = b, cls._divmod(a, b)
            s, s1 = s1, cls._sub(s, cls._mul(q, s1))
            t, t1 = t1, cls._sub(t, cls._mul(q, t1))

        a, a1 = cls._monic(a, lc_pinv=True)
        if a1 > 2:
            for i in range(len(s)):
                s[i] *= a1
                s[i] %= p
            for i in range(len(t)):
                t[i] *= a1
                t[i] %= p
        return a, s, t

    @classmethod
    def _invert(cls, a, b):
        p = cls.p
        if b == []:
            raise ZeroDivisionError('division by zero polynomial')

        s, s1 = [1], []
        while b:
            a, (q, b) = b, cls._divmod(a, b)
            s, s1 = s1, cls._sub(s, cls._mul(q, s1))
        if len(a) != 1:
            raise ZeroDivisionError('inverse does not exist')

        # ensure monic a
        a1 = int(gmpy2.invert(a[0], p))
        for i in range(len(s)):
            s[i] *= a1
            s[i] %= p
        return s

    @classmethod
    def _is_irreducible(cls, a):
        p = cls.p
        if cls._deg(a) <= 0:
            return False

        b = [0, 1]
        for _ in range(cls._deg(a) // 2):
            b = cls._powmod(b, p, modulus=a)
            if cls._gcd(cls._sub(b, [0, 1]), a) != [1]:
                return False

        return True

    @classmethod
    def _next_irreducible(cls, a):
        # TODO: skip X^d + i etc. to limit run-time for large(r) p
        p = cls.p
        a = cls._to_int(a)
        while True:
            a += 1
            if a % p == 0:
                a += 1
            _a = cls._from_int(a)
            if _a[-1] != 1:  # ensure monic a
                a = p**len(_a)
                continue
            if cls._is_irreducible(_a):
                break

        return _a

    @classmethod
    def from_terms(cls, s, x=X):
        """Convert string s with sum of powers of x to a polynomial."""
        return cls(cls._from_terms(s, x), check=False)

    @classmethod
    def to_terms(cls, a, x=X):
        """Convert polynomial a to a string with sum of powers of x."""
        a = cls._intern(a)
        return cls._to_terms(a, x)

    @classmethod
    def deg(cls, a):
        """Degree of polynomial a (-1 if a is zero polynomial)."""
        a = cls._intern(a)
        return cls._deg(a)

    def degree(self):
        """Degree of polynomial (-1 for zero polynomial)."""
        return self._deg(self.value)

    def monic(self, lc_pinv=False):
        """Monic version of polynomial.

        Zero polynomial remains unchanged.
        If lc_pinv is set, inverse of leading coefficient is also returned (0 for zero polynomial).
        """
        cls = type(self)
        a = cls._monic(self.value, lc_pinv=lc_pinv)
        if lc_pinv:
            a, a1 = a
            return cls(a, check=False), a1

        return cls(a, check=False)

    def reverse(self, d=None):
        """Reverse of polynomial (basically, coefficients in reverse order).

        For example, reverse of x + 2x^2 + 3x^3 is 3 + 2x + x^2.
        If d is None (default), d is set to the degree of the given poynomial.
        Otherwise, the given polynomial is first padded with zeros or truncated
        to attain the given degree d, d>=-1, before it is reversed.
        """
        cls = type(self)
        return cls(cls._reverse(self.value, d=d), check=False)

    def __neg__(self):
        cls = type(self)
        return cls(cls._neg(self.value), check=False)

    def __pos__(self):
        cls = type(self)
        return cls(cls._pos(self.value), check=False)

    @classmethod
    def add(cls, a, b):
        """Add polynomials a and b."""
        a = cls._intern(a)
        b = cls._intern(b)
        return cls(cls._add(a, b), check=False)

    def __add__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(cls._add(self.value, other), check=False)

    __radd__ = __add__

    @classmethod
    def sub(cls, a, b):
        """Subtract polynomials a and b."""
        a = cls._intern(a)
        b = cls._intern(b)
        return cls(cls._sub(a, b), check=False)

    def __sub__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(cls._sub(self.value, other), check=False)

    def __rsub__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(cls._sub(other, self.value), check=False)

    @classmethod
    def mul(cls, a, b):
        """Multiply polynomials a and b."""
        a = cls._intern(a)
        b = cls._intern(b)
        return cls(cls._mul(a, b), check=False)

    def __mul__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(cls._mul(self.value, other), check=False)

    __rmul__ = __mul__

    @classmethod
    def lshift(cls, a, n):
        """Multiply polynomial a by X^n."""
        a = cls._intern(a)
        return cls(cls._lshift(a, n), check=False)

    def __lshift__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        cls = type(self)
        return cls(cls._lshift(self.value, other), check=False)

    def __rlshift__(self, other):
        return NotImplemented

    @classmethod
    def rshift(cls, a, n):
        """Quotient for polynomial a divided by X^n, assuming a is multiple of X^n."""
        a = cls._intern(a)
        return cls(cls._rshift(a, n), check=False)

    def __rshift__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        cls = type(self)
        return cls(cls._rshift(self.value, other), check=False)

    def __rrshift__(self, other):
        return NotImplemented

    def __floordiv__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(cls._divmod(self.value, other)[0], check=False)

    def __rfloordiv__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(cls._divmod(other, self.value)[0], check=False)

    @classmethod
    def mod(cls, a, b):
        """Reduce polynomial a modulo polynomial b, for nonzero b."""
        a = cls._intern(a)
        b = cls._intern(b)
        return cls(cls._mod(a, b), check=False)

    def __mod__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(cls._mod(self.value, other), check=False)

    def __rmod__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(cls._mod(other, self.value), check=False)

    @classmethod
    def divmod(cls, a, b):
        """Divide polynomial a by polynomial b with remainder, for nonzero b."""
        a = cls._intern(a)
        b = cls._intern(b)
        q, r = cls._divmod(a, b)
        return cls(q, check=False), cls(r, check=False)

    def __divmod__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        q, r = cls._divmod(self.value, other)
        return cls(q, check=False), cls(r, check=False)

    def __rdivmod__(self, other):
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        q, r = cls._divmod(other, self.value)
        return cls(q, check=False), cls(r, check=False)

    @classmethod
    def powmod(cls, a, n, b):
        """Polynomial a to the power of n modulo polynomial b, for nonzero b."""
        a = cls._intern(a)
        b = cls._intern(b)
        return cls(cls._powmod(a, n, modulus=b), check=False)

    def __pow__(self, other):
        cls = type(self)
        return cls(cls._powmod(self.value, other), check=False)

    @classmethod
    def gcd(cls, a, b):
        """Greatest common divisor of polynomials a and b."""
        a = cls._intern(a)
        b = cls._intern(b)
        return cls(cls._gcd(a, b), check=False)

    @classmethod
    def gcdext(cls, a, b):
        """Extended GCD for polynomials a and b.

        Return d, s, t satisfying s a + t b = d = gcd(a,b).
        """
        a = cls._intern(a)
        b = cls._intern(b)
        d, s, t = cls._gcdext(a, b)
        return cls(d, check=False), cls(s, check=False), cls(t, check=False)

    @classmethod
    def invert(cls, a, b):
        """Inverse of polynomial a modulo polynomial b, for nonzero b."""
        a = cls._intern(a)
        b = cls._intern(b)
        return cls(cls._invert(a, b), check=False)

    @classmethod
    def is_irreducible(cls, a):
        """Test polynomial a for irreducibility."""
        a = cls._intern(a)
        return cls._is_irreducible(a)

    @classmethod
    def next_irreducible(cls, a):
        """Return lexicographically next monic irreducible polynomial > a.

        E.g., X < X+1 < X^2+X+1 < X^3+X+1 < X^3+X^2+1 < ... for p=2.
        """
        a = cls._intern(a)
        return cls(cls._next_irreducible(a), check=False)

    def __repr__(self):
        return self._to_terms(self.value)

    def __lt__(self, other):
        """Strictly less-than comparison."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return self._to_int(self.value) <= self._to_int(other)

    def __le__(self, other):
        """Less-than or equal comparison."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return self._to_int(self.value) < self._to_int(other)

    def __eq__(self, other):
        """Equality test."""
        other = self._coerce(other)
        if other is NotImplemented:
            return False

        return self.value == other

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return self._to_int(self.value) >= self._to_int(other)

    def __gt__(self, other):
        """Strictly greater-than comparison."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return self._to_int(self.value) > self._to_int(other)

    def __ne__(self, other):
        """Negated equality test."""
        other = self._coerce(other)
        if other is NotImplemented:
            return True

        return self.value != other

    def __hash__(self):
        """Make polynomials hashable (e.g., for LRU caching)."""
        return hash((type(self).__name__, tuple(self.value)))

    def __bool__(self):
        """Truth value testing.

        Return False if this polynomial is zero, True otherwise.
        """
        return bool(self.value)


class BinaryPolynomial(Polynomial):
    """Polynomials over GF(2) represented as nonnegative integers."""

    __slots__ = ()

    p = 2

    def __int__(self):
        return self.value

    def _getitem(self, key):
        return (self.value >> key) & 1

    def __iter__(self):
        a = self.value
        while a:
            a, r = divmod(a, 2)
            yield r

    def __call__(self, x):
        """Evaluate polynomial at given x."""
        return bin(self.value).count('1', 2)%2 if x%2 else 0

    def __hash__(self):
        """Make polynomials hashable (e.g., for LRU caching)."""
        return hash((type(self).__name__, self.value))

    def to_bytes(self, length, byteorder):
        return self.value.to_bytes(length, byteorder)

    @staticmethod
    def _from_int(a):
        return abs(a)

    @staticmethod
    def _to_int(a):
        return a

    @staticmethod
    def _from_list(a):
        s = 0
        for ai in reversed(a):
            s <<= 1
            s += ai
        return s

    @staticmethod
    def _to_list(a):
        c = []
        while a:
            a, r = divmod(a, 2)
            c.append(r)
        return c

    @staticmethod
    def _from_terms(s, x=X):
        s = ''.join(s.split())  # remove all whitespace
        a = 0
        for term in s.split('+'):
            if term == '0':
                t = 0
            elif term == '1':
                t = 1  # 2^0
            elif term == x:
                t = 2  # 2^1
            elif term.startswith(f'{x}^'):
                t = 1 << int(term[2:], base=0)
            else:  # invalid term
                raise ValueError('ill formatted polynomial')

            a ^= t
        return a

    @staticmethod
    def _to_terms(a, x=X):
        if a == 0:
            return '0'

        s = ''
        for i in range(a.bit_length(), -1, -1):
            if (a >> i) & 1:
                if i == 0:
                    s += '+1'     # x^0 = 1
                elif i == 1:
                    s += f'+{x}'  # x^1 = x
                else:
                    s += f'+{x}^{i}'
        return s[1:]

    @staticmethod
    def _deg(a):
        return a.bit_length() - 1

    @staticmethod
    def _monic(a, lc_pinv=False):
        if lc_pinv:
            return a, int(a != 0)

        return a

    @staticmethod
    def _reverse(a, d=None):
        e = 0 if d is None else d + 1 - a.bit_length()
        if e < 0:
            a &= (1 << d+1) - 1  # truncate to d+1 >= 0 least-significant bits
        a = int(format(a, 'b')[::-1], 2)  # reverse bits in a
        if e > 0:
            a <<= e  # pad with e zeros
        return a

    @staticmethod
    def _neg(a):
        return a

    @staticmethod
    def _pos(a):
        return a

    @staticmethod
    def _add(a, b):
        return a ^ b

    _sub = _add

    @staticmethod
#    @functools.cache
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

    @staticmethod
    def _lshift(a, n):
        return a << n

    @staticmethod
    def _rshift(a, n):
        return a >> n

    @staticmethod
#    @functools.cache
    def _mod(a, b):
        if b is None:  # see _powmod()
            return a

        if b == 0:
            raise ZeroDivisionError('division by zero polynomial')

        m = a.bit_length()
        n = b.bit_length()
        if m < n:
            return a

        b <<= m - n
        a ^= b
        for i in range(m-2, n-2, -1):
            b >>= 1
            if (a >> i) & 1:
                a ^= b
        return a

    @staticmethod
    def _divmod(a, b):
        if b == 0:
            raise ZeroDivisionError('division by zero polynomial')

        m = a.bit_length()
        n = b.bit_length()
        if m < n:
            return 0, a

        b <<= m - n
        q = 1
        a ^= b
        for i in range(m-2, n-2, -1):
            b >>= 1
            q <<= 1
            if (a >> i) & 1:
                q ^= 1
                a ^= b
        return q, a

    @staticmethod
    def _gcd(a, b):
        while b:
            a, b = b, BinaryPolynomial._mod(a, b)
        return a

    @staticmethod
    def _gcdext(a, b):
        s, s1 = 1, 0
        t, t1 = 0, 1
        while b:
            a, (q, b) = b, BinaryPolynomial._divmod(a, b)
            s, s1 = s1, s ^ BinaryPolynomial._mul(q, s1)
            t, t1 = t1, t ^ BinaryPolynomial._mul(q, t1)
        return a, s, t

    @staticmethod
    def _invert(a, b):
        if b == 0:
            raise ZeroDivisionError('division by zero polynomial')

        s, s1 = 1, 0
        while b:
            a, (q, b) = b, BinaryPolynomial._divmod(a, b)
            s, s1 = s1, s ^ BinaryPolynomial._mul(q, s1)
        if a != 1:
            raise ZeroDivisionError('inverse does not exist')

        return s

    @staticmethod
    def _is_irreducible(a):
        if a <= 1:
            return False

        b = 2
        for _ in range(BinaryPolynomial._deg(a) // 2):
            b = BinaryPolynomial._mul(b, b)
            b = BinaryPolynomial._mod(b, a)
            if BinaryPolynomial._gcd(b^2, a) != 1:
                return False

        return True

    @staticmethod
    def _next_irreducible(a):
        if a <= 1:
            a = 2
        else:
            a += 1 + a%2
            while not BinaryPolynomial._is_irreducible(a):
                a += 2
        return a

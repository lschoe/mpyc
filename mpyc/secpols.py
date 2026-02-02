"""This module supports secure (univariate) polynomial arithmetic.

A secure polynomial can be viewed as a sequence of secret-shared coefficients.
The length of the sequence is public, but the leading coefficient is not required
to be nonzero. This way only an upper bound on the degree is revealed.

The sequence of coefficients is implemented as a secure (NumPy) array.
A polynomial a_0 + a_1 X + ... + a_d X^d is represented by an array of length n >= d+1
of the form [a_0 a_1 ... a_d 0 ... 0], where the number of trailing zeros may vary.

Currently, only polynomials over prime fields GF(p) are supported, as secure counterpart
to MPyC's gfpx module. (Secure polynomials over integers and fixed-point numbers will be
considered later.) Moreover, for certain operations, p must be sufficiently large, in
particular compared to (the public upper bound on) the degree of a given polynomial.

The operators +,-,*,<<,>>,**,//,%, and function divmod are overloaded, providing secure
polynomial arithmetic, while hiding the exact degree of the results throughout.
The operators <,<=,>,>=,==,!= are overloaded as well, using the
lexicographic order for polynomials (zero polynomial is the smallest).
Evaluation of a polynomial at a public or secret point is supported as well.

A couple more advanced secure operations such as GCD, extended GCD, modular inverse,
and modular powers are also supported as well as a simple irreducibility test.

Current implementation is relatively basic, not yet fully optimized.
"""

import operator
from mpyc.numpy import np
from mpyc.gfpx import GFpX, Polynomial
from mpyc.asyncoro import mpc_coro
from mpyc.sectypes import SecureObject, SecureFiniteFieldArray
from mpyc.mpctools import reduce

runtime = None


class secpoly(SecureObject):

    __slots__ = ()

    def __init__(self, value=None, sectype=None, shape=None):
        """Initialize a secure polynomial to the given value, where value is a GFpX polynomial,
        a 1D int array, or a 1D secure (finite field) array.

        If value is None (default), sectype must be given and shape must be a 1D shape.
        Also, if value is an int array, sectype must be given.

        If sectype is None, it is inferred from the given value.
        """
        if isinstance(value, Polynomial):
            if sectype is None:
                sectype = runtime.SecFld(value.p)
            value = sectype.array(np.array(value._to_list(value), dtype=object))
        elif isinstance(value, np.ndarray):
            value = sectype.array(value)
        elif value is None:
            assert shape is not None and len(shape) == 1
            value = sectype.array(shape=shape)    # empty secure array (placeholder)
        elif not isinstance(value, SecureFiniteFieldArray):
            raise TypeError('None, polynomial, int array, or secure field array required.')

        super().__init__(value)
        if self.sectype is None:
            raise ValueError('sectype missing')

    @property
    def sectype(self):
        """Secure type of coefficients."""
        return self.share.sectype

    def set_share(self, value):
        self.share.set_share(value.share)

    def _coerce(self, other):
        if not isinstance(other, secpoly):
            other = secpoly(other, self.sectype)
        elif self.sectype != other.sectype:
            raise TypeError('inconsistent sectypes')

        return other

    def __neg__(self):
        return secpoly(-self.share)

    def __pos__(self):
        return secpoly(self.share)  # TODO: +self.share doesn't work, copy strategy ...

    @staticmethod
    def _add(a, b):
        if len(a) == len(b):  # fast path
            return a + b

        if len(a) < len(b):
            a, b = b, a
        return np.concatenate((a[:len(b)] + b, a[len(b):]))

    def __add__(self, other):
        """Add polynomials of secret degree."""
        other = self._coerce(other)
        return secpoly(secpoly._add(self.share, other.share))

    __radd__ = __add__

    @staticmethod
    def add(a, b):
        """Add polynomials a and b of secret degree."""
        return secpoly(secpoly._add(a.share, b.share))

    @staticmethod
    def _sub(a, b):
        m = len(a)
        n = len(b)
        if m == n:  # fast path
            c = a - b
        elif m > n:
            c = np.concatenate((a[:n] - b, a[n:]))
        else:
            b = -b
            c = np.concatenate((a + b[:m], b[m:]))
        return c

    def __sub__(self, other):
        other = self._coerce(other)
        return secpoly(secpoly._sub(self.share, other.share))

    def __rsub__(self, other):
        other = self._coerce(other)
        return secpoly(secpoly._sub(other.share, self.share))

    @staticmethod
    def sub(a, b):
        """Subtract polynomials a and b of secret degree."""
        return secpoly(secpoly._sub(a.share, b.share))

    @staticmethod
    def _mul(a, b):
        if len(a) == 0 or len(b) == 0:
            return type(a)(np.array([], dtype=int))

        return runtime.np_convolve(a, b)

    def __mul__(self, other):
        other = self._coerce(other)
        return secpoly(secpoly._mul(self.share, other.share))

    __rmul__ = __mul__

    @staticmethod
    def mul(a, b):
        """Multiply polynomials a and b of secret degree."""
        return secpoly(secpoly._mul(a.share, b.share))

    @staticmethod
    def if_else(c, a, b):
        """Secure selection based on binary condition c between polynomials a and b.

        Condition c must be of a secure number type compatible with a and b
        and its value should be 0 or 1.
        """
        # TODO: make this general
        a = a.share
        b = b.share
        if len(a) == len(b):
            # fast path
            d = np.where(c, a, b)
        else:
            d = secpoly._add(c * secpoly._sub(a, b), b)
        return secpoly(d)

    @staticmethod
    def _powmod(a, n, modulus=None):
        if n == 0:
            return a.sectype.array(np.array([1]))

        if n < 0:
            if modulus is None:
                raise ValueError('negative exponent')

            a = secpoly._invert(a, modulus)
            n = -n
        c = a
        for i in range(n.bit_length() - 2, -1, -1):
            c = secpoly._mul(c, c)
            c = secpoly._mod(c, modulus)
            if (n >> i) & 1:
                c = secpoly._mul(c, a)
                c = secpoly._mod(c, modulus)
        return c

    @staticmethod
    def powmod(a, n, b):
        """Polynomial a to the power of n modulo polynomial b, for nonzero b.

        Public n, for now.
        """
        return secpoly(secpoly._powmod(a.share, n, modulus=b.share))

    def __pow__(self, other):
        return secpoly(secpoly._powmod(self.share, other))

    @staticmethod
    def _invert(a, b):
        _, u, _ = secpoly._gcdext(a, b)
        return u

    @staticmethod
    def invert(a, b):
        """"Inverse of polynomial a modulo b.

        Inverse is assumed to exist.
        """
        _, u, _ = secpoly.gcdext(a, b)
        return u

    def __getitem__(self, key):  # NB: no set_item to prevent mutability
        # TODO: more advanced indexing, see also mpyc.gfpx
        if not isinstance(key, int):
            # TODO: consider use of key.__index__() for more generality
            raise IndexError('use int for indexing secure polynomials')

        if key < 0:
            raise IndexError('negative index not allowed for secure polynomials')

        try:
            v = self.share[key]
        except IndexError:
            v = self.sectype(0)
        return v

    @staticmethod
    def _degree(a):
        if not a:  # TODO: fix np_find (and find) for empty a, using secure type of a
            return type(a).sectype(-1)

        assert len(a) <= type(a).sectype.field.modulus  # NB: len(a) assumed sufficiently small
        # TODO: add use of alternative representation(s) for _degree(a)
        # (e.g., using a binary or unary encoding with bits of type sectype, or using a secint)
        return len(a) - 1 - runtime.np_find(np.flip(a) == 0, 0, bits=True)

    def degree(self):
        """Degree of polynomial.

        Degree of zero polynomial is -1.
        """
        return secpoly._degree(self.share)

    @staticmethod
    def _monic(a, lc_pinv=False):
        """Monic version of polynomial.

        If lc_pinv is set, inverse of leading coefficient is also returned (0 for zero polynomial).
        """
        if not a:  # TODO: fix np_find (and find) for empty a, using secure type of a
            return a

        d = secpoly._degree(a)  # set degree obliviously
        n = len(a)
        x = runtime.unit_vector(d+1, n+1)  # NB: if d=-1, value for x irrelevant, as a=0 already
        x = runtime.np_fromlist(x)
        lc1 = 1 / (x @ runtime.np_concatenate((np.zeros(1, dtype=int), a)))  # TODO: avoid use of
        # check=True for np.zeros() in FiniteFieldArray.__init__(), also in other cases throughout
        a *= lc1
        if lc_pinv:
            lc1 = runtime.if_else(d == -1, type(d)(0), lc1)  # pseudoinverse
            return a, lc1

        return a

    def monic(self):
        """Monic version of polynomial

        Zero polynomial remains unchanged.
        """
        return secpoly(secpoly._monic(self.share))

    @staticmethod
    def _reverse(a, d=None):
        n = len(a)
        if isinstance(d, int):  # d>=-1
            if d < -1:
                raise ValueError('degree d must be at least -1')

            if d+1 < n:
                a = runtime.np_getitem(a, slice(d+1))  # truncate
                a = runtime.np_flip(a)
            elif d+1 > n:
                a = runtime.np_flip(a)
                a = runtime.np_concatenate((np.zeros(d+1 - n, dtype=int), a))  # pad with zeros
            else:
                a = runtime.np_flip(a)
            return a

        if not n:
            return a

        if d is None:
            d = secpoly._degree(a)  # set degree obliviously
        else:
            #  assuming -1 <= d <= n-1
            if not isinstance(d, type(a).sectype):
                d = runtime.convert(d, type(a).sectype)
            x = runtime.unit_vector(d+1, n+1)[1:]   # 0 <= d+1 < n+1
            x = runtime.np_fromlist(x)
            x = np.flip(np.cumsum(np.flip(x)))
            a *= x  # truncate

        # TODO: check alternative using np.roll with secret offset:
        # return = runtime.np_flip(runtime.np_roll(a, n-1 - d))
        x = runtime.unit_vector(d, n)  # NB: if d=-1, value for x irrelevant, as a=0 already
        x = runtime.np_fromlist(x)
        # rotate and flip in one go
        return np.stack(tuple(np.roll(x, -i) for i in range(n))) @ a

    def reverse(self, d=None):
        """Reverse of polynomial by specified degree d.

        Basically, coefficients are put in reverse order. For example,
        reverse of x + 2x^2 + 3x^3 is 3 + 2x + x^2.

        If d is None (default), d is set to the (secret) degree of the given poynomial.
        Otherwise, the given polynomial is first padded with zeros or truncated
        to attain the given degree d, d>=-1, before it is reversed.
        If d is secret, then -1 <= d <= len(a) -1 is assumed, where a is the
        secure array holding the secret-shared coefficients.
        """
        return secpoly(secpoly._reverse(self.share, d=d))

    def truncate(self, n):
        """Truncate polynomial modulo X^n, for nonnegative n."""
        return secpoly(self.share[:n])

    @mpc_coro
    @staticmethod
    async def _div(a, b):  # assume b != 0
        m, n = len(a), len(b)
        if not m:
            return a

        assert m > 0 and n > 0
        stype = type(a)
        await runtime.returnType((stype, (m,)))

        field = stype.sectype.field
        poly = GFpX(field.modulus)

        degb = secpoly._degree(b)
        a = np.flip(a)
        b = np.flip(b)
        b = np.roll(b, degb + 1)

        field_relative_size = field.order.bit_length() // runtime.options.sec_param
        if field_relative_size < 2:  # small and medium-sized fields
            while await runtime.is_zero_public((s0 := runtime._np_randoms(stype.sectype, 1))[0]):
                pass  # ensure s0 nonzero
            s = runtime._np_randoms(stype.sectype, m - 1)
            s = runtime.np_concatenate((s0, s))
        else:
            s = runtime._np_randoms(stype.sectype, m)
        u = secpoly._mul(b, s)[:m]  # RSR over (F[X]/(X^m))^* = {f in F[X] | deg f < m and f0!=0}
        # TODO: truncated [:m] secpoly multiplication (modulo X^m) to avoid resharing unused terms
        u = await runtime.output(u)
        assert u[0] != 0
        u = poly._from_list(u.value.tolist())  # NB: remove any trailing zeros
        u = poly._invert(u, [0]*m + [1])  # invert u modulo X^m
        u = field.array(np.array(u))
        v = secpoly._mul(s, u)[:m]
        q = secpoly._mul(a, v)[:m]
        l = max(m, n).bit_length() + field.order.bit_length()  # TODO: check bit length l
        degb = runtime.convert(degb, runtime.SecInt(l=l))
        d = runtime.max(m - degb, 0) - 1
        d = runtime.convert(d, b.sectype)
        q = secpoly._reverse(q, d)
        return q

    @staticmethod
    def _mod(a, b):  # assume b != 0
        if b is None:  # see _powmod()
            return a  # NB: in-place

        q = secpoly._div(a, b)
        return secpoly._sub(a, secpoly._mul(q, b))[:len(b) - 1]

    def __floordiv__(self, other):
        other = self._coerce(other)
        return secpoly(secpoly._div(self.share, other.share))

    def __rfloordiv__(self, other):
        other = self._coerce(other)
        return secpoly(secpoly._div(other.share, self.share))

    def __mod__(self, other):
        other = self._coerce(other)
        a, b = self.share, other.share
        q = secpoly._div(a, b)
        return secpoly(secpoly._sub(a, secpoly._mul(q, b))[:len(b) - 1])

    @staticmethod
    def mod(a, b):
        """Reduce polynomial a modulo polynomial b, for nonzero b."""
        return secpoly(secpoly._mod(a.share, b.share))

    def __rmod__(self, other):
        other = self._coerce(other)
        a, b = other.share, self.share
        q = secpoly._div(a, b)
        return secpoly(secpoly._sub(a, secpoly._mul(q, b))[:len(b) - 1])

    def __divmod__(self, other):
        other = self._coerce(other)
        a, b = self.share, other.share
        q = secpoly._div(a, b)
        return secpoly(q), secpoly(secpoly._sub(a, secpoly._mul(q, b))[:len(b) - 1])

    def __rdivmod__(self, other):
        other = self._coerce(other)
        a, b = other.share, self.share
        q = secpoly._div(a, b)
        return secpoly(q), secpoly(secpoly._sub(a, secpoly._mul(q, b))[:len(b) - 1])

    @staticmethod
    def _lshift(a, n):
        if not a:
            a = np.copy(a)
        else:
            a = np.concatenate((np.zeros(n, dtype=int), a))
        return a

    def __lshift__(self, n):
        """Multiply polynomial by X^n."""
        return secpoly(secpoly._lshift(self.share, n))

    @staticmethod
    def _rshift(a, n):
        return a[n:]

    def __rshift__(self, n):
        """Quotient of polynomial divided by X^n."""
        return secpoly(secpoly._rshift(self.share, n))

    @staticmethod
    def _gcpx(a, b):
        """Secure greatest common power of x dividing a and b."""
        x = a != 0
        y = b != 0
        z = x + y - x * y  # bitwise or
        _, f_i = runtime.np_find(z, 1, e=None)
        # TODO: consider keeping f_i in number range if z contains no 1, e.g., setting e='len(x)-1'
        return f_i

    @staticmethod
    def _gcd(a, b):
        if len(a) < len(b):
            a, b = b, a
        n = len(a)
        if not n:
            return a

        if n > len(b):
            b = np.concatenate((b, np.zeros(n - len(b), dtype=int)))
        # len(a)=len(b)=n ensured
        e = secpoly._gcpx(a, b)
        f = np.roll(a, n - e)
        g = np.roll(b, n - e)  # TODO: combine rolls ?
        c = f[0] == 0
        f, g = np.where(c, g, f), np.where(c, f, g)  # TODO: combine wheres ?
        # f[0] != 0 ensured unless f=g=0
        stype = type(a)
        field = stype.sectype.field
        l = 1 + max(n.bit_length(), field.modulus.bit_length())  # TODO: check bit length l
        secint = runtime.SecInt(l=l)
        delta = secint(1)
        d = n  # deg(f)<=n-1 and deg(g)<=n-1<n
        for i in range(2*d - 1):  # NB: 2d-1 steps suffice provided deg(f)<=d and deg(g)<d
            l = (i+1).bit_length()  # TODO: check use of bit length of i+1
            delta_gt0 = 1 - runtime.sgn((delta-1-(i%2))/2, l=l, LT=True)
            _delta_gt0 = runtime.convert(delta_gt0, stype.sectype)
            g_0 = g[0] != 0
            _g_0 = runtime.convert(g_0, secint)
            c = _delta_gt0 * g_0
            d = c * secpoly._sub(g, f)
            f, g = secpoly._add(f, d), secpoly._sub(g, d)
            delta *= (1 - 2 * delta_gt0 * _g_0)
            g = f[0]*g - g[0]*f  # ensure g[0]=0
            g = g[1:]
            delta += 1
            if not g:
                break
        f = secpoly._monic(f)
        f = np.roll(f, e)
        return f

    @staticmethod
    def _divstepsx2(n, a, b):
        assert n >= 0
        stype = type(a)
        field = stype.sectype.field
        f, g = a, b
        a, b = (alpha := 1/a[0]) * a, alpha * b  # ensure a[0]=1
        u = r = stype(np.array([1]))
        v = q = stype(np.array([]))
        l = 1 + max(n.bit_length(), field.modulus.bit_length())  # TODO: check bit length l
        secint = runtime.SecInt(l=l)
        delta = secint(1)
        for i in range(n):
            if not g:
                continue
            l = (i+1).bit_length()  # TODO: check use of bit length of i+1
            delta_gt0 = 1 - runtime.sgn((delta-1-(i%2))/2, l=l, LT=True)
            _delta_gt0 = runtime.convert(delta_gt0, stype.sectype)
            g_0 = g[0] != 0
            _g_0 = runtime.convert(g_0, secint)
            c = _delta_gt0 * g_0
            d = c * secpoly._sub(g, f)
            f, g = secpoly._add(f, d), secpoly._sub(g, d)
            d = c * secpoly._sub(q, u)
            u, q = secpoly._add(u, d), secpoly._sub(q, d)
            d = c * secpoly._sub(r, v)
            v, r = secpoly._add(v, d), secpoly._sub(r, d)
            delta *= (1 - 2 * delta_gt0 * _g_0)
            f0, g0 = f[0], g[0]
            g = f0*g - g0*f  # ensure g[0]=0
            q = f0*q - g0*u
            r = f0*r - g0*v
            r0 = r[0]
            r = secpoly._sub(r, r0 * a)  # ensure r[0]=0
            q = secpoly._add(q, r0 * b)  # ensure q[0]=0
            g = g[1:]
            r = r[1:]
            q = q[1:]
            delta += 1
        return delta, f, g, (u, v, q, r)

    @staticmethod
    def _gcdext(a, b):
        m, n = len(a), len(b)
        if m < n:
            a = np.concatenate((a, np.zeros(n - m, dtype=int)))
        elif n < m:
            b = np.concatenate((b, np.zeros(m - n, dtype=int)))
        n = len(a)  # TODO: check case n=0, maybe return a if so
        # len(a)=len(b)=n ensured
        assert len(a) == len(b)
        e = secpoly._gcpx(a, b)
        f = np.roll(a, n - e)
        g = np.roll(b, n - e)  # TODO: combine rolls ?
        c = f[0] == 0
        f, g = np.where(c, g, f), np.where(c, f, g)  # TODO: combine wheres ?
        # f[0] != 0 ensured unless f=g=0
        d = n  # see secpoly._gcd()
        delta, f, g, (u, v, _, _) = secpoly._divstepsx2(2*d-1, f, g)
        f, lc1 = secpoly._monic(f, lc_pinv=True)
        f = np.roll(f, e)
        u = u * lc1
        v = v * lc1
        u, v = np.where(c, v, u), np.where(c, u, v)  # TODO: combine wheres ?
        return f, u, v

    @staticmethod
    def gcd(a, b):
        """Greatest common divisor of polynomials a and b."""
        return secpoly(secpoly._gcd(a.share, b.share))

    @staticmethod
    def gcdext(a, b):
        """Extended GCD of polynomials a and b."""
        f, u, v = secpoly._gcdext(a.share, b.share)
        return secpoly(f), secpoly(u), secpoly(v)

    @staticmethod
    def is_irreducible(a):
        """Test polynomial a for irreducibility."""
        # NB: constant polynomials are not irreducible
        D = len(a.share) - 1  # maximum degree
        if D <= 0:
            return a.sectype(0)

        p = a.sectype.field.modulus
        poly = GFpX(p)
        X = secpoly(poly('x'))
        b = X
        c = [secpoly.gcd((b := secpoly.powmod(b, p, a)) - X, a) for _ in range(D // 2)]
        c = reduce(operator.mul, c, poly(1))
        d = a.degree()
        return (d != -1) * (d != 0) * (c == poly(1))

    def copy(self):
        return secpoly(self.share.copy())

    @staticmethod
    def _lt(a, b):
        """Lexicographic less-than comparison."""
        # NB: coefficients are considered as unsigned field elements
        d = secpoly._degree(secpoly._sub(a, b))
        d = runtime.unit_vector(d+1, max(len(a), len(b))+1)[1:]
        d = runtime.np_fromlist(d)  # TODO: np_unit_vector() for GF(p)
        a = a @ d[:len(a)]
        b = b @ d[:len(b)]
        stype = type(d).sectype
        secint = runtime.SecInt(l=stype.field.order.bit_length()+2)
        a, b = runtime.convert([a, b], secint)
        c = a < b
        c = runtime.convert(c, stype)
        return c

    def __lt__(self, other):
        """Secure strictly less-than comparison."""
        other = self._coerce(other)
        return secpoly._lt(self.share, other.share)

    def __le__(self, other):
        """Secure less-than or equal comparison."""
        other = self._coerce(other)
        return 1 - secpoly._lt(other.share, self.share)

    def __eq__(self, other):
        """Secure equality test."""
        other = self._coerce(other)
        return np.all((self - other).share == 0, axis=0)

    def __ge__(self, other):
        """Secure greater-than or equal comparison."""
        other = self._coerce(other)
        return 1 - secpoly._lt(self.share, other.share)

    def __gt__(self, other):
        """Secure strictly greater-than comparison."""
        other = self._coerce(other)
        return secpoly._lt(other.share, self.share)

    def __ne__(self, other):
        """Secure negated equality test."""
        other = self._coerce(other)
        return np.any((self - other).share != 0, axis=0)

    def __call__(self, x):
        """Evaluate polynomial at given x."""
        n = len(self.share)
        if isinstance(x, SecureObject):
            a = runtime.np_fromlist([x])
        else:
            a = np.array([x])
        y = np.vander(a, n, increasing=True)[0]
        return self.share @ y

    @staticmethod
    def _input(x, senders):
        """Input a list x of secure polynomials. See runtime.input()."""
        shares = [map(secpoly, runtime.input(a.share, senders)) for a in x]
        shares = list(map(list, zip(*shares)))
        return shares

    @staticmethod
    async def _output(x, receivers, threshold):
        """Output a list x of secure polynomials. See runtime.output()."""
        stype = x[0].sectype
        poly = GFpX(stype.field.modulus)
        y = await runtime.gather([runtime.output(a.share, receivers, threshold) for a in x])
        if y[0] is None:
            return y

        y = [poly(poly._from_list(a.value.tolist()), check=False) for a in y]
        return y

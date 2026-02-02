"""This module supports several types of finite groups.

A finite group is a set of group elements together with a group operation.
The group operation is a binary operation, usually written multiplicatively
(optionally, the group operation can be written additively).

The default Python operators to manipulate group elements are the (binary)
operator @ for the group operation, the (unary) operator ~ for inversion of
group elements, and the (binary) operator ^ for repeated application of
the group operation. The alternative Python operators used for additive and
multiplicative notation are:

    - default:         a @ b,    ~a,    a^n    (a^-1 = ~a)
    - additive:        a + b,    -a,    n*a    (-1*a = -a)
    - multiplicative:  a * b,   1/a,    a**n   (a**-1 = 1/a)

for arbitrary group elements a, b, and integer n.

Six types of groups are currently supported, aimed mainly at applications
in cryptography:

    - symmetric groups of any degree n (n>=0)
    - quadratic residue groups modulo a safe prime
    - Schnorr groups (prime-order subgroups of the multiplicative group of a finite field)
    - elliptic curve groups (Edwards curves, a Koblitz curve, and Barreto-Naehrig curves)
    - hyperelliptic curve groups, mainly of genus 2 and 3
    - class groups of imaginary quadratic fields

The structure of most of these groups will be trivial, preferably cyclic or even
of prime order. Where applicable, a generator of the group (or a sufficiently
large subgroup) is provided to accommodate discrete log and Diffie-Hellman
hardness assumptions.
"""

import math
import decimal
import random
import functools
from mpyc.gmpy import powmod, gcdext, is_prime, next_prime, prev_prime, legendre, isqrt, iroot
from mpyc.gfpx import GFpX, Polynomial
from mpyc.finfields import GF, find_prime_root


class FiniteGroupElement:
    """Abstract base class for finite groups.

    Overview Python operators for group operation, inverse, and repeated operation:

        - default notation: @, ~, ^ (matmul, invert, xor).
        - additive notation: +, -, * (add, sub, mul)
        - multiplicative notation: *, 1/ (or, **-1), ** (mul, truediv (or, pow), pow)
    """

    __slots__ = 'value'
    value: object  # for detection by pylint

    order = None
    is_additive = False
    is_multiplicative = False
    identity = None
    is_abelian = None
    is_cyclic = None
    generator = None  # generates large subgroup, preferably the entire group

    def __matmul__(self, other):  # overload @
        group = type(self)
        if self is other:
            return group.operation2(self)

        if isinstance(other, group):
            return group.operation(self, other)

        return NotImplemented

    def __invert__(self):  # overload ~
        return self.inversion(self)

    def __xor__(self, other):  # overload ^
        if isinstance(other, int):
            return self.repeat(self, other)

        return NotImplemented

    def __add__(self, other):
        group = type(self)
        if not group.is_additive:
            raise TypeError('group not additive')

        return group.__matmul__(self, other)

    def __neg__(self):
        group = type(self)
        if not group.is_additive:
            raise TypeError('group not additive')

        return group.__invert__(self)

    def __sub__(self, other):
        group = type(self)
        if not group.is_additive:
            raise TypeError('group not additive')

        return group.__matmul__(self, group.__invert__(other))

    def __mul__(self, other):
        group = type(self)
        if group.is_multiplicative:
            return group.__matmul__(self, other)

        if group.is_additive:
            return NotImplemented

        raise TypeError('* not defined for group')

    def __rmul__(self, other):
        group = type(self)
        if group.is_multiplicative:
            if group.is_abelian:
                return group.__matmul__(self, other)

            return group.__matmul__(group(other), self)

        if group.is_additive:
            return group.__xor__(self, other)

        raise TypeError('* not defined for group')

    def __truediv__(self, other):
        group = type(self)
        if not group.is_multiplicative:
            raise TypeError('group not multiplicative')

        return group.__matmul__(self, group.__invert__(other))

    def __rtruediv__(self, other):
        group = type(self)
        if not group.is_multiplicative:
            raise TypeError('group not multiplicative')

        if other != 1:
            raise TypeError('only 1/. supported')

        return group.__invert__(self)

    def __pow__(self, other):
        group = type(self)
        if not group.is_multiplicative:
            raise TypeError('group not multiplicative')

        return group.__xor__(self, other)

    def __eq__(self, other):
        group = type(self)
        if not isinstance(other, group):
            return NotImplemented

        return group.equality(self, other)

    def __hash__(self):
        """Make finite group elements hashable (e.g., for LRU caching)."""
        return hash((type(self).__name__, self.value))

    def __repr__(self):
        return repr(self.value)

    @classmethod
    def operation(cls, a, b, /):
        """Return a @ b."""
        raise NotImplementedError

    @classmethod
    def operation2(cls, a, /):
        """Return a @ a."""
        return cls.operation(a, a)

    @classmethod
    def inversion(cls, a, /):
        """Return @-inverse of a (written ~a)."""
        raise NotImplementedError

    def inverse(self):
        """For convenience."""
        return self.inversion(self)

    @classmethod
    def equality(cls, a, b, /):
        """Return a == b."""
        raise NotImplementedError

    @staticmethod
    def repeat(a, n):
        """Return nth @-power of a (written a^n), for any integer n."""
        cls = type(a)
        if n == 0:
            return cls.identity

        if n < 0:
            a = cls.inversion(a)
            n = -n
        c = a
        for i in range(n.bit_length() - 2, -1, -1):
            c = cls.operation2(c)
            if (n >> i) & 1:
                c = cls.operation(c, a)
        return c


class SymmetricGroupElement(FiniteGroupElement):
    """Common base class for symmetric groups.

    Symmetric groups contains all permutations of a fixed length (degree).
    Permutations of {0,...,n-1} represented as length-n tuples with unique
    entries in {0,...,n-1}, n>=0.
    """

    __slots__ = ()

    degree = None

    def __init__(self, value=None, check=True):
        if value is None:  # default to identity element
            value = tuple(range(self.degree))
        elif isinstance(value, list):
            value = tuple(value)
        if check:
            if len(value) != self.degree or set(value) != set(range(self.degree)):
                raise ValueError(f'valid length-{self.degree} permutation required')

        self.value = value

    @classmethod
    def operation(cls, p, q, /):
        """First p then q."""
        return cls(tuple(q.value[j] for j in p.value), check=False)

    @classmethod
    def inversion(cls, p, /):
        n = len(p.value)
        q = [None] * n
        for i in range(n):
            q[p.value[i]] = i
        return cls(tuple(q), check=False)

    @classmethod
    def equality(cls, p, q, /):
        return p.value == q.value


@functools.cache
def SymmetricGroup(n):
    """Create type for symmetric group of degree n, n>=0."""
    name = f'Sym({n})'
    Sym = type(name, (SymmetricGroupElement,), {'__slots__': ()})
    Sym.degree = n
    Sym.order = math.factorial(n)
    Sym.is_abelian = n <= 2
    Sym.is_cyclic = n <= 2  # 2-elt generating set (1, 0, 2, ..., n-1) plus (1, ..., n-1, 0)
    Sym.identity = Sym()
    globals()[name] = Sym  # NB: exploit (almost?) unique name dynamic Sym type
    return Sym


class QuadraticResidue(FiniteGroupElement):
    """Common base class for groups of quadratic residues modulo an odd prime.

    Quadratic residues modulo p represented by GF(p)* elements.
    """

    __slots__ = ()

    is_multiplicative = True
    is_abelian = True
    is_cyclic = True
    field: type  # multiplicative group of this prime field is used
    gap = None

    def __init__(self, value=1, check=True):
        if check:
            if not isinstance(value, self.field):
                if isinstance(value, int):
                    value = self.field(value)
                else:
                    raise TypeError('int or prime field element required')

            if value == 0 or not value.is_sqr():
                raise ValueError('quadratic residue required')

        self.value = value

    @classmethod
    def operation(cls, a, b, /):
        return cls(a.value * b.value, check=False)

    @classmethod
    def inversion(cls, a, /):
        return cls(1/a.value, check=False)

    @classmethod
    def equality(cls, a, b, /):
        return a.value == b.value

    @classmethod
    def repeat(cls, a, n):
        return cls(a.value**n, check=False)

    def __int__(self):
        return int(self.value)

    @classmethod
    def encode(cls, m):
        """Encode message m in a quadratic residue."""
        gap = cls.gap
        field = cls.field
        modulus = field.modulus
        for i in range(1, gap):
            if legendre(i, modulus) == 1:
                a = m * gap + i
                if legendre(a, modulus) == 1:
                    M = cls(field(a), check=False)
                    Z = cls(field(i), check=False)
                    return M, Z

        raise ValueError('message encoding failed, try larger gap')

    @classmethod
    def decode(cls, M, Z):
        """Decode message from given group elements."""
        gap = cls.gap
        return int((M.value - Z.value) / gap)


def _find_safe_prime(l):
    """Find safe prime p of bit length l, l>=2.

    Hence, q=(p-1)/2 is also prime (except when l=2 and p=3).
    It is also ensured that p=3 (mod 4), hence p is a Blum prime.
    """
    IKE_options_l_k = {768: 149686, 1024: 129093, 1536: 741804, 2048: 124476,
                       3072: 1690314, 4096: 240904, 6144: 929484, 8192: 4743158}
    if l in IKE_options_l_k:
        # Compute pi to the required precision.
        decimal.setcontext(decimal.Context(prec=round(l / math.log2(10))))
        # See https://docs.python.org/3/library/decimal.html for following recipe:
        decimal.getcontext().prec += 2  # extra digits for intermediate steps
        three = decimal.Decimal(3)
        lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na+8
            d, da = d + da, da+32
            t = (t * n) / d
            s += t
        decimal.getcontext().prec -= 2
        pi_l = +s  # NB: unary plus applies the new precision

        # Following https://kivinen.iki.fi/primes to compute IKE prime p:
        k = IKE_options_l_k[l]
        fixedbits = 64
        epi = math.floor(pi_l * 2**(l - 2*fixedbits - 2)) + k
        p = 2**l - 2**(l - fixedbits) - 1 + epi * 2**fixedbits
    else:
        if l == 2:
            p = 3
        else:
            q = prev_prime(1 << l-1)
            while not is_prime(2*q+1):
                q = prev_prime(q)
            # q is a Sophie Germain prime
            p = int(2*q + 1)
    return p


def QuadraticResidues(p=None, l=None):
    """Create type for quadratic residues group given (bit length l of) odd prime modulus p.

    The group of quadratic residues modulo p is of order n=(p-1)/2.
    Given bit length l>2, p will be chosen such that n is also an odd prime.
    If l=2, the only possibility is p=3, hence n=1.
    """
    if l is not None:
        if p is None:
            p = _find_safe_prime(l)
    elif p is None:
        p = 3
    if p%2 == 0:
        raise ValueError('odd prime modulus required')

    return _QuadraticResidues(p)


@functools.cache
def _QuadraticResidues(p):
    field = GF(p)  # raises if p is not prime
    g = 2
    while legendre(g, p) != 1:
        g += 1
    # g is generator if p is a safe prime

    l = p.bit_length()
    name = f'QR{l}({p})'
    QR = type(name, (QuadraticResidue,), {'__slots__': ()})
    QR.field = field
    QR.gap = 128  # TODO: calculate gap as a function of bit length of p
    QR.order = p >> 1
    QR.identity = QR()
    QR.generator = QR(g)
    globals()[name] = QR  # NB: exploit (almost?) unique name dynamic QR type
    return QR


class SchnorrGroupElement(FiniteGroupElement):
    """Common base class for prime-order subgroups of the multiplicative group of a finite field."""
    # TODO: consider merging with QuadraticResidues class

    __slots__ = ()

    is_multiplicative = True
    is_abelian = True
    is_cyclic = True
    field: type  # multiplicative group of this (prime) field is used

    def __init__(self, value=1, check=True):
        if check:
            if not isinstance(value, self.field):
                if isinstance(value, int):
                    value = self.field(value)
                else:
                    raise TypeError('int or prime field element required')

            if value**self.order != 1:
                raise ValueError('subgroup elt required')

        self.value = value

    @classmethod
    def operation(cls, a, b, /):
        return cls(a.value * b.value, check=False)

    @classmethod
    def inversion(cls, a, /):
        return cls(1/a.value, check=False)

    @classmethod
    def equality(cls, a, b, /):
        return a.value == b.value

    @classmethod
    def repeat(cls, a, n):
        return cls(a.value**n, check=False)

    def __int__(self):
        return int(self.value)

    @classmethod
    def encode(cls, m):
        """Encode message m in group element g^m."""
        g = cls.generator
        return cls(g.value**m, check=False), g  # g as dummy Z

    @classmethod
    def decode(cls, M, Z):
        """Decode message from given group element."""
        g = cls.generator
        h = cls.identity
        for m in range(1024):  # TODO: get rid of hard-coded 1024 bound (also in secgroups module)
            if h != M:
                h = cls.operation(g, h)
            else:
                break
        return m


def SchnorrGroup(p=None, q=None, g=None, l=None, n=None):
    """Create type for Schnorr group of odd prime order q.

    If q is not given, q will be the largest n-bit prime, n>=2.
    If p is not given, p will be the least l-bit prime, l>n, such that q divides p-1.

    If l and/or n are not given, default bit lengths will be set (2<=n<l).
    """
    n_l = ((160, 1024), (192, 1536), (224, 2048), (256, 3072), (384, 7680))
    if p is None:
        if q is None:
            if n is None:
                if l is None:
                    l = 2048
                n = next((n for n, _ in n_l if _ >= l), 512)
            q = prev_prime(1 << n)
        else:
            if n is None:
                n = q.bit_length()
            assert q%2 and is_prime(q)
        if l is None:
            l = next((l for _, l in n_l if _ >= n), 15360)

        # n-bit prime q
        w = (1 << l-2) // q + 1  # w*q >= 2^(l-2), so p = 2*w*q + 1 > 2^(l-1)
        p = 2*w*q + 1
        while not is_prime(p):
            p += 2*q
        # p < 2^l provided gap between n and l is sufficiently large
    else:
        assert q is not None  # if p is given, q must be given as well
        assert (p - 1) % q == 0
        assert q%2 and is_prime(q)
        assert is_prime(p)
        if l is None:
            l = p.bit_length()
        if n is None:
            n = q.bit_length()
    assert l == p.bit_length()
    assert n == q.bit_length()

    p = int(p)
    q = int(q)
    if g is None:
        w = (p-1) // q
        i = 2
        while (g := powmod(i, w, p)) == 1:
            i += 1
        g = int(g)
    return _SchnorrGroup(p, q, g)


@functools.cache
def _SchnorrGroup(p, q, g):
    field = GF(p)  # raises if modulus is not prime
    l = p.bit_length()
    n = q.bit_length()
    name = f'SG{l}:{n}({p}:{q})'
    SG = type(name, (SchnorrGroupElement,), {'__slots__': ()})
    SG.field = field
    SG.order = q
    SG.identity = SG()
    SG.generator = SG(g)
    globals()[name] = SG  # NB: exploit (almost?) unique name dynamic SG type
    return SG


class EllipticCurvePoint(FiniteGroupElement):
    """Common base class for elliptic curve groups."""

    __slots__ = ()

    is_additive = True
    is_multiplicative = False
    is_abelian = True
    oblivious = None  # set oblivious=True if arithmetic works in MPC setting
    field: type  # elliptic curve is defined over this field
    _identity = None
    gap = None

    def __getitem__(self, key):  # NB: no __setitem__ to prevent mutability
        return self.value[key]

    # TODO: reconsider use of properties x, y, z (and t) ... currently used for pairings
    @property
    def x(self):
        return self.value[0]

    @property
    def y(self):
        return self.value[1]

    @property
    def z(self):
        return self.value[2]

    @classmethod
    def ysquared(cls, x):
        """Return value of y^2 as a function of x, for a point (x, y) on the curve."""
        raise NotImplementedError

    def normalize(self):
        """Convert to unique affine representation."""
        raise NotImplementedError

    @classmethod
    def encode(cls, m):
        """Encode message m in x-coordinate of a point on the curve."""
        field = cls.field  # TODO: extend this to non-prime fields
        gap = cls.gap
        modulus = field.modulus
        for i in range(gap):
            x_0 = field(i)
            ysquared_0 = cls.ysquared(x_0)
            if legendre(int(ysquared_0), modulus) == 1:
                x_m = field(m * gap + i)
                ysquared_m = cls.ysquared(x_m)
                if legendre(int(ysquared_m), modulus) == 1:
                    M = cls((x_m, ysquared_m.sqrt()), check=False)
                    Z = cls((x_0, ysquared_0.sqrt()), check=False)
                    return M, Z

        raise ValueError('message encoding failed, try larger gap')

    @classmethod
    def decode(cls, M, Z):
        """Decode message from given group elements."""
        gap = cls.gap
        return int((M.normalize()[0] - Z.normalize()[0]) / gap)


class EdwardsCurvePoint(EllipticCurvePoint):
    """Common base class for (twisted) Edwards curves."""
    # pylint: disable=W0223 (abstract-method)

    __slots__ = ()

    a = None
    d = None

    @classmethod
    def ysquared(cls, x):
        x2 = x**2
        return (1 - cls.a * x2) / (1 - cls.d * x2)

    def __init__(self, value=None, check=True):
        field = self.field
        if value is None:
            value = map(field, self._identity)
        elif 2 == len(value) < len(self._identity):  # convert affine to target
            value = list(value) + [field(1)]  # z = 1
            if len(value) < len(self._identity):
                value += [value[0] * value[1]]  # t = x * y
        if check:
            value = list(value)
            for i in range(len(value)):
                if not isinstance(value[i], field):
                    value[i] = field(value[i])
            x, y = value[:2]
            z = value[2] if value[2:] else field(1)
            x, y = x / z, y / z
            if value[3:]:
                t = value[3] / z
                if t != x * y:
                    raise ValueError('incorrect extended coordinate')

            if y**2 != self.ysquared(x):
                raise ValueError('point not on curve')

        self.value = tuple(value)


class EdwardsAffine(EdwardsCurvePoint):
    """Edwards curves with affine coordinates."""

    __slots__ = ()

    _identity = (0, 1)
    oblivious = True

    @classmethod
    def inversion(cls, pt, /):
        x, y = pt
        return cls((-x, y), check=False)

    @classmethod
    def operation(cls, pt1, pt2, /):
        """Add Edwards points using affine coordinates (projective with z=1)."""
        # see https://hyperelliptic.org/EFD/g1p/data/edwards/projective/addition/mmadd-2007-bl
        x1, y1 = pt1
        x2, y2 = pt2
        C = x1 * x2
        D = y1 * y2
        E = cls.d * C * D
        x3 = (1 - E) * ((x1 + y1) * (x2 + y2) - C - D)
        y3 = (1 + E) * (D - cls.a * C)
        z3_inv = 1 / (1 - E**2)
        x3 = x3 * z3_inv
        y3 = y3 * z3_inv
        return cls((x3, y3), check=False)

    def normalize(self):
        return self

    @classmethod
    def equality(cls, pt1, pt2, /):
        return pt1.value == pt2.value


class EdwardsProjective(EdwardsCurvePoint):
    """Edwards curves with projective coordinates."""

    __slots__ = ()

    _identity = (0, 1, 1)
    oblivious = True

    @classmethod
    def inversion(cls, pt, /):
        x, y, z = pt
        return cls((-x, y, z), check=False)

    @classmethod
    def operation(cls, pt1, pt2, /):
        """Add Edwards points with (homogeneous) projective coordinates."""
        # see https://www.hyperelliptic.org/EFD/g1p/data/twisted/projective/addition/add-2008-bbjlp
        x1, y1, z1 = pt1
        x2, y2, z2 = pt2
        A = z1 * z2
        B = A**2
        C = x1 * x2
        D = y1 * y2
        E = cls.d * C * D
        F = B - E
        G = B + E
        x3 = A * F * ((x1 + y1) * (x2 + y2) - C - D)
        y3 = A * G * (D - cls.a * C)
        z3 = F * G
        return cls((x3, y3, z3), check=False)

    def normalize(self):
        cls = type(self)
        x, y, z = self
        z_inv = 1 / z
        x, y = x * z_inv, y * z_inv
        return cls((x, y, cls.field(1)), check=False)

    @classmethod
    def equality(cls, pt1, pt2, /):
        x1, y1, z1 = pt1
        x2, y2, z2 = pt2
        return x1 * z2 == x2 * z1 and y1 * z2 == y2 * z1


class EdwardsExtended(EdwardsCurvePoint):
    """Edwards curves with extended coordinates."""

    __slots__ = ()

    _identity = (0, 1, 1, 0)
    oblivious = True

    @classmethod
    def inversion(cls, pt, /):
        x, y, z, t = pt
        return cls((-x, y, z, -t), check=False)

    @classmethod
    def operation(cls, pt1, pt2, /):
        """Add (twisted a=-1) Edwards points in extended projective coordinates."""
        # see https://eprint.iacr.org/2008/522 Hisil et al., Section 4.2 for 4 processors
        x1, y1, z1, t1 = pt1
        x2, y2, z2, t2 = pt2
        r1, r2, r3, r4 = y1 - x1, y2 - x2, y1 + x1, y2 + x2
        r1, r2, r3, r4 = r1 * r2, r3 * r4, 2*cls.d * t1 * t2, 2 * z1 * z2
        r1, r2, r3, r4 = r2 - r1, r4 - r3, r4 + r3, r2 + r1
        pt3 = r1 * r2, r3 * r4, r2 * r3, r1 * r4
        return cls(pt3, check=False)

    @classmethod
    def operation2(cls, pt, /):
        """Doubling (twisted a=-1) Edwards point in extended projective coordinates."""
        # specialized addition for case pt1=pt2
        x, y, z, t = pt
        r1, r2, r3, r4 = (y - x)**2, (y + x)**2, 2*cls.d * t**2, 2 * z**2
        r1, r2, r3, r4 = r2 - r1, r4 - r3, r4 + r3, r2 + r1
        pt2 = r1 * r2, r3 * r4, r2 * r3, r1 * r4
        return cls(pt2, check=False)

    def normalize(self):
        cls = type(self)
        x, y, z, _ = self
        z_inv = 1 / z
        x, y = x * z_inv, y * z_inv
        return cls((x, y, cls.field(1), x * y), check=False)

    @classmethod
    def equality(cls, pt1, pt2, /):
        x1, y1, z1, _ = pt1
        x2, y2, z2, _ = pt2
        return x1 * z2 == x2 * z1 and y1 * z2 == y2 * z1


class WeierstrassCurvePoint(EllipticCurvePoint):
    """Common base class for (short) Weierstrass curves."""
    # pylint: disable=W0223 (abstract-method)

    __slots__ = ()

    a = None
    b = None

    @classmethod
    def ysquared(cls, x):
        return x**3 + cls.a * x + cls.b

    def __init__(self, value=None, check=True):
        field = self.field
        if value is None or len(value) == 0:
            value = list(map(field, self._identity))
        elif 2 == len(value) < len(self._identity):  # convert affine to target
            value = list(value) + [field(1)]  # z = 1
        if check and value:
            value = list(value)
            for i in range(len(value)):
                if not isinstance(value[i], field):
                    value[i] = field(value[i])
            x, y = value[:2]
            z = value[2] if value[2:] else field(1)
            if z != 0:
                if isinstance(self, WeierstrassJacobian):
                    x, y = x / z**2, y / z**3
                else:
                    x, y = x / z, y / z
                if y**2 != self.ysquared(x):
                    raise ValueError('point not on curve')

        self.value = tuple(value)


class WeierstrassAffine(WeierstrassCurvePoint):
    """Short Weierstrass curves with affine coordinates."""

    __slots__ = ()

    _identity = ()
    oblivious = False

    @classmethod
    def inversion(cls, pt, /):
        if pt == cls.identity:
            return pt

        x, y = pt
        return cls((x, -y), check=False)

    @classmethod
    def operation(cls, pt1, pt2, /):
        """Add Weierstrass points with affine coordinates."""
        if pt1 == cls.identity:
            return pt2

        if pt2 == cls.identity:
            return pt1

        if pt1 == pt2:
            return cls.operation2(pt1)

        x1, y1 = pt1
        x2, y2 = pt2
        if x1 == x2:
            return cls.identity  # y1 == -y2

        r = (y1 - y2) / (x1 - x2)
        x3 = r**2 - x1 - x2
        y3 = r * (x1 - x3) - y1
        return cls((x3, y3), check=False)

    @classmethod
    def operation2(cls, pt, /):
        """Double Weierstrass point with affine coordinates."""
        if pt == cls.identity:
            return cls.identity

        x, y = pt
        if y == 0:
            return cls.identity

        r = (3*x**2 + cls.a) / (2*y)
        x2 = r**2 - 2*x
        y2 = r * (x - x2) - y
        return cls((x2, y2), check=False)

    def normalize(self):
        return self

    @classmethod
    def equality(cls, pt1, pt2, /):
        return pt1.value == pt2.value


class WeierstrassProjective(WeierstrassCurvePoint):
    """Short Weierstrass curves with projective coordinates."""

    __slots__ = ()

    _identity = (0, 1, 0)
    oblivious = True

    @classmethod
    def inversion(cls, pt, /):
        x, y, z = pt
        return cls((x, -y, z), check=False)

    @classmethod
    def operation(cls, pt1, pt2, /):
        """Add Weierstrass points with projective coordinates."""
        # see https://eprint.iacr.org/2015/1060  Renes et al., Algorithm 7
        x1, y1, z1 = pt1
        x2, y2, z2 = pt2
        assert cls.a == 0
        b3 = 3*cls.b
        t0, t1, t2 = x1 * x2, y1 * y2, z1 * z2
        t3 = (x1 + y1) * (x2 + y2) - t0 - t1
        t4 = (y1 + z1) * (y2 + z2) - t1 - t2
        y3 = b3 * ((x1 + z1) * (x2 + z2) - t0 - t2)
        t0 *= 3
        t2 *= b3
        z3 = t1 + t2
        t1 -= t2
        x3 = t3 * t1 - t4 * y3
        y3 = t0 * y3 + t1 * z3
        z3 = t4 * z3 + t0 * t3
        return cls((x3, y3, z3), check=False)

    @classmethod
    def operation2(cls, pt, /):
        """Double Weierstrass point with projective coordinates."""
        # see https://eprint.iacr.org/2015/1060  Renes et al., Algorithm 9
        x, y, z = pt
        t0 = y**2
        z2 = 8*t0
        t2 = 3*cls.b * z**2
        x2 = t2 * z2
        y2 = t0 + t2
        z2 *= y * z
        t0 -= 3*t2
        y2 = t0 * y2 + x2
        x2 = 2*t0 * x * y
        return cls((x2, y2, z2), check=False)

    def normalize(self):
        cls = type(self)
        x, y, z = self
        if z == 0:
            return cls.identity

        z_inv = 1 / z
        x, y = x * z_inv, y * z_inv
        return cls((x, y, cls.field(1)), check=False)

    @classmethod
    def equality(cls, pt1, pt2, /):
        x1, y1, z1 = pt1
        x2, y2, z2 = pt2
        if z1 == 0 and z2 == 0:
            return True

        return x1 * z2 == x2 * z1 and y1 * z2 == y2 * z1


class WeierstrassJacobian(WeierstrassCurvePoint):
    """Short Weierstrass curves with Jacobian coordinates."""

    __slots__ = ()

    _identity = (0, 1, 0)
    oblivious = False

    @classmethod
    def inversion(cls, pt, /):
        x, y, z = pt
        return cls((x, -y, z), check=False)

    @classmethod
    def operation(cls, pt1, pt2, /):
        """Add Weierstrass points with Jacobian coordinates."""
        # see https://hyperelliptic.org/EFD/g1p/data/shortw/jacobian-0/addition/add-2007-bl
        if pt1[2] == 0:
            return pt2

        if pt2[2] == 0:
            return pt1

        x1, y1, z1 = pt1
        x2, y2, z2 = pt2
        z1z1 = z1**2
        z2z2 = z2**2
        u1 = x1 * z2z2
        u2 = x2 * z1z1
        s1 = y1 * z2 * z2z2
        s2 = y2 * z1 * z1z1
        h = u2 - u1
        r = 2*(s2 - s1)
        if h == 0 and r == 0:
            # pt1 == pt2
            return cls.operation2(pt1)

        i = (2*h)**2
        j = h * i
        v = u1 * i
        x3 = r**2 - j - 2*v
        y3 = r * (v - x3) - 2*s1 * j
        z3 = ((z1 + z2)**2 - z1z1 - z2z2) * h
        return cls((x3, y3, z3), check=False)

    @classmethod
    def operation2(cls, pt, /):
        """Double Weierstrass point with Jacobian coordinates."""
        # see https://hyperelliptic.org/EFD/g1p/data/shortw/jacobian-0/doubling/dbl-2009-l
        x1, y1, z1 = pt
        a = x1**2
        b = y1**2
        c = b**2
        d = 2*((x1 + b)**2 - a - c)
        e = 3*a
        f = e**2
        x2 = f - 2*d
        y2 = e * (d - x2) - 8*c
        z2 = 2*y1 * z1
        return cls((x2, y2, z2), check=False)

    def normalize(self):
        cls = type(self)
        x, y, z = self
        if z == 0:
            return cls.identity

        z_inv = 1 / z
        z_inv2 = z_inv**2
        x, y = x * z_inv2, y * z_inv * z_inv2
        return cls((x, y, cls.field(1)), check=False)

    @classmethod
    def equality(cls, pt1, pt2, /):
        x1, y1, z1 = pt1
        x2, y2, z2 = pt2
        if z1 == 0 and z2 == 0:
            return True

        z12, z22 = z1**2, z2**2
        return x1 * z22 == x2 * z12 and y1 * z2 * z22 == y2 * z1 * z12


def EllipticCurve(curvename='Ed25519', coordinates=None):
    """Create elliptic curve type for a selection of built-in curves.
    The default coordinates used with these curves are 'affine'.

    The following Edwards curves and Weierstrass curves are built-in:

        - 'Ed25519': see https://en.wikipedia.org/wiki/EdDSA#Ed25519
        - 'Ed448': aka "Goldilocks", see https://en.wikipedia.org/wiki/Curve448
        - 'secp256k1': Bitcoin's Koblitz curve from https://www.secg.org/sec2-v2.pdf
        - 'BN256': Barreto-Naehrig curve, https://eprint.iacr.org/2010/186
        - 'BN256_twist': sextic twist of Barreto-Naehrig curve

    These curves can be used with 'affine' (default) and 'projective' coordinates.
    The Edwards curves can also be used with 'extended' coordinates, and the
    Weierstrass curves with 'jacobian' coordinates.
    """
    if coordinates is None:
        coordinates = 'affine'
    return _EllipticCurve(curvename, coordinates)


@functools.cache
def _EllipticCurve(curvename, coordinates):
    if curvename.startswith('Ed'):
        if curvename == 'Ed25519':
            p = 2**255 - 19
            gf = GF(p)
        elif curvename == 'Ed448':
            p = 2**448 - 2**224 - 1
            gf = GF(p)
        else:
            raise ValueError('invalid curvename')

        name = f'E({gf.__name__}){curvename}{coordinates}'
        match coordinates:
            case 'extended':
                base = EdwardsExtended
            case 'affine':
                base = EdwardsAffine
            case 'projective':
                base = EdwardsProjective
            case _:
                raise ValueError('invalid coordinates')

        EC = type(name, (base,), {'__slots__': ()})
        EC.field = gf

        if curvename == 'Ed25519':
            EC.a = gf(-1)  # twisted
            EC.d = gf(-121665) / gf(121666)
            y = gf(4) / gf(5)
            x2 = (1 - y**2) / (EC.a - EC.d * y**2)
            x = x2.sqrt()
            x = x if x.value%2 == 0 else -x  # enforce "positive" (even) x coordinate
            base_pt = (x, y)
            EC.order = 2**252 + 27742317777372353535851937790883648493
        else:  # 'Ed448'
            EC.a = gf(1)
            EC.d = gf(-39081)
            y = gf(19)
            x2 = (1 - y**2) / (EC.a - EC.d * y**2)
            x = x2.sqrt()
            x = x if 2*x.value < p else -x  # enforce principal root
            base_pt = (x, y)
            EC.order = 2**446 - int('8335dc163bb124b65129c96fde933d8d723a70aadc873d6d54a7bb0d', 16)
    elif curvename.startswith('BN'):
        u = 1868033**3
        p = 36*u**4 + 36*u**3 + 24*u**2 + 6*u + 1  # p = 3 (mod 4)
        if curvename == 'BN256':
            gf = GF(p)
        elif curvename == 'BN256_twist':
            gf = GF(GFpX(p)('x^2+1'))  # i^2 == -1
        else:
            raise ValueError('invalid curvename')

        name = f'E({gf.__name__}){curvename}{coordinates}'
        match coordinates:
            case 'jacobian':
                base = WeierstrassJacobian
            case 'affine':
                base = WeierstrassAffine
            case 'projective':
                base = WeierstrassProjective
            case _:
                raise ValueError('invalid coordinates')

        EC = type(name, (base,), {'__slots__': ()})
        EC.field = gf

        if curvename == 'BN256':
            EC.a = gf(0)
            EC.b = gf(3)
            base_pt = (gf(1), gf(-2))
        else:  # 'BN256_twist'
            EC.a = gf('0')
            xi = gf('x+3')
            EC.b = gf('3') / xi
            x = gf([64746500191241794695844075326670126197795977525365406531717464316923369116492,
                    21167961636542580255011770066570541300993051739349375019639421053990175267184])
            y = gf([17778617556404439934652658462602675281523610326338642107814333856843981424549,
                    20666913350058776956210519119118544732556678129809273996262322366050359951122])
            base_pt = (x, y)
        EC.order = p - 6*u**2
    elif curvename == 'secp256k1':
        p = 2**256 - 2**32 - 2**9 - 2**8 - 2**7 - 2**6 - 2**4 - 1  # p = 3 (mod 4)
        gf = GF(p)

        name = f'E({gf.__name__}){curvename}{coordinates}'
        match coordinates:
            case 'jacobian':
                base = WeierstrassJacobian
            case 'affine':
                base = WeierstrassAffine
            case 'projective':
                base = WeierstrassProjective
            case _:
                raise ValueError('invalid coordinates')

        EC = type(name, (base,), {'__slots__': ()})
        EC.field = gf

        EC.a = gf(0)
        EC.b = gf(7)
        x = gf(int('79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798', 16))
        y = gf(int('483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8', 16))
        base_pt = (x, y)
        EC.order = int('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141', 16)
    else:
        raise ValueError('curve not supported')

    assert is_prime(EC.order)
    EC.curvename = curvename
    EC.field.is_signed = False  # for consistency between sectypes and regular types
    EC.is_cyclic = True  # these EC groups are cyclic (but not all EC groups are)
    EC.gap = 256  # TODO: optimize gap value
    EC.identity = EC(check=False)
    EC.generator = EC(base_pt, check=False)
    globals()[name] = EC  # NB: exploit (almost?) unique name dynamic EC type
    return EC


class HyperellipticCurveDivisor(FiniteGroupElement):
    """Common base class for divisors in Jacobian of a hyperelliptic curve.

    Arbitrary genus, using (affine) Mumford representation.and algorithms from Cantor's 1987 paper.
    """

    __slots__ = ()

    is_additive = True
    is_multiplicative = False
    is_abelian = True
    is_cyclic = True  # NB: we use a cyclic (sub)group.
    genus = None
    field: type  # curve is defined over this field
    _identity = (1, 0)  # Mumford representation (u,v)
    gap = None
    f: Polynomial

    def __init__(self, value=None, check=True):
        field = self.field
        poly = GFpX(field.modulus)
        if value is None:
            value = map(poly, self._identity)
        if check:
            u, v = value
            if not isinstance(u, poly):
                u = poly(u)
            if not isinstance(v, poly):
                v = poly(v)
            value = (u, v)
            if (self.f - v**2) % u:
                raise ValueError('value not in Jacobian')

        self.value = tuple(value)

    def __getitem__(self, key):  # NB: no __setitem__ to prevent mutability
        return self.value[key]

    @property
    def u(self):
        return self.value[0]

    @property
    def v(self):
        return self.value[1]

    @classmethod
    def ysquared(cls, x):
        return cls.field(cls.f(x.value))

    @classmethod
    def encode(cls, m):
        """Encode message m in constant term of monic polynomial of divisor.

        Divisor (u,v) with deg u=1, u(x)=0 and deg v=0, v(x)=y, where x=-m and y^2=f(x).
        Hence, u[0]=m=-x, u[1]=1, and v[0]=y for rational point (x,y).
        """
        field = cls.field  # TODO: extend this to non-prime fields
        gap = cls.gap
        modulus = field.modulus
        for i in range(gap):
            x_0 = field(i)
            ysquared_0 = cls.ysquared(-x_0)
            if legendre(int(ysquared_0), modulus) == 1:
                x_m = field(m * gap + i)
                ysquared_m = cls.ysquared(-x_m)
                if legendre(int(ysquared_m), modulus) == 1:
                    M = cls(([x_m.value, 1], [ysquared_m.sqrt().value]), check=False)
                    Z = cls(([x_0.value, 1], [ysquared_0.sqrt().value]), check=False)
                    return M, Z

        raise ValueError('message encoding failed, try larger gap')

    @classmethod
    def decode(cls, M, Z):
        """Decode message from given group elements."""
        gap = cls.gap
        return int((M.u[0] - Z.u[0]) / gap)

    @classmethod
    def class_number(cls):
        """Count elements of Jacobian by counting unique Mumford representations (u,v)."""
        assert cls.genus <= 3
        h = 1  # deg u = 0, hence (u,v)=(1,0), deg v = -1
        if cls.genus == 0:
            return h

        poly = type(cls.f)
        p = poly.p
        p2 = (p+1) // 2
        f = cls.f.value
        mod = poly._mod
        sq = poly._sq
        v = []  # keep same list for v throughout
        for u_0 in range(p):
            u = [u_0, 1]  # deg u = 1
            f_u = mod(f, u)
            # test all v, deg v < 1
            if f_u == []:  # deg v = -1
                h += 1
            for v_0 in range(1, p2):
                if [v_0**2 % p] == f_u:  # deg v = 0
                    h += 2

            if cls.genus == 1:
                continue

            for u_1 in range(p):
                u = [u_0, u_1, 1]  # deg u = 2
                f_u = mod(f, u)
                # test all v, deg v < 2
                if f_u == []:  # deg v = -1
                    h += 1
                v.append(None)
                for v_0 in range(p):
                    if v_0 and v_0 < p2 and [v_0**2 % p] == f_u:  # deg v = 0
                        h += 2
                    v[0] = v_0
                    v.append(None)
                    for v_1 in range(1, p2):
                        v[1] = v_1
                        if mod(sq(v), u) == f_u:  # deg v = 1
                            h += 2
                    del v[-1]
                del v[-1]

                if cls.genus == 2:
                    continue

                for u_2 in range(p):
                    u = [u_0, u_1, u_2, 1]  # deg u = 3
                    f_u = mod(f, u)
                    # test all v, deg v < 3
                    if f_u == []:  # deg v = -1
                        h += 1
                    v.append(None)
                    for v_0 in range(p):
                        if v_0 and v_0 < p2 and [v_0**2 % p] == f_u:  # deg v = 0
                            h += 2
                        v[0] = v_0
                        v.append(None)
                        for v_1 in range(p):
                            v[1] = v_1
                            if v_1 and v_1 < p2 and sq(v) == f_u:  # deg v = 1
                                h += 2
                            v.append(None)
                            for v_2 in range(1, p2):
                                v[2] = v_2
                                if mod(sq(v), u) == f_u:  # deg v = 2
                                    h += 2
                            del v[-1]
                        del v[-1]
                    del v[-1]
        return h

    @staticmethod
    def _reduce(f, genus, D):
        u, v = D
        while u.degree() > genus:
            u = (f - v**2) // u
            v = (-v) % u
        u = u.monic()
        return u, v

    @staticmethod
    def _operation(f, genus, D1, D2):
        # formula (C3a) from Cantor's 1987 paper
        poly = type(f)
        u1, v1 = D1
        u2, v2 = D2
        d, _, h2 = poly.gcdext(u1, u2)
        if d == 1:
            u = u1 * u2
            v = (v2 + h2 * u2 * (v1 - v2)) % u
        else:
            d, h, h3 = poly.gcdext(d, v1 + v2)
            h2 *= h
            u = u1 * u2 // d**2
            v = (v2 + (h2 * u2 * (v1 - v2) + h3 * (f - v2**2)) // d) % u
        return HyperellipticCurveDivisor._reduce(f, genus, (u, v))

    @classmethod
    def operation(cls, D1, D2, /):
        return cls(HyperellipticCurveDivisor._operation(cls.f, cls.genus, D1, D2), check=False)

    @staticmethod
    def _operation2(f, genus, D):
        # formula (C5a) from Cantor's 1987 paper
        poly = type(f)
        u, v = D
        d, _, h3 = poly.gcdext(u, 2*v)
        if d == 1:  # TODO: check if d=1 can be assumed if f is irreducible
            u = u**2
            v = (v + h3 * (f - v**2)) % u
        else:
            u = (u // d)**2
            v = (v + h3 * ((f - v**2) // d)) % u
        return HyperellipticCurveDivisor._reduce(f, genus, (u, v))

    @classmethod
    def operation2(cls, D, /):
        # formula (C5a) from Cantor's 1987 paper
        return cls(HyperellipticCurveDivisor._operation2(cls.f, cls.genus, D), check=False)

    @classmethod
    def inversion(cls, D, /):
        u, v = D
        return cls((u, -v), check=False)  # (-v) % u = -v because deg v < deg u

    @classmethod
    def equality(cls, D1, D2, /):
        return D1.value == D2.value


class HCDivisorCL(HyperellipticCurveDivisor):
    """Costello-Lauter formulas for genus 2.

    With one exception, only divisors (u,v) with u of full degree 2 are assumed.
    Such divisors are represented as a 6-tuple (u1, u0, v1, v0, u1u1, u1u0),
    where u(x)=x^2+u1x+u0 and v(x)=v1x+v0.

    The only exception is that the identity (1,0) is also considered,
    represented by the 6-tuple (0,0,0,0,0,0).

    See "Group Law Computations on Jacobians of Hyperelliptic Curves" by Costello and Lauter.
    """

    genus = 2
    _identity = (0,) * 6  # (u1, u0, v1, v0, u1u1, u1u0)

    def __init__(self, value=None, check=True):
        if value is None:
            value = map(self.field, self._identity)
        elif len(value) == 4:
            u1, u0, v1, v0 = value
            value = (u1, u0, v1, v0, u1**2, u1 * u0)
        if check:
            field = self.field
            value = list(value)
            for i in range(len(value)):
                if not isinstance(value[i], field):
                    value[i] = field(value[i])
            if value[0]**2 != value[4] or value[0] * value[1] != value[5]:
                raise ValueError('incorrect extended coordinates')

            if (self.f - self.v**2) % self.u:
                raise ValueError('value not in Jacobian')

        self.value = tuple(value)

    @property
    def u(self):
        poly = GFpX(self.field.modulus)
        if self.value == self._identity:
            a = [1]
        else:
            a = [self.value[1].value, self.value[0].value, 1]
        return poly(a, check=False)

    @property
    def v(self):
        poly = GFpX(self.field.modulus)
        if self.value == self._identity:
            a = []
        else:
            a = [self.value[3].value, self.value[2].value]
        return poly(a, check=False)

    def __repr__(self):
        return str((self.u, self.v))

    @classmethod
    def encode(cls, m):
        """Encode message m in terms of monic polynomial of divisor.

        Divisor (u,v) with deg u=2, u(x)=(x+m)^2=x^2+2mx+m^2=0
        and deg v=0, v(x)=y, where x=-m and y^2=f(x).
        Hence, u[2]=1, u[1]=2m=-2x, u[0]=m^2, v[0]=yfor rational point (x,y).
        """
        field = cls.field  # TODO: extend this to non-prime fields
        gap = cls.gap
        modulus = field.modulus
        for i in range(gap):
            x_0 = field(i)
            ysquared_0 = cls.ysquared(-x_0)
            if legendre(int(ysquared_0), modulus) == 1:
                x_m = field(m * gap + i)
                ysquared_m = cls.ysquared(-x_m)
                if legendre(int(ysquared_m), modulus) == 1:
                    M = cls((2*x_m, x_m**2, field(0), ysquared_m.sqrt()), check=False)
                    Z = cls((2*x_0, x_0**2, field(0), ysquared_0.sqrt()), check=False)
                    return M, Z

    @classmethod
    def decode(cls, M, Z):
        """Decode message from given group elements."""
        gap = cls.gap
        return int((M.u[1] - Z.u[1]) / (2*gap))

    @classmethod
    def operation(cls, D1, D2, /):
        if D1.value == cls._identity:
            return D2

        if D2.value == cls._identity:
            return D1

        u11, u10, v11, v10, u11u11, u11u10 = D1
        u21, u20, v21, v20, u21u21, u21u20 = D2
        try:
            uv = cls._AffADD(u11, u10, v11, v10, u11u11, u11u10,
                             u21, u20, v21, v20, u21u21, u21u20)
        except Exception as e:
            # fall back to general addition
            poly = type(cls.f)
            D1 = (poly([u10.value, u11.value, 1]), poly([v10.value, v11.value]))
            D2 = (poly([u20.value, u21.value, 1]), poly([v20.value, v21.value]))
            u, v = HyperellipticCurveDivisor._operation(cls.f, cls.genus, D1, D2)
            if (u, v) == (1, 0):
                uv = None
            else:
                F = cls.field
                uv = F(u[1]), F(u[0]), F(v[1]), F(v[0])
        return cls(uv, check=False)

    @classmethod
    def operation2(cls, D, /):
        if D.value == cls._identity:
            return D

        u1, u0, v1, v0, u1u1, u1u0 = D
        F = cls.field
        try:
            uv = cls._AffDBL(u1, u0, v1, v0, u1u1, u1u0, F(cls.f[2]), F(cls.f[3]))
        except Exception as e:
            # fall back to general doubling
            poly = type(cls.f)
            D = (poly([u0.value, u1.value, 1]), poly([v0.value, v1.value]))
            u, v = HyperellipticCurveDivisor._operation2(cls.f, cls.genus, D)
            uv = F(u[1]), F(u[0]), F(v[1]), F(v[0])
        return cls(uv, check=False)

    @classmethod
    def inversion(cls, D, /):
        if D.value == cls._identity:
            return D

        u1, u0, v1, v0, u1u1, u1u0 = D
        v1 = -v1
        v0 = -v0
        uv = u1, u0, v1, v0, u1u1, u1u0
        return cls(uv, check=False)

    @classmethod
    def _AffADD(cls, u1, u0, v1, v0, u1s, u1u0, u1d, u0d, v1d, v0d, u1ds, u1du0d):
        # cf. Costello--Lauter Table 1
        M1 = u0 - u0d
        M2 = u1du0d - u1u0
        M3 = u1 - u1d
        M4 = u1ds - u1s + M1
        z1 = v0d - v0
        z2 = v1d - v1
        return cls._fin(M1, M2, M3, M4, z1, z2, u1, u0, v1, v0, u1s, u1u0, u1 + u1d, v1d, u1ds)

    @classmethod
    def _AffDBL(cls, u1, u0, v1, v0, u1s, u1u0, f2, f3):
        # cf. Costello--Lauter Table 1
        v1s = v1**2  # 1S
        tu1v1 = (u1 + v1)**2 - u1s - v1s  # 1S
        M1 = 2*v0 + tu1v1
        M2 = -2*v1 * (u0 + 2*u1s)  # 1M
        M3 = 2*v1
        M4 = 2*(v0 - tu1v1)
        z1 = f2 + + 2*u1u0 + 2*u1s * u1 - v1s  # 1M
        z2 = f3 - 2*u0 + 3*u1s
        return cls._fin(M1, M2, M3, M4, z1, z2, u1, u0, v1, v0, u1s, u1u0, 2*u1, v1, u1s)  # 2M + 2S

    @classmethod
    def _fin(cls, M1, M2, M3, M4, z1, z2, u1, u0, v1, v0, u1s, u1u0, uS, v1d, u1ds):
        t1 = (M2 - z1) * (M4 + z2)  # M
        t2 = (M2 + z1) * (M4 - z2)  # M
        t3 = (M1 + z1) * (M3 - z2)  # M
        t4 = (M1 - z1) * (M3 + z2)  # M
        l2 = t2 - t1
        l3 = t4 - t3
        d = t1 + t2 - t3 - t4 + 2*(M1 - M2) * (M3 + M4)  # M
        A = 1/(d * l3)  # M + I
        B = d * A  # M
        C = d * B  # M
        D = l2 * B  # M
        E = l3**2 * A  # S + M
        Cs = C**2  # S
        u1dd = 2*D - Cs - uS
        u0dd = D**2 + C * (v1 + v1d) - ((u1dd - Cs) * uS + u1s + u1ds)/2  # S + 2M
        uu1dd = u1dd**2  # S
        uu0dd = u1dd * u0dd  # M
        v1dd = D * (u1 - u1dd) + uu1dd - u0dd - u1s + u0  # M
        v0dd = D * (u0 - u0dd) + uu0dd - u1u0  # M
        v1dd = -(E * v1dd + v1)  # M
        v0dd = -(E * v0dd + v0)  # M
        return u1dd, u0dd, v1dd, v0dd, uu1dd, uu0dd  # total I + 17M + 4S


def HyperellipticCurve(curvename=None, coordinates=None, p=None, l=None, genus=None):
    """Create type for hyperelliptic curve group with given parameters.

    With curvename='kummer1271' the genus-2 curve due to Gaudry and Schost is obtained,
    which is defined modulo p=2^127-1, the twelfth Mersenne prime.

    By default, curvename='DGS', which stands for Dobson, Galbraith, and Smith, who
    specified a way to generate random Jacobians together with a ...

    Alternatively, given p or its bit length l as modulus for the underlying prime field,
    a random curve of the specified genus (default 3) will be generated, using the method
    of Dobson, Galbraith and Smith (Algorithm 4 from https://eprint.iacr.org/2020/196).
    The randomness is seeded with the prime number p.

    The coordinates used with these curves are 'affine', by default.
    Currently, the only alternative coordinates are the Costello--Lauter 'extended' coordinates.
    """
    if curvename is None:
        curvename = 'DGS'
    if curvename == 'DGS':
        if genus is None:
            genus = 3
        # e.g., l = 5 gives order 29389 which is approx p^3 = 31**3 = 29791
        if p is None:
            p = find_prime_root(l)[0]  # l-bit Blum prime p (p mod 4 = 3)
    elif curvename == 'kummer1271':
        p = 2**127 - 1
        genus = 2
        coordinates = 'extended'
    else:
        raise ValueError('curve not supported')

    if coordinates is None:
        coordinates = 'affine'

    return _HyperellipticCurve(p, genus, curvename, coordinates)


@functools.cache
def _HyperellipticCurve(p, genus, curvename, coordinates):
    gf = GF(p)
    poly = GFpX(p)
    name = f'HC({gf.__name__}){curvename}'
    if curvename == 'DGS':
        # Generate random (u,v) and irreducible f such that u | f - v^2:
        rnd = random.Random(p)  # randomness seeded with p
        u = poly([rnd.randrange(p) for _ in range(genus)] + [1])
        v = poly([rnd.randrange(p) for _ in range(genus)])
        while True:
            w = poly([rnd.randrange(p) for _ in range(genus+1)] + [1])
            f = v**2 + u * w  # monic of degree 2*genus+1
            if poly.gcd(f, f.deriv()) == 1 and poly.is_irreducible(f):
                break
        n = None
    elif curvename == 'kummer1271':  # Gaudry & Schost curve
        f = [81689052950067229064357938692912969725,   # x^0
             9855732443590990513334918966847277222,    # x^1
             154735094972565041023366918099598639851,  # x^2
             76637216448498510246042731975843417626,   # x^3
             64408548613810695909971240431892164827,   # x^4
             1]                                        # x^5
        # NB: f has 5 linear factors
        assert poly.gcd(poly(f), poly(f).deriv()) == 1
        n = 1809251394333065553571917326471206521441306174399683558571672623546356726339
        # n is prime and order of Jacobian is 16 * n
        u = poly('x^2+53887750494529953094583234541973147544x+'
                 '152781149156717595995762065350002864540')
        v = poly('117497929065723271999297121045670554255x+'
                 '93722789515836547535106638431311448542')
        # subgroup <(u,v)> of prime order n
    else:
        raise ValueError('curve not supported')

    if genus == 2 and coordinates == 'extended':
        base = HCDivisorCL
        # ensure f[4]=0, by switching to isomorphic curve x:=x-f[4]/5
        f45 = gf(f[4])/5
        f3 = 10*f45**2 - 4*f[4]*f45 + f[3]
        f2 = -10*f45**3 + 6*f[4]*f45**2 - 3*f[3]*f45 + f[2]
        f1 = 5*f45**4 - 4*f[4]*f45**3 + 3*f[3]*f45**2 - 2*f[2]*f45 + f[1]
        f0 = -f45**5 + f[4]*f45**4 - f[3]*f45**3 + f[2]*f45**2 - f[1]*f45 + f[0]
        f = poly([f0.value, f1.value, f2.value, f3.value, 0, 1])
        u1 = u[1] - 2*f45
        u0 = u[0] - u[1]*f45 + f45**2
        v1 = gf(v[1])
        v0 = v[0] - v[1]*f45
        base_pt = (u1, u0, v1, v0, u1**2, u1 * u0)
    else:
        base = HyperellipticCurveDivisor
        base_pt = (u, v)

    HC = type(name, (base,), {'__slots__': ()})
    HC.field = gf
    HC.genus = genus
    HC.curvename = curvename
    HC.f = f
    HC.field.is_signed = False  # for consistency between sectypes and regular types
    HC.is_cyclic = True  # NB: these HC subgroups are cyclic by definition
    HC.gap = 256  # TODO: optimize gap value
    HC.identity = HC(check=False)
    HC.generator = HC(base_pt, check=False)
    if n is None and genus <= 3 and genus * p.bit_length() <= 3:
        n = HC.class_number()
    HC.order = n
    assert HC.order is None or HC.generator^HC.order == HC.identity
    globals()[name] = HC  # NB: exploit (almost?) unique name dynamic HC type
    return HC


class ClassGroupForm(FiniteGroupElement):
    """Common base class for class groups of imaginary quadratic fields.

    Represented by primitive positive definite forms (a,b,c) of discriminant D<0.
    That is, all forms (a,b,c) with D=b^2-4ac<0 satisfying gcd(a,b,c)=1 and a>0.
    """

    __slots__ = ()

    is_multiplicative = True
    is_abelian = True
    discriminant: int
    bit_length = None
    gap = None
    order = None

    def __init__(self, value=None, check=True):
        """Create a binary quadratic form (a,b,c).

        Invariant: form (a,b,c) is reduced.
        """
        if value is None:  # set principal form = identity
            k = self.discriminant%2
            value = (1, k, (k**2 - self.discriminant) // 4)
            check = False
        elif isinstance(value, list):
            value = tuple(value)
        if len(value) == 2:
            a, b = value
            c = (b**2 - self.discriminant) // (4*a)
            value = (a, b, c)
            check = True
        if check:
            a, b, c = value
            if b**2 - 4*a * c != self.discriminant:
                raise ValueError('wrong discriminant')

            if a <= 0:
                raise ValueError('positive definite form required')

            value = ClassGroupForm._reduce((a, b, c))
        self.value = value

    def __getitem__(self, key):  # NB: no __setitem__ to prevent mutability
        return self.value[key]

    # See Henri Cohen's book "A Course in Computational Algebraic Number Theory", Chapter 5.
    @staticmethod
    def _reduce(f):  # Cohen: Algorithm 5.4.2
        a, b, c = f
        # normalize
        r = (a - b) // (2*a)
        b, c = b + 2*r*a, a*r**2 + b*r + c
        while not (-a < b <= a <= c and (a != c or b >= 0)):  # check reduced
            # reduce
            s = (c + b) // (2*c)
            a, b, c = c, -b + 2*s*c, c*s**2 - b*s + a
        return a, b, c

    @classmethod
    def operation(cls, f1, f2, /):  # Cohen: Algorithm 5.4.9 (NUCOMP)
        if f1[0] < f2[0]:
            f1, f2 = f2, f1
        a1, b1, c1 = f1
        a2, b2, c2 = f2
        s = (b1 + b2) // 2
        n = b2 - s

        d, u, v = gcdext(a2, a1)
        if d == 1:
            A = -u * n
            d1 = d
        elif s % d == 0:
            A = -u * n
            d1 = d
            a1 //= d1
            a2 //= d1
            s //= d1
        else:
            d1, u1, _ = gcdext(s, d)
            if d1 > 1:
                a1 //= d1
                a2 //= d1
                s //= d1
                d //= d1
            l = (-u1 * (u * (c1 % d) + v * (c2 % d))) % d
            A = -u * (n // d) + l * (a1 // d)
        A = A % a1
        A1 = a1 - A
        if A1 < A:
            A = - A1

        d, v3 = a1, A
        v2, v = 1, 0
        z = 0
        L = iroot(-cls.discriminant//4, 4)[0]
        while abs(v3) > L:  # partial Euclid
            d, (q, v3) = v3, divmod(d, v3)
            v, v2, = v2, v - q * v2
            z += 1
        if z%2:
            v2, v3 = -v2, -v3

        if z == 0:
            Q1 = a2 * v3
            f = (Q1 + n) // d
            g = (v3 * s + c2) // d
            a3 = d * a2
            b3 = 2*Q1 + b2
            c3 = v3 * f + g * d1  # erratum Cohen (step 6)
        else:
            b = (a2 * d + n * v) // a1
            Q1 = b * v3
            Q2 = Q1 + n
            f = Q2 // d
            e = (s * d + c2 * v) // a1
            Q3 = e * v2
            Q4 = Q3 - s
            g = Q4 // v
            a3 = d * b + d1 * e * v
            b3 = Q1 + Q2 + d1 * (Q3 + Q4)
            c3 = v3 * f + d1 * g * v2
        f3 = int(a3), int(b3), int(c3)  # NB: convert from gmpy2.mpz, if gmpy2 is used for gcdext()
        return cls(cls._reduce(f3), check=False)

    @classmethod
    def operation2(cls, f, /):  # Cohen: Algorithm 5.4.8 (NUDUPL)
        a, b, c = f
        d1, u, _ = gcdext(b, a)
        assert d1 == 1  # because -discriminant is prime
        A = a // d1
        B = b // d1
        C = (-c * u) % A
        C1 = A - C
        if C1 < C:
            C = -C1

        d, v3 = A, C
        v2, v = 1, 0
        z = 0
        L = iroot(-cls.discriminant//4, 4)[0]
        while abs(v3) > L:  # partial Euclid
            d, (q, v3) = v3, divmod(d, v3)
            v, v2, = v2, v - q * v2
            z += 1
        if z%2:
            v2, v3 = -v2, -v3

        if z == 0:
            g = (B * v3 + c) // d
            a2 = d**2
            b2 = b + 2*d * v3
            c2 = v3**2 + g * d1
        else:
            e = (c * v + B * d) // A
            h = e * v2
            g = (h - B) // v
            a2 = d**2 + d1 * e * v
            b2 = d1 * (h + v * g) + 2*d * v3
            c2 = v3**2 + d1 * g * v2
        f2 = int(a2), int(b2), int(c2)  # NB: convert from gmpy2.mpz, if gmpy2 is used for gcdext()
        return cls(cls._reduce(f2), check=False)

    @classmethod
    def inversion(cls, f, /):
        a, b, c = f
        return cls(cls._reduce((a, -b, c)), check=False)

    @classmethod
    def equality(cls, f1, f2, /):
        return f1.value == f2.value

    @classmethod
    def encode(cls, m):
        """Encode message m in the first coefficient of a form."""
        D = cls.discriminant
        gap = cls.gap
        assert (m+1) * gap <= isqrt(-D)/2  # ensure M (and Z) will be reduced
        assert gap%4 == 0
        for i in range(0, gap, 4):
            a_0 = i + 3
            b_0 = int(powmod(D, (a_0+1)//4, a_0))
            if (b_0**2 - D) % a_0 == 0:
                # NB: gcd(a_0,b_0)=1 because -D is prime
                a_m = m * gap + i + 3
                b_m = int(powmod(D, (a_m+1)//4, a_m))
                if (b_m**2 - D) % a_m == 0:
                    # NB: gcd(a_m,b_m)=1 because -D is prime
                    b_m = a_m - b_m if D%2 != b_m%2 else b_m
                    c_m = (b_m**2 - D) // (4*a_m)
                    M = cls((a_m, b_m, c_m), check=False)
                    b_0 = a_0 - b_0 if D%2 != b_0%2 else b_0
                    c_0 = (b_0**2 - D) // (4*a_0)
                    Z = cls((a_0, b_0, c_0), check=False)
                    return M, Z  # NB: M and Z are reduced

        raise ValueError('message encoding failed, try larger gap')

    @classmethod
    def decode(cls, M, Z):
        """Decode message from given group elements."""
        gap = cls.gap
        return (M[0] - Z[0]) // gap


def _class_number(D):  # Cohen: Algorithm 5.3.5
    """Compute the class number h(D) for squarefree discriminant D < 0, D = 1 (mod 4)."""
    h = 1
    for b in range(1, 1 + isqrt(-D // 3), 2):
        b2 = b**2
        a = max(b, 2)
        a2 = max(b2, 4)
        q = (b2 - D) >> 2
        while a2 <= q:
            if q % a == 0:
                h += 1 if a == b or a2 == q else 2
            a2 += (a << 1) | 1
            a += 1
    return h


def _calculate_gap(l):
    """Calculate gap size for en/decoding.

    Gap must be a multiple of 4.
    """
    gap = l
    while gap != (gap := round(3.5 * l * math.log(gap))):
        pass
    return gap - gap%4


def ClassGroup(Delta=None, l=None):
    """Create type for class group, given (bit length l of) discriminant Delta.

    The following conditions are imposed on discriminant Delta:

        - Delta < 0, only supporting class groups of imaginary quadratic field
        - Delta = 1 (mod 4), preferably Delta = 1 (mod 8)
        - -Delta is prime

    This implies that Delta is a fundamental discriminant.
    """
    if l is not None:
        if Delta is None:
            # find fundamental discriminant Delta of bit length l >= 2
            p = next_prime(1 << l-1)
            while p != 3 and p != 11 and p%8 != 7:
                p = next_prime(p)
            Delta = int(-p)  # D = 1 mod 4, and even D = 1 mod 8 if possible (and -D is prime)
    elif Delta is None:
        Delta = -3
    if Delta%4 != 1:
        raise ValueError('discriminant required to be 1 modulo 4, preferably 1 modulo 8')

    if Delta >= 0 or not is_prime(-Delta):
        raise ValueError('negative prime discriminant required')

    return _ClassGroup(Delta)


@functools.cache
def _ClassGroup(Delta):
    l = Delta.bit_length()
    name = f'Cl{l}({Delta})'
    Cl = type(name, (ClassGroupForm,), {'__slots__': ()})
    Cl.discriminant = Delta
    Cl.bit_length = l
    Cl.gap = _calculate_gap(l)
    if l <= 24:
        Cl.order = _class_number(Delta)
    else:
        Cl.order = None  # NB: leave order as "unknown"
    Cl.identity = Cl()

    # Class groups likely to have large cyclic subgroups, see Conjecture 5.10.1 in Cohen.
    if Delta%8 == 1:
        # Use the following generator from the Chia VDF competition,
        # see https://www.chia.net/2018/11/07/chia-vdf-competition-guide.en.html:
        g = Cl((2, 1, (1 - Delta) // 8))  # order of g around sqrt(-D/4)
    else:
        g = Cl.identity  # trivial generator
    Cl.generator = g
    Cl.is_cyclic = True  # We use the (sub)group generated by g.
    globals()[name] = Cl  # NB: exploit (almost?) unique name dynamic Cl type
    return Cl

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

Five types of groups are currently supported, aimed mainly at applications
in cryptography:

    - symmetric groups of any degree n (n>=0)
    - quadratic residue groups modulo a safe prime
    - Schnorr groups (prime-order subgroups of the multiplicative group of a finite field)
    - elliptic curve groups (Edwards curves, a Koblitz curve, and Barreto-Naehrig curves)
    - class groups of imaginary quadratic fields

The structure of most of these groups will be trivial, preferably cyclic or even
of prime order. Where applicable, a generator of the group (or a sufficiently
large subgroup) is provided to accommodate discrete log and Diffie-Hellman
hardness assumptions.
"""

import math
import decimal
import functools
from mpyc.gmpy import powmod, gcdext, is_prime, next_prime, prev_prime, legendre, isqrt, iroot
from mpyc.gfpx import GFpX
from mpyc.finfields import GF


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
        group = type(self)
        return group.inversion(self)

    def __xor__(self, other):  # overload ^
        if isinstance(other, int):
            group = type(self)
            return group.repeat(self, other)

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

    def inverse(self):  # TODO: reconsider use of instance methods
        """For convenience."""
        return type(self).inversion(self)

    @classmethod
    def equality(cls, a, b, /):
        """Return a == b."""
        raise NotImplementedError

    @staticmethod
    def repeat(a, n):
        """Return nth @-power of a (written a^n), for any integer n."""
        cls = type(a)
        if n < 0:
            a = cls.inversion(a)
            n = -n
        d = a
        c = cls.identity
        for i in range(n.bit_length() - 1):
            # d = a^(2^i) holds
            if (n >> i) & 1:
                c = cls.operation(c, d)
            d = cls.operation2(d)
        if n:
            c = cls.operation(c, d)
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
        r1, r2, r3, r4 = r1 * r2, r3 * r4, 2*cls.d * t1 * t2, 2*z1 * z2
        r1, r2, r3, r4 = r2 - r1, r4 - r3, r4 + r3, r2 + r1
        pt3 = r1 * r2, r3 * r4, r2 * r3, r1 * r4
        return cls(pt3, check=False)

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
        if coordinates == 'extended':
            base = EdwardsExtended
        elif coordinates == 'affine':
            base = EdwardsAffine
        elif coordinates == 'projective':
            base = EdwardsProjective
        else:
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
        if coordinates == 'jacobian':
            base = WeierstrassJacobian
        elif coordinates == 'affine':
            base = WeierstrassAffine
        elif coordinates == 'projective':
            base = WeierstrassProjective
        else:
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
        if coordinates == 'jacobian':
            base = WeierstrassJacobian
        elif coordinates == 'affine':
            base = WeierstrassAffine
        elif coordinates == 'projective':
            base = WeierstrassProjective
        else:
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
    while True:
        gap1 = round(3.5 * l * math.log(gap))
        if gap != gap1:
            gap = gap1
        else:
            break
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
    elif Delta%4 == 1:
        g = Cl.identity  # trivial generator
    Cl.generator = g
    Cl.is_cyclic = True  # We use the (sub)group generated by g.
    globals()[name] = Cl  # NB: exploit (almost?) unique name dynamic Cl type
    return Cl

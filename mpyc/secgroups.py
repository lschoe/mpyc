"""This module provides secure (secret-shared) types of finite groups in MPyC.

Secure versions of all groups supported by the module mpyc.fingroups are available:
symmetric groups, quadratic residues, elliptic curve groups, and class groups.
"""

import itertools
import functools
import inspect
import asyncio
from mpyc.finfields import FiniteFieldElement
import mpyc.fingroups as fg
from mpyc.thresha import _recombination_vector
from mpyc import asyncoro
from mpyc.sectypes import SecureObject, SecureFiniteField, SecureInteger
from mpyc.seclists import seclist
import mpyc.mpctools

runtime = None


class SecureFiniteGroup(SecureObject):
    """Abstract base class for secure (secret-shared) finite groups elements."""

    __slots__ = ()

    group: type
    sectype: type  # secure type of elementary shares representing a group element
    identity = None

    def __matmul__(self, other):  # overload @
        cls = type(self)
        if self is other:
            return cls.operation2(self)

        if isinstance(other, cls.group):
            other = cls(other)  # TODO: exploit other was in the clear
        elif not isinstance(other, cls):
            return NotImplemented

        return cls.operation(self, other)

    def __rmatmul__(self, other):  # overload @
        if isinstance(other, self.group):
            other = type(self)(other)
        else:
            return NotImplemented

        return type(self).operation(other, self)

    def __invert__(self):  # overload ~
        return type(self).inversion(self)

    def __xor__(self, other):  # overload ^
        return type(self).repeat(self, other)   # check other is int?

    def __add__(self, other):
        if not self.group.is_additive:
            raise TypeError('group not additive')

        return type(self).__matmul__(self, other)

    def __radd__(self, other):
        if not self.group.is_additive:
            raise TypeError('group not additive')

        return type(self).__rmatmul__(self, other)

    def __neg__(self):
        if not self.group.is_additive:
            raise TypeError('group not additive')

        return type(self).__invert__(self)

    def __sub__(self, other):
        if not self.group.is_additive:
            raise TypeError('group not additive')

        other = type(other).__invert__(other)   # maybe add type check other fg or secgrp
        return type(self).__matmul__(self, other)

    def __rsub__(self, other):
        if not self.group.is_additive:
            raise TypeError('group not additive')

        secgrp = type(self)
        a = secgrp.__invert__(self)
        return secgrp.__rmatmul__(a, other)

    def __mul__(self, other):
        if self.group.is_multiplicative:
            return type(self).__matmul__(self, other)

        if self.group.is_additive:
            return NotImplemented

        raise TypeError('* not defined for group')

    def __rmul__(self, other):
        if self.group.is_multiplicative:
            if self.group.is_abelian:
                return type(self).__matmul__(self, other)

            other = type(self)(other)
            return type(self).__matmul__(other, self)

        if self.group.is_additive:
            return type(self).__xor__(self, other)

        raise TypeError('* not defined for group')

    def __truediv__(self, other):
        if not self.group.is_multiplicative:
            raise TypeError('group not multiplicative')

        other = type(other).__invert__(other)
        return type(self).__matmul__(self, other)

    def __rtruediv__(self, other):
        if not self.group.is_multiplicative:
            raise TypeError('group not multiplicative')

        secgrp = type(self)
        # assume other is fg elt or 1
        a = secgrp.__invert__(self)
        if other == 1:
            return a

        return secgrp.__rmatmul__(a, other)

    def __pow__(self, other):
        if not self.group.is_multiplicative:
            raise TypeError('group not multiplicative')

        return type(self).__xor__(self, other)

    def __eq__(self, other):
        secgrp = type(self)
        if isinstance(other, self.group):
            other = secgrp(other)
        elif not isinstance(other, secgrp):
            return NotImplemented

        return secgrp.equality(self, other)

    def __ne__(self, other):
        return 1 - self.__eq__(other)

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

    @classmethod
    def equality(cls, a, b, /):
        """Return a == b."""
        raise NotImplementedError

    def inverse(self):
        """For ease of use."""  # instance method a.inverse()
        return type(self).inversion(self)

    @classmethod
    def _input(cls, x, senders):
        """Input a list x of secure group elements. See runtime.input()."""
        is_tuple = isinstance(x[0].share, tuple)
        if is_tuple:
            r = len(x[0].share)
            x = [_ for a in x for _ in a.share]
        else:
            x = [a.share for a in x]
        shares = runtime.input(x, senders)
        if is_tuple:
            shares = [[s[i:i + r] for i in range(0, len(s), r)] for s in shares]
        return [[cls(a) for a in s] for s in shares]

    @classmethod
    async def _output(cls, x, receivers, threshold):
        """Output a list x of secure group elements. See runtime.output()."""
        is_tuple = isinstance(x[0].share, tuple)
        if is_tuple:
            r = len(x[0].share)
            x = [_ for a in x for _ in a.share]
        else:
            x = [a.share for a in x]
        y = await runtime.output(x, receivers, threshold)
        if issubclass(cls, SecureSymmetricGroupElement):
            y = list(map(int, y))
        if is_tuple:
            y = [y[i:i + r] for i in range(0, len(y), r)]
        group = lambda a: cls.group(a, check=False)
        return list(map(group, y))

    @classmethod
    def if_else(cls, c, a, b):
        """Secure selection based on binary condition c between group elements a and b.

        Condition c must be of a secure number type compatible with the group,
        and its value should be 0 or 1. Input a must be compatible with the group as
        well, either of the secure type cls or of type cls.group. Same for input b.
        """
        if not isinstance(c, cls.sectype):
            c = runtime.convert(c, cls.sectype)
        if not isinstance(a, SecureObject):
            a = cls(a)
        if not isinstance(b, SecureObject):
            b = cls(b)
        if isinstance(a.share, tuple):
            a = [_.share for _ in a.share]
            b = [_.share for _ in b.share]
        else:
            a = a.share
            b = b.share
        return cls(runtime.if_else(c, a, b))

    @classmethod
    def repeat(cls, a, x):
        """Return xth @-power of a (written a^x), for any integral number x.

        Base a is either a public or a secure group element.
        Exponent x is either a public or a secure integral number.
        Possibly a, x are lists (of same length)
        """
        secgrp = cls

        if (isinstance(a, (SecureQuadraticResidue, SecureSchnorrGroupElement))
           and isinstance(x, int)):
            return type(a)(a.share**x)  # special case for fast exp

        # Case: Public exponent, secret output.
        if isinstance(x, (int, FiniteFieldElement)):  # assume a secure group elt
            return type(a).group.repeat(a, x)

        if not isinstance(a, SecureObject):
            assert isinstance(x, (SecureFiniteField, SecureInteger))
            # Case: Public base, secret exponent, secret output.
            return repeat_public_base_secret_output(a, x, secgrp)

        # Case: Secret base, secret exponent, secret output
        return repeat_secret_base_secret_output(a, x, secgrp)

    @classmethod
    def repeat_public(cls, a, x):
        return repeat_public_base_public_output(a, x)

# TODO: refactor secure repeat functions


def repeat_secret_base_secret_output(a, x, secgrp):
    """Compute [a]^[x]->[a^x]."""
    x = runtime.to_bits(x)
    b = a
    c = secgrp.if_else(x[0], a, secgrp.identity)
    for x_i in x[1:]:
        b = b @ b
        c = secgrp.if_else(x_i, c @ b, c)
    return c


@asyncoro.mpc_coro
async def repeat_public_base_secret_output(a, x, secgrp):
    """Compute a^[x]->[a^x]."""
    # a in prime order group, x in Z.
    # x is a secure prime field element or secure int.
    await runtime.returnType(secgrp)
    field = x.field
    m = len(runtime.parties)
    lambda_i = _recombination_vector(field, range(1, m+1), 0)[runtime.pid]
    x_i = await runtime.gather(x)
    e_i = int(lambda_i * x_i)
    if isinstance(x, SecureFiniteField) and x.subfield is not None:
        # value of x in prime field
        e_i %= field.characteristic
    c_i = secgrp.group.repeat(a, e_i)
    c = runtime.input(secgrp(c_i))
    return mpyc.mpctools.reduce(secgrp.operation, c)


@asyncoro.mpc_coro
async def repeat_public_base_public_output(a, x) -> asyncio.Future:
    """Multi-exponentiation for given base(s) a and exponent(s) x."""
    # a is a group element, or a list of group elements.
    # x is secure number (prime field element, or integer), or a list of secure numbers.
    if not isinstance(a, list):
        a, x = [a], [x]
    field = x[0].field
    group = type(a[0])
    m = len(runtime.parties)
    lambda_i = _recombination_vector(field, range(1, m+1), 0)[runtime.pid]
    x_i = await runtime.gather(x)
    e_i = [int(lambda_i * s_i) for s_i in x_i]
    if isinstance(x, SecureFiniteField) and x.subfield is not None:
        # values in x in prime field
        for j in range(len(x)):
            e_i[j] %= field.characteristic
    c_i = functools.reduce(group.operation, map(group.repeat, a, e_i))
    c = await runtime.transfer(c_i)
    return functools.reduce(group.operation, c)


class SecureSymmetricGroupElement(SecureFiniteGroup):
    """Common base class for secure (secret-shared) symmetric group elements."""

    __slots__ = ()

    def __init__(self, value=None):
        """Ensure all coefficients of value are of secure field type.

        Enforce value is a tuple.
        """
        n = self.group.degree
        if value is None:
            value = [None] * n
        elif isinstance(value, self.group):
            value = value.value
        else:
            if not (isinstance(value, (tuple, list)) and len(value) == n):
                raise ValueError(f'tuple/list of length {n} required')

        secfld = self.sectype
        value = tuple(secfld(a) if not isinstance(a, secfld) else a for a in value)
        super().__init__(value)

    def set_share(self, value):
        for a, b in zip(self.share, value):
            a.set_share(b.share)

    @classmethod
    def operation(cls, p, q, /):
        """First p then q."""
        q = seclist(q.share)
        return cls(tuple(q[j] for j in p.share))

    @classmethod
    def inversion(cls, p, /):
        n = len(p.share)
        q = seclist(p.share)  # use p.share as dummy of the right type
        for i in range(n):
            q[p.share[i]] = i
        return cls(tuple(q))

    @classmethod
    def equality(cls, p, q, /):  # return type is self.sectype
        return seclist(p.share) == seclist(q.share)


class SecureQuadraticResidue(SecureFiniteGroup):
    """Common base class for secure (secret-shared) quadratic residues."""

    __slots__ = ()

    def __init__(self, value=None):
        """Ensure value is of secure field type."""
        if isinstance(value, self.group):
            value = value.value
        secfld = self.sectype
        if not isinstance(value, secfld):
            value = secfld(value)
        super().__init__(value)

    def set_share(self, value):
        self.share.set_share(value.share)

    @classmethod
    def operation(cls, a, b, /):
        return cls(a.share * b.share)

    @classmethod
    def inversion(cls, a, /):
        return cls(1/a.share)

    @classmethod
    def equality(cls, a, b, /):
        return a.share == b.share

    @classmethod
    def decode(cls, M, Z, gap=128):
        return (M.share - Z.share) / gap


class SecureSchnorrGroupElement(SecureFiniteGroup):
    """Common base class for secure (secret-shared) Schnorr group elements."""

    __slots__ = ()

    def __init__(self, value=None):
        """Ensure value is of secure field type."""
        if isinstance(value, self.group):
            value = value.value
        secfld = self.sectype
        if not isinstance(value, secfld):
            value = secfld(value)
        super().__init__(value)

    def set_share(self, value):
        self.share.set_share(value.share)

    @classmethod
    def operation(cls, a, b, /):
        return cls(a.share * b.share)

    @classmethod
    def inversion(cls, a, /):
        return cls(1/a.share)

    @classmethod
    def equality(cls, a, b, /):
        return a.share == b.share

    @classmethod
    def decode(cls, M, Z):
        g = cls.group.generator
        h = cls.group.identity
        x = [h]
        for _ in range(15):  # TODO: get rid of hard-coded 15 bound
            h = cls.group.operation(h, g)
            x.append(h)
        m = runtime.find(x, M, bits=False)
        return m  # M = g^m


class SecureEllipticCurvePoint(SecureFiniteGroup):
    """Common base class for secure (secret-shared) elliptic curve points."""

    __slots__ = ()

    def __init__(self, value=None):
        """Ensure all coefficients are of secure field type.

        Enforce value is a tuple.
        """
        n = len(self.group.identity.value)
        if value is None:
            value = [None] * n
        elif isinstance(value, self.group):
            value = value.value
        else:
            if not (isinstance(value, (tuple, list)) and len(value) == n):
                raise ValueError(f'tuple/list of length {n} required')

        secfld = self.sectype
        value = tuple(secfld(a) if not isinstance(a, secfld) else a for a in value)
        super().__init__(value)

    def set_share(self, value):
        for a, b in zip(self.share, value):
            a.set_share(b.share)

    def __getitem__(self, key):  # NB: no set_item to prevent mutability
        return self.share[key]

    @classmethod
    def operation(cls, a, b, /):
        group = cls.group
        c = group.operation(group(a.share, check=False), group(b.share, check=False))
        return cls(c)

    @classmethod
    def inversion(cls, a, /):
        group = cls.group
        c = group.inversion(group(a.share, check=False))
        return cls(c)

    def normalize(self):
        cls = type(self)
        group = cls.group
        if issubclass(group, fg.WeierstrassProjective):
            field = group.field
            x, y, z = self
            zis0 = z == 0
            z_inv = 1 / (z + zis0)
            c = zis0.if_else([field(0), field(1)], [x, y])
            c = runtime.scalar_mul(z_inv, c)
            return cls(c + [1 - zis0])

        c = group(self.share, check=False).normalize()
        return cls(c)

    @classmethod
    def equality(cls, a, b, /):
        return runtime.all(u == v for u, v in zip(a.normalize(), b.normalize()))

    @classmethod
    def decode(cls, M, Z, gap=256):
        return (M.normalize()[0] - Z.normalize()[0]) / gap


@asyncoro.mpc_coro
async def _divmod(a, b):  # TODO: cleanup and integrate this function in mpyc.runtime
    """Secure integer division divmod(a, b) via NR."""
    secint = type(a)
    await runtime.returnType(secint, 2)
    secfxp = runtime.SecFxp(2*secint.bit_length+2)
    a1, b1 = runtime.convert([a, b], secfxp)
    q = a1 / b1
    q = runtime.convert(q, secint)
    r = a - b * q
    q, r = (r < 0).if_else([q - 1, r + b], [q, r])  # correction using one <
#    q, r = runtime.if_else(r >= b, [q + 1, r - b], [q, r])  # correction using one <
    assert await runtime.output(a == b * q + r), await runtime.output([q, r, a, b])
    assert await runtime.output(0 <= r), await runtime.output([q, r, a, b])
    assert await runtime.output(r < b), await runtime.output([q, r, a, b])
    return q, r


@asyncoro.mpc_coro
async def _bit_length(a):  # TODO: integrate this function in mpyc.runtime
    stype = type(a)  # TODO: extend to case a < 0 ?
    await runtime.returnType(stype, 2)

    Zp = stype.field
    l = stype.bit_length
    r_bits = await runtime.random_bits(Zp, l)
    r_modl = 0
    for r_i in reversed(r_bits):
        r_modl <<= 1
        r_modl += r_i.value
    r_divl = runtime._random(Zp, 1<<runtime.options.sec_param)
    if runtime.options.no_prss:
        r_divl = (await r_divl)[0]
    r_divl = r_divl.value
    a = await runtime.gather(a)
    c = await runtime.output(a + ((1<<l) + (r_divl << l) + r_modl))
    c = c.value % (1<<l)

    c_bits = [(c >> i) & 1 for i in range(l)]
    d_bits = [stype((1 - r_bits[i] if c_bits[i] else r_bits[i]).value) for i in range(l)]
    h_bits = runtime.schur_prod([stype(1 - r_bits[i]) for i in range(l-1) if not c_bits[i]],
                                [d_bits[i+1] for i in range(l-1) if not c_bits[i]])
    for i in range(l-2, -1, -1):
        if not c_bits[i]:
            d_bits[i+1] = h_bits.pop()

    # locate most significant 1 in d_bits (assumes a >=0)
    d_bits.reverse()
    k, k2 = runtime.find(d_bits, 1, cs_f=lambda b, i: (i+b, (b+1) << i))
    k, k2 = l-1 - k, (1<<l-1) / k2  # k2 = 2**k

    k_u = runtime.unit_vector(k, l)  # 0<=k<l assumed
    k_u = await runtime.gather(k_u)
    psums = list(itertools.accumulate(k_u))
    pp = await runtime.schur_prod(psums, [c_bits[i] - r_bits[i] for i in range(l)])
    for i in range(l):
        r_bits[i] += pp[i]

    s_sign = (await runtime.random_bits(Zp, 1, signed=True))[0].value
    e = [None] * (l+1)
    sumXors = 0
    for i in range(l-1, -1, -1):
        c_i = c_bits[i]
        r_i = r_bits[i].value
        e[i] = Zp(s_sign + r_i - c_i + 3*sumXors)
        sumXors += 1 - r_i if c_i else r_i
    e[l] = Zp(s_sign - 1 + 3*sumXors)
    g = await runtime.is_zero_public(stype(runtime.prod(e)))
    z = Zp(1 - s_sign if g else 1 + s_sign)/2
    return k + 1 - z, k2 * stype(2 - z)  # position + 1 is bit length


class SecureClassGroupForm(SecureFiniteGroup):
    """Common base class for secure (secret-shared) class group forms."""

    __slots__ = ()

    def __init__(self, value=None):
        """Ensure all coefficients are of secure type.

        Enforce value is a tuple.
        """
        n = 3
        if value is None:
            value = [None] * n
        elif isinstance(value, self.group):
            value = value.value
        else:
            if not (isinstance(value, (tuple, list)) and len(value) == n):
                raise ValueError(f'tuple/list of length {n} required')

        secint = self.sectype
        value = tuple(secint(a) if not isinstance(a, secint) else a for a in value)
        super().__init__(value)

    def set_share(self, value):
        for a, b in zip(self.share, value):
            a.set_share(b.share)

    def __getitem__(self, key):  # NB: no set_item to prevent mutability
        return self.share[key]

    @classmethod
    def _reduce(cls, f):
        """Secure reduction of given form.

        Reduction algorithm avoiding secure integer division in the main loop
        based on Algorithm 3 from "A New GCD Algorithm for Quadratic Number
        Rings with Unique Factorization" by Agarwal and Frandsen, LATIN 2006,
        LNCS 3887, pp. 30-42, Springer.
        See https://users-cs.au.dk/gudmund/Documents/38870030.pdf.
        """

        def right_action_Tm(m, f):
            a, b, c = f
            return [a, 2*m * a + b, m * (m * a + b) + c]

        a, b, c = f
        secint = cls.sectype
        len_b = secint.bit_length
        for _ in range((cls.group.discriminant.bit_length()+1)//2):
            sgn_b = 1 - 2*runtime.sgn(b, l=len_b, LT=True)
            len_b -= 1
            sizeb = _bit_length(sgn_b * b)
            sizea = _bit_length(a)
            pow2 = sizeb[1] / sizea[1] / 2  # 2**(len(b) - len(a) - 1)
            m = -sgn_b * pow2
            a, b, c = (sgn_b * b > 2*a).if_else(right_action_Tm(m, (a, b, c)), [a, b, c])
            a, b, c = (a > c).if_else([c, -b, a], [a, b, c])

        m, _ = _divmod(a - b, 2*a)
        a, b, c = right_action_Tm(m, (a, b, c))
        a, b, c = (a > c).if_else([c, -b, a], [a, b, c])
        b = ((b < 0) * (a == c)).if_else(-b, b)
        b = (b == -a).if_else(-b, b)
        return a, b, c

    # See Henri Cohen's book "A Course in Computational Algebraic Number Theory", Chapter 5.
    @classmethod
    def operation(cls, f1, f2, /):  # Cohen: Algorithm 5.4.7 (Shanks 1969)
        a1, b1, _ = f1
        a2, b2, c2 = f2
        s = (b1 + b2)/2
        _d, _, y1 = runtime.gcdext(a1, a2, l=type(a1).bit_length//2)
        d, x2, y2 = runtime.gcdext(s, _d, l=type(a1).bit_length//2)
        v1 = a1 / d
        v2 = a2 / d
        # Set r = y1 y2 (s - b2) - x2 c2 (mod v1) in two steps:
        r = _divmod(_divmod(y1*y2, v1)[1] * (s - b2) - x2 * c2, v1)[1]
        a3 = v1*v2
        b3 = b2 + 2*v2*r
        c3 = (b3**2 - cls.group.discriminant) / (4*a3)
        return cls(cls._reduce((a3, b3, c3)))

    @classmethod
    def operation2(cls, f, /):
        a, b, c = f  # NB: a>0, b!=0, and gcd(a,b)=1 because -discriminant is prime
        x2 = runtime.inverse(b, a, l=type(a).bit_length//2)
        _, r = _divmod(x2*c, a)
        a2 = a**2
        b2 = b - 2*a*r
        c2 = (b2**2 - cls.group.discriminant) / (4*a2)
        return cls(cls._reduce((a2, b2, c2)))

    @classmethod
    def inversion(cls, f, /):
        a, b, c = f
        b = ((b != a) * (a != c)).if_else(-b, b)
        return cls((a, b, c))

    @classmethod
    def equality(cls, f1, f2, /):
        v0 = f1.share[0] == f2.share[0]
        v1 = f1.share[1] == f2.share[1]
        return v0 * v1  # TODO: optimize batch test eq. with random r in field

    @classmethod
    def decode(cls, M, Z):
        gap = cls.group.gap
        return (M.share[0] - Z.share[0]) / gap


@functools.cache
def SecGrp(group):
    """Secure version of given finite group."""
    if issubclass(group, fg.SymmetricGroupElement):
        base = SecureSymmetricGroupElement
        sectype = runtime.SecFld(min_order=group.degree)
    elif issubclass(group, fg.QuadraticResidue):
        base = SecureQuadraticResidue
        sectype = runtime.SecFld(2*group.order+1)
    elif issubclass(group, fg.SchnorrGroupElement):
        base = SecureSchnorrGroupElement
        sectype = runtime.SecFld(group.field.order)
    elif issubclass(group, fg.EllipticCurvePoint):
        base = SecureEllipticCurvePoint
        sectype = runtime.SecFld(group.field.order)
        assert group.oblivious
    elif issubclass(group, fg.ClassGroupForm):
        base = SecureClassGroupForm
        sectype = runtime.SecInt(group.bit_length + 3)  # TODO: check bit length (+2 is not enough)
    else:
        raise NotImplementedError

    name = f'SecGrp({group.__name__})'
    secgrp = type(name, (base,), {'__slots__': ()})
    secgrp.__doc__ = 'Class of secret-shared finite group elements.'
    secgrp.group = group
    secgrp.sectype = sectype
    secgrp.identity = secgrp(group.identity)  # TODO: how much is gained by avoiding secgrp() here?
    globals()[name] = secgrp  # NB: exploit (almost) unique name dynamic SecureGroup type
    return secgrp


def _toSecGrpFunc(GroupFunc):
    """Create secure group constructor for given GroupFunc."""
    name = f'Sec{GroupFunc.__name__}'
    sig = inspect.signature(GroupFunc)
    # docstring based on GroupFunc's signature and docstring
    doc = f'''Call {name}(...) is equivalent to SecGrp({GroupFunc.__name__}(...)),
    returning secure version of {GroupFunc.__name__} from mpyc.fingroups.

    {GroupFunc.__name__}{sig}:

    {GroupFunc.__doc__}
'''

    def SecGrpFunc(*args, **kwargs):
        return SecGrp(GroupFunc(*args, **kwargs))
    SecGrpFunc.__name__ = name
    SecGrpFunc.__doc__ = doc
    SecGrpFunc.__signature__ = sig
    globals()[name] = SecGrpFunc


_toSecGrpFunc(fg.SymmetricGroup)     # make SecSymmetricGroup as secure SymmetricGroup version
_toSecGrpFunc(fg.QuadraticResidues)  # make SecQuadraticResidues as secure QuadraticResidues version
_toSecGrpFunc(fg.SchnorrGroup)       # make SecSchnorrGroup as secure SchnorrGroup version
_toSecGrpFunc(fg.EllipticCurve)      # make SecEllipticCurve as secure EllipticCurve version
_toSecGrpFunc(fg.ClassGroup)         # make SecClassGroup as secure ClassGroup version

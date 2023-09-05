"""Module for information-theoretic and pseudorandom threshold secret sharing.

Threshold secret sharing assumes secure channels for communication.
Pseudorandom secret sharing (PRSS) allows one to share pseudorandom
secrets without any communication, as long as the parties
agree on a (unique) common public input for each secret.

PRSS relies on parties having agreed upon the keys for a pseudorandom
function (PRF).
"""

__all__ = ['random_split', 'recombine', 'pseudorandom_share', 'pseudorandom_share_zero',
           'np_random_split', 'np_recombine', 'np_pseudorandom_share', 'np_pseudorandom_share_0',
           'PRF']

from math import prod
import functools
from hashlib import shake_128
import secrets
from mpyc.numpy import np


def random_split(field, s, t, m):
    """Split each secret given in s into m random Shamir shares.

    The (maximum) degree for the Shamir polynomials is t, 0 <= t < m.
    Return matrix of shares, one row per party.
    """
    p = field.modulus
    order = field.order
    _0 = type(p)(0)  # _0 is int(0) or gfpx.Polynomial(0)
    shares = [[None] * len(s) for _ in range(m)]
    T_is_field = isinstance(s[0], field)  # all elts assumed of same type
    for h, s_h in enumerate(s):
        if T_is_field:
            s_h = s_h.value
        c = [secrets.randbelow(order) for _ in range(t)]
        # polynomial f(X) = s[h] + c[t-1] X + c[t-2] X^2 + ... + c[0] X^t
        for i1 in range(1, m+1):
            y = _0
            for c_j in c:
                y = (y + c_j) * i1
            shares[i1-1][h] = (y + s_h) % p
    return shares


def np_random_split(field, s, t, m):
    """Split each secret given in s into m random Shamir shares.

    The (maximum) degree for the Shamir polynomials is t, 0 <= t < m.
    Return matrix of shares, one row per party.
    """
    p = field.modulus
    tp = type(p)  # int or gfpx.Polynomial
    if isinstance(s, field.array):
        s = s.value
    n = len(s)
    _randbelow = secrets.randbelow
    order = field.order
    C = np.fromiter((_randbelow(order) for _ in range(t * n)), 'O', count=t * n).reshape(t, n)
    V = np.vander(np.array([tp(i) for i in range(1, m+1)], dtype='O'), N=t+1, increasing=True)
    # NB: each entry in first column of V is a 1 of type int (also if tp is gfpx.Polynomial)
    shares = (V @ np.concatenate((s.reshape(1, n), C))) % p
    return shares


@functools.cache
def _recombination_vector(field, xs, x_r):
    """Compute and store a recombination vector.

    A recombination vector depends on the field, the x-coordinates xs
    of the shares and the x-coordinate x_r of the recombination point.
    """
    xs = [field(x).value for x in xs]
    x_r = field(x_r).value
    vector = []
    for i, x_i in enumerate(xs):
        coefficient_n = field(1)
        coefficient_d = field(1)
        for j, x_j in enumerate(xs):
            if i != j:
                coefficient_n *= (x_r - x_j)
                coefficient_d *= (x_i - x_j)
        vector.append((coefficient_n / coefficient_d).value)
    return vector


def recombine(field, points, x_rs=0):
    """Recombine shares given by points into secrets.

    Recombination is done for x-coordinates x_rs.
    """
    xs, shares = list(zip(*points))
    if not isinstance(x_rs, list):
        x_rs = (x_rs,)
    width = len(x_rs)
    vector = [_recombination_vector(field, xs, x_r) for x_r in x_rs]

    n = len(shares[0])
    sums = [[0] * n for _ in range(width)]
    T_is_field = isinstance(shares[0][0], field)  # all elts assumed of same type
    for i, share_i in enumerate(shares):
        for h in range(n):
            s = share_i[h]
            if T_is_field:
                s = s.value
            # type(s) is int or gfpx.Polynomial
            for r in range(width):
                sums[r][h] += s * vector[r][i]
    if T_is_field:
        for r in range(width):
            for h in range(n):
                sums[r][h] = field(sums[r][h])
    if isinstance(x_rs, tuple):
        sums = sums[0]
    return sums


def np_recombine(field, points, x_rs=0):
    """Recombine shares given by points into secrets.

    Recombination is done for x-coordinates x_rs.
    """
    xs, shares = list(zip(*points))
    if not isinstance(x_rs, list):
        x_rs = (x_rs,)
    vector = np.array([_recombination_vector(field, xs, x_r) for x_r in x_rs], dtype='O')
    shares = field.array(shares)
    sums = vector @ shares
    if isinstance(x_rs, tuple):
        sums = sums[0]
    return sums


@functools.cache
def _f_S_i(field, m, i, S):
    """Compute and store polynomial f_S evaluated for party i.

    Polynomial f_S is 1 at 0 and 0 for all parties j outside S."""
    points = [(0, [1])] + [(x+1, [0]) for x in range(m) if x not in S]
    return recombine(field, points, i+1)[0]


def pseudorandom_share(field, m, i, prfs, uci, n):
    """Return pseudorandom Shamir shares for party i for n random numbers.

    The shares are based on the pseudorandom functions for party i,
    given in prfs, which maps subsets of parties to PRF instances.
    Input uci is used to evaluate the PRFs on a unique common input.
    """
    sums = [0] * n
    # iterate over (m-1 choose t) subsets for degree t.
    for S, prf_S in prfs.items():
        f_S_i = _f_S_i(field, m, i, S)
        prl = prf_S(uci, n)
        for h in range(n):
            sums[h] += prl[h] * f_S_i
    for h in range(n):
        sums[h] = field(sums[h])
    return sums


def np_pseudorandom_share(field, m, i, prfs, uci, n):
    """Return pseudorandom Shamir shares for party i for n random numbers.

    The shares are based on the pseudorandom functions for party i,
    given in prfs, which maps subsets of parties to PRF instances.
    Input uci is used to evaluate the PRFs on a unique common input.
    """
    # sum over (m-1 choose t) subsets for degree t.
    s = sum(prf_S(uci, (n,)) * _f_S_i(field, m, i, S)
            for S, prf_S in prfs.items())
    return field.array(s)


def pseudorandom_share_zero(field, m, i, prfs, uci, n):
    """Return pseudorandom Shamir shares for party i for n sharings of 0.

    The shares are based on the pseudorandom functions for party i,
    given in prfs, which maps subsets of parties to PRF instances.
    Input uci is used to evaluate the PRFs on a unique common input.
    """
    _0 = type(field.modulus)(0)  # _0 is int(0) or gfpx.Polynomial(0)
    i1 = i + 1
    sums = [0] * n
    # iterate over (m-1 choose t) subsets for degree t.
    for S, prf_S in prfs.items():
        f_S_i = _f_S_i(field, m, i, S)
        d = m - len(S)
        prl = prf_S(uci, n * d)
        for h in range(n):
            y = _0
            for j in range(d):
                y = (y + prl[h * d + j]) * i1
            sums[h] += y * f_S_i
    for h in range(n):
        sums[h] = field(sums[h])
    return sums


def np_pseudorandom_share_0(field, m, i, prfs, uci, n):
    """Return pseudorandom Shamir shares for party i for n sharings of 0.

    The shares are based on the pseudorandom functions for party i,
    given in prfs, which maps subsets of parties to PRF instances.
    Input uci is used to evaluate the PRFs on a unique common input.
    """
    vtype = type(field.modulus)  # int or gfpx.Polynomial
    d = m - len(next(iter(prfs.keys())))  # subsets all of same size m-t
    i1s = np.array([vtype(i+1)**j for j in range(1, d+1)], dtype='O')
    sums = vtype(0)
    # iterate over (m-1 choose t) subsets for degree t.
    for S, prf_S in prfs.items():
        f_S_i = _f_S_i(field, m, i, S)
        prl = prf_S(uci, (n, d))
        sums += (prl @ i1s) * f_S_i
    return field.array(sums)


class PRF:
    """A pseudorandom function (PRF).

    A PRF is determined by a secret key and a public maximum.
    """

    def __init__(self, key, bound):
        """Create a PRF determined by the given key and (upper) bound.

        The key is given as a byte string.
        Output values will be in range(bound).
        """
        self.key = key
        self.max = bound
        self.byte_length = ((bound - 1).bit_length() + 7) // 8
        if bound & (bound - 1):  # no power of 2
            self.byte_length += len(self.key)

    def __call__(self, s, n=None):
        """Return pseudorandom number(s) in range(self.max) for given input bytes s.

        The numbers are a deterministic function of self.key and input s.
        If n is None, a single number is returned.
        If n is a (nonnegative) integer, a length-n list is returned.
        If n is a shape, a shape-n array is returned.
        """
        if isinstance(n, tuple):
            shape = n
            n = prod(shape)
        else:
            shape = None
        n_ = 1 if n is None else n
        if n_ == 0:
            iterable = ()
        elif not (l := self.byte_length):
            iterable = (0 for _ in range(n_))
        else:
            dk = shake_128(self.key + s).digest(n_ * l)
            byteorder = 'little'
            from_bytes = int.from_bytes  # cache
            bound = self.max
            iterable = (from_bytes(dk[i:i + l], byteorder) % bound for i in range(0, n_ * l, l))
        if shape is None:
            x = list(iterable)
        else:
            x = np.fromiter(iterable, object, count=n_).reshape(shape)
        return x[0] if n is None else x

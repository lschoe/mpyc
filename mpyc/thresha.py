"""Module for information-theoretic and pseudorandom threshold secret sharing.

Threshold secret sharing assumes secure channels for communication.
Pseudorandom secret sharing (PRSS) allows one to share pseudorandom
secrets without any communication, as long as the parties
agree on a (unique) common public input for each secret.

PRSS relies on parties having agreed upon the keys for a pseudorandom
function (PRF).
"""

__all__ = ['random_split', 'recombine', 'pseudorandom_share',
           'pseudorandom_share_zero', 'PRF']

import functools
import hashlib
import secrets


def random_split(s, t, m):
    """Split each secret given in s into m random Shamir shares.

    The (maximum) degree for the Shamir polynomials is t, 0 <= t < n.
    Return matrix of shares, one row per party.
    """
    field = type(s[0])
    p = field.modulus
    order = field.order
    T = type(p)  # T is int or gfpx.Polynomial
    n = len(s)
    shares = [[None] * n for _ in range(m)]
    for h in range(n):
        c = [secrets.randbelow(order) for _ in range(t)]
        # polynomial f(X) = s[h] + c[t-1] X + c[t-2] X^2 + ... + c[0] X^t
        for i in range(m):
            y = 0 if T is int else T(0)
            for c_j in c:
                y += c_j
                y *= i+1
            shares[i][h] = (y + s[h].value) % p
    return shares


@functools.lru_cache(maxsize=None)
def _recombination_vector(field, xs, x_r):
    """Compute and store a recombination vector.

    A recombination vector depends on the field, the x-coordinates xs
    of the shares and the x-coordinate x_r of the recombination point.
    """
    xs = [field(x).value for x in xs]
    x_r = field(x_r).value
    vector = []
    for i, x_i in enumerate(xs):
        coefficient_d = field(1)
        coefficient_n = field(1)
        for j, x_j in enumerate(xs):
            if i != j:
                coefficient_d *= (x_r - x_j)
                coefficient_n *= (x_i - x_j)
        vector.append((coefficient_d / coefficient_n).value)
    return vector


def recombine(field, points, x_rs=0):
    """Recombine shares given by points into secrets.

    Recombination is done for x-coordinates x_rs.
    """
    xs, shares = list(zip(*points))
    if not isinstance(x_rs, list):
        x_rs = (x_rs,)
    n = len(shares[0])
    width = len(x_rs)
    T_is_field = isinstance(shares[0][0], field)  # all elts assumed of same type
    vector = [_recombination_vector(field, xs, x_r) for x_r in x_rs]
    sums = [[0] * n for _ in range(width)]
    for i, share_i in enumerate(shares):
        for h in range(n):
            s = share_i[h]
            if T_is_field:
                s = s.value
            # type(s) is int or gfpx.Polynomial
            for r in range(width):
                sums[r][h] += s * vector[r][i]
    for r in range(width):
        for h in range(n):
            sums[r][h] = field(sums[r][h])
    if isinstance(x_rs, tuple):
        return sums[0]

    return sums


@functools.lru_cache(maxsize=None)
def _f_S_i(field, m, i, S):
    """Compute and store polynomial f_S evaluated for party i.

    Polynomial f_S is 1 at 0 and 0 for all parties j outside S."""
    points = [(0, [1])] + [(x+1, [0]) for x in range(m) if x not in S]
    return recombine(field, points, i+1)[0].value


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


def pseudorandom_share_zero(field, m, i, prfs, uci, n):
    """Return pseudorandom Shamir shares for party i for n sharings of 0.

    The shares are based on the pseudorandom functions for party i,
    given in prfs, which maps subsets of parties to PRF instances.
    Input uci is used to evaluate the PRFs on a unique common input.
    """
    T = type(field.modulus)  # T is int or T is gfpx.Polynomial
    sums = [0] * n
    # iterate over (m-1 choose t) subsets for degree t.
    for S, prf_S in prfs.items():
        f_S_i = _f_S_i(field, m, i, S)
        d = m - len(S)
        prl = prf_S(uci, n * d)
        for h in range(n):
            y = 0 if T is int else T(0)
            for j in range(d):
                y += prl[h * d + j]
                y *= i+1
            sums[h] += y * f_S_i
    for h in range(n):
        sums[h] = field(sums[h])
    return sums


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
        """Return a number or length-n list of numbers in range(self.max) for input bytes s."""
        if n == 0:
            return []

        n_ = 1 if n is None else n
        l = self.byte_length
        if not l:
            x = [0] * n_
        else:
            dk = hashlib.pbkdf2_hmac('sha1', self.key, s, 1, n_ * l)
            byteorder = 'little'
            from_bytes = int.from_bytes  # cache
            bound = self.max
            x = [from_bytes(dk[i:i + l], byteorder) % bound for i in range(0, n_ * l, l)]
        return x[0] if n is None else x

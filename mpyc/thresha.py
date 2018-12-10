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
    T = type(p) # T is int or gf2x.Polynomial
    n = len(s)
    shares = [[None] * n for _ in range(m)]
    for h in range(n):
        c = [secrets.randbelow(order) for _ in range(t)]
        # polynomial f(x) = s[h] + c[t-1] x + c[t-2] x^2 + ... + c[0] x^t
        for i in range(m):
            y = 0 if T is int else T(0)
            for c_j in c:
                y += c_j
                y *= i + 1
            shares[i][h] = (y + s[h].value) % p
    return shares

#Cache recombination vectors, which depend on the field and
#the x-coordinates of the shares and the recombination point.
_recombination_vectors = {}

def recombine(field, points, x_rs=0):
    """Recombine shares given by points into secrets.

    Recombination is done for x-coordinates x_rs.
    """
    xs, shares = list(zip(*points))
    if not isinstance(x_rs, list):
        x_rs = (x_rs,)
    width = len(x_rs)
    vector = [None] * width
    for r, x_r in enumerate(x_rs):
        try:
            vector[r] = _recombination_vectors[(field, xs, x_r)]
        except KeyError:
            vector[r] = []
            for i, x_i in enumerate(xs):
                x_i = field(x_i)
                coefficient = field(1)
                for j, x_j in enumerate(xs):
                    x_j = field(x_j)
                    if i != j:
                        coefficient *= (x_r - x_j) / (x_i - x_j)
                vector[r].append(coefficient.value)
            _recombination_vectors[(field, xs, x_r)] = vector[r]
    m = len(shares)
    n = len(shares[0])
    sums = [[0] * n for _ in range(width)]
    T_is_field = isinstance(shares[0][0], field) # all elts assumed of same type
    for i in range(m):
        for h in range(n):
            s = shares[i][h]
            if T_is_field:
                s = s.value
            # type(s) is int or gf2x.Polynomial
            for r in range(width):
                sums[r][h] += s * vector[r][i]
    for r in range(width):
        for h in range(n):
            sums[r][h] = field(sums[r][h])
    if isinstance(x_rs, tuple):
        return sums[0]

    return sums

#Cache coefficients used to construct shares, which depend on the field,
#the party concerned, and the subset.
_f_in_i_cache = {}

def pseudorandom_share(field, m, i, prfs, uci, n):
    """Return pseudorandom Shamir shares for party i for n random numbers.

    The shares are based on the pseudorandom functions for party i,
    given in prfs, which maps subsets of parties to PRF instances.
    Input uci is used to evaluate the PRFs on a unique common input.
    """
    s = str(uci)
    sums = [0] * n
    # iterate over (m-1 choose t) subsets for degree t.
    for subset, prf in prfs.items():
        try:
            f_in_i = _f_in_i_cache[(field, i, subset)]
        except KeyError:
            complement = frozenset(range(m)) - subset
            points = [(0, [1])] + [(x + 1, [0]) for x in complement]
            f_in_i = recombine(field, points, i + 1)[0].value
            _f_in_i_cache[(field, i, subset)] = f_in_i
        prl = prf(s, n)
        for h in range(n):
            sums[h] += prl[h] * f_in_i
    for h in range(n):
        sums[h] = field(sums[h])
    return sums

def pseudorandom_share_zero(field, m, i, prfs, uci, n):
    """Return pseudorandom Shamir shares for party i for n sharings of 0.

    The shares are based on the pseudorandom functions for party i,
    given in prfs, which maps subsets of parties to PRF instances.
    Input uci is used to evaluate the PRFs on a unique common input.
    """
    s = str(uci)
    sums = [0] * n
    # iterate over (m-1 choose t) subsets for degree t.
    for subset, prf in prfs.items():
        try:
            f_in_i = _f_in_i_cache[(field, i, subset)]
        except KeyError:
            complement = frozenset(range(m)) - subset
            points = [(0, [1])] + [(x + 1, [0]) for x in complement]
            f_in_i = recombine(field, points, i + 1)[0].value
            _f_in_i_cache[(field, i, subset)] = f_in_i
        d = m - len(subset)
        prl = prf(s, n * d)
        T = type(field.modulus) # T is int or T is gf2x.Polynomial
        for h in range(n):
            y = 0 if T is int else T(0)
            for j in range(d):
                y += prl[h * d + j]
                y *= i + 1
            sums[h] += y * f_in_i
    for h in range(n):
        sums[h] = field(sums[h])
    return sums

class PRF:
    """A pseudorandom function (PRF) with 128-bit keys.

    A PRF is determined by a secret key and a public maximum.
    """

    def __init__(self, key, bound):
        """Create a PRF determined by the given key and (upper) bound.

        The key is a hex string, whereas bound is a number.
        Output values will be in range(bound).
        """
        self.key = int(key, 16).to_bytes(16, byteorder='little') #128-bit key
        self.max = bound
        self.byte_length = len(self.key) + ((bound-1).bit_length() + 7) // 8

    def __call__(self, s, n=None):
        """Return a number or list of numbers in range(self.max) for input string s."""
        n_ = n if n else 1
        l = self.byte_length
        dk = hashlib.pbkdf2_hmac('sha1', self.key, s.encode(), 1, n_ * l)
        x = [int.from_bytes(dk[i * l: (i+1) * l], byteorder='little') % self.max for i in range(n_)]
        return x if n else x[0]

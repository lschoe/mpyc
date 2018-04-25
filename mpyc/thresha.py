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

def random_split(s, d, n):
    """Split each secret given in s into n random Shamir shares.

    The (maximum) degree for the Shamir polynomials is d, 0 <= d < n.
    Return matrix of shares, one row per party.
    """
    p = s[0].modulus
    m = len(s)
    shares = [[None] * m for _ in range(n)]
    for h in range(m):
        c = [secrets.randbelow(p) for _ in range(d)]
        # polynomial f(x) = s[h] + c[0] x + c[1] x^2 + ... + c[d-1] x^d
        for i in range(n):
            y = 0
            for c_k in c:
                y += c_k
                y *= i + 1
            shares[i][h] = (y + s[h].value) % p
    return shares

#Cache recombination vectors, which depend on the x-coordinates of the shares
#and the recombination point.
_recombination_vectors = {}

def recombine(field, points, x_rs=0):
    """Recombine shares given by points into secrets.

    Recombination is done for x-coordinates x_rs.
    """
    xs, shares = list(zip(*points))
    if not isinstance(x_rs, list):
        x_rs = (x_rs,)
    vector = [None] * len(x_rs)
    for r, x_r in enumerate(x_rs):
        try:
            vector[r] = _recombination_vectors[(xs, x_r)] #field ?
        except KeyError:
            vector[r] = []
            x_r = field(x_r)
            for i, x_i in enumerate(xs):
                x_i = field(x_i)
                coefficient = field(1)
                for j, x_j in enumerate(xs):
                    x_j = field(x_j)
                    if i != j:
                        coefficient *= (x_r - x_j) / (x_i - x_j)
                vector[r].append(coefficient.value)
            _recombination_vectors[(xs, x_r.value)] = vector[r]
    m = len(shares[0])
    sums = [[0] * m for _ in range(len(x_rs))]
    for i in range(len(shares)):
        for h in range(m):
            s = shares[i][h]
            if not isinstance(s, int):
                s = s.value
            for r in range(len(sums)):
                sums[r][h] += s * vector[r][i]
    for h in range(m):
        for r in range(len(sums)):
            sums[r][h] = field(sums[r][h])
    if isinstance(x_rs, tuple):
        return sums[0]
    else:
        return sums

#Cache coefficients used to construct shares, which depend on the field,
#the party concerned, and the subset.
_f_in_i_cache = {}

def pseudorandom_share(field, n, i, prfs, uci, m):
    """Return pseudorandom Shamir shares for party i for m random numbers.

    The shares are based on the pseudorandom functions for party i,
    given in prfs, which maps subsets of parties to PRF instances.
    Input uci is used to evaluate the PRFs on a unique common input.
    """
    s = str(uci)
    sums = [0] * m
    # iterate over (n-1 choose d) subsets for degree d.
    for subset, prf in prfs.items():
        try:
            f_in_i = _f_in_i_cache[(field, i, subset)]
        except KeyError:
            complement = frozenset(range(n)) - subset
            points = [(0, [1])] + [(x + 1, [0]) for x in complement]
            f_in_i = recombine(field, points, i + 1)[0].value
            _f_in_i_cache[(field, i, subset)] = f_in_i
        for h in range(m):
            sums[h] += prf(s + str(h)) * f_in_i
    for h in range(m):
        sums[h] = field(sums[h])
    return sums

def pseudorandom_share_zero(field, n, i, prfs, uci, m):
    """Return pseudorandom Shamir shares for party i for m sharings of 0.

    The shares are based on the pseudorandom functions for party i,
    given in prfs, which maps subsets of parties to PRF instances.
    Input uci is used to evaluate the PRFs on a unique common input.
    """
    s = str(uci)
    sums = [0] * m
    # iterate over (n-1 choose d) subsets for degree d.
    for subset, prf in prfs.items():
        try:
            f_in_i = _f_in_i_cache[(field, i, subset)]
        except KeyError:
            complement = frozenset(range(n)) - subset
            points = [(0, [1])] + [(x + 1, [0]) for x in complement]
            f_in_i = recombine(field, points, i + 1)[0].value
            _f_in_i_cache[(field, i, subset)] = f_in_i
        d = n - len(subset)
        for h in range(m):
            y = 0
            for k in range(d):
                y += prf(s + str((h, k)))
                y *= i + 1
            sums[h] += y * f_in_i
    for h in range(m):
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

    def __call__(self, s):
        """Return a number in range(self.max) for input string s."""
        dk = hashlib.pbkdf2_hmac('sha1', self.key, s.encode(), 1, self.byte_length)
        return int.from_bytes(dk, byteorder='little') % self.max

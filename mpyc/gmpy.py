"""This module collects all gmpy2 functions used by MPyC.

Plus a function for factoring prime powers,
and a function for rational reconstruction.

Stubs of limited functionality and efficiency are provided
in case the gmpy2 package is not available.
"""

import os
import math


def factor_prime_power(x):  # TODO: move this to a separate math/number theory module?
    """Return p and d for a prime power x = p**d."""
    if x <= 1:
        raise ValueError('number not a prime power')

    k = 10
    # test whether p is below 2**k, for positive k
    p = 2
    while p < 1<<k:
        if x % p == 0:
            d = 0
            while x > 1:
                x, r = divmod(x, p)
                if r == 0:
                    d += 1
                else:
                    raise ValueError('number not a prime power')

            return int(p), d

        p = next_prime(p)

    # find prime factors of d
    p, d = x, 1
    while is_square(p):
        p, d = isqrt(p), 2*d
    e = 3
    while k * e <= p.bit_length():
        w, b = iroot(p, e)
        if b:
            p, d = w, e * d
        else:
            e = next_prime(e)

    if is_prime(p):
        return int(p), int(d)

    raise ValueError('number not a prime power')


def ratrec(x, y, N=None, D=None):
    """Return rational reconstruction (n, d) of x modulo y.
    That is,  n/d = x (mod y) with -N <= n <= N and 0 < d <= D,
    provided 2*N*D < y.

    Default N=D=None will set both N and D to sqrt(y/2) approximately.
    """
    if N is None:
        if D is None:
            D = max(1, isqrt((y-1)//2))
        N = (y-1) // (2*D)
    elif D is None:
        D = (y-1) // (2*N) if N else 1
    if N < 0 or D <= 0 or 2 * N * D >= y:
        raise ValueError('rational reconstruction not supported')

    # Wang's algorithm, assuming N >= 0, D > 0, 2*N*D < y
    n0, n = x, y
    d0, d = 1, 0
    while n > N:
        n0, (q, n) = n, divmod(n0, n)
        d0, d = d, d0 - q * d
    if d < 0:
        n, d = -n, -d
    if d <= D and math.gcd(n, d) == 1:
        return n, d

    raise ValueError('rational reconstruction not possible')


try:
    if os.getenv('MPYC_NOGMPY') == '1':
        raise ImportError  # stubs will be loaded

    from gmpy2 import (is_prime, next_prime, powmod, gcdext, invert,
                       legendre, is_square, isqrt, iroot)
except ImportError:
    # load stubs, if MPYC_NOGMPY is set, or if gmpy2 import fails
    import random

    def is_prime(x, n=25):
        """Return True if x is probably prime, else False if x is
        definitely composite, performing up to n Miller-Rabin
        primality tests.
        """
        if x <= 2 or x%2 == 0:
            return x == 2

        # odd x >= 3
        for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53):
            if x % p == 0:
                return x == p

        r, s = 0, x-1
        while s%2 == 0:
            r += 1
            s //= 2
        for _ in range(n):
            a = random.randint(2, x-2)
            b = pow(a, s, x)
            if b in (1, x-1):
                continue
            for _ in range(r-1):
                b = (b * b) % x
                if b == x-1:
                    break
            else:
                return False

        return True

    def next_prime(x):
        """Return the next probable prime number > x."""
        if x <= 1:
            x = 2
        else:
            x += 1 + x%2
            while not is_prime(x):
                x += 2
        return x

    def powmod(x, y, m):
        """Return (x**y) mod m."""
        return pow(x, y, m)

    def gcdext(a, b):
        """Return a 3-element tuple (g, s, t) such that g == gcd(a, b) and g == a*s + b*t.

        The particular values output for s and t are consistent with gmpy2, satisfying the
        following convention inherited from the GMP library, defining s and t uniquely.

        Normally, abs(s) < abs(b)/(2g) and abs(t) < abs(a)/(2g) will hold.
        When no such s and t exist, we put s=0 and t=sign(b), if this is because abs(a)=abs(b)=g.
        Otherwise, we put s=sign(a), if b=0 or abs(b)=2g, and we put t=sign(b), if a=0 or abs(a)=2g.
        """
        g, f = a, b
        s, s1 = 1, 0
        t, t1 = 0, 1
        while f:
            g, (q, f) = f, divmod(g, f)
            s, s1 = s1, s - q * s1
            t, t1 = t1, t - q * t1
        if g < 0:
            g, s, t = -g, -s, -t
        elif g == 0:
            s = 0  # case a=b=0
        if (a < 0 < b or b < 0 < a) and abs(b) == 2*g:
            s, t = -s, t - s * (abs(a) // g)
        return g, s, t

    def invert(x, m):
        """Return y such that x*y == 1 modulo m.

        Raises ZeroDivisionError if no inverse y exists (or, if m is zero).
        """
        if not m:
            raise ZeroDivisionError('invert() division by 0')

        m = abs(m)
        if m == 1:
            return 0

        a, b, = x, m
        s, s1 = 1, 0
        while b:
            a, (q, b) = b, divmod(a, b)
            s, s1 = s1, s - q * s1
        if a != 1:
            raise ZeroDivisionError('invert() no inverse exists')

        y = s + m if s < 0 else s  # ensure 0 < y < m
        return y

    def legendre(x, y):
        """Return the Legendre symbol (x|y), assuming y is an odd prime."""
        z = pow(x, (y-1)//2, y)
        if z > 1:  # z == y-1
            z = -1
        return z

    def is_square(x):
        """Return True if x is a perfect square, False otherwise."""
        if x&15 not in (0, 1, 4, 9):  # quick modulo 16 test
            return False

        y = isqrt(x)
        return x == y**2

    def isqrt(x):
        """Return integer square root of nonnegative x."""
        if x == 0:
            return x

        k = (x.bit_length() - 1) // 2
        y = 1<<k
        for i in range(k-1, -1, -1):
            z = y | 1<<i
            if z**2 <= x:
                y = z
        return y

    def iroot(x, n):
        """Return (y, b) where y is the integer nth root of x and b is True if y is exact."""
        if x == 0:
            return x, True

        k = (x.bit_length() - 1) // n
        y = 1<<k
        for i in range(k-1, -1, -1):
            z = y | 1<<i
            if z**n <= x:
                y = z
        return y, x == y**n

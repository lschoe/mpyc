"""This module collects all gmpy2 functions used by MPyC.

Plus a function for factoring prime powers.

Stubs of limited functionality and efficiency are provided
in case the gmpy2 package is not available.
"""


def factor_prime_power(x):  # TODO: move this to a separate math/number theory module
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


try:
    from gmpy2 import is_prime, next_prime, powmod, invert, legendre, is_square, isqrt, iroot
except ImportError:
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
            a = random.randrange(1, x-1)
            b = pow(a, s, x)
            if b == 1:
                continue
            for _ in range(r):
                if b == x-1:
                    break
                if b == 1:
                    return False

                b = (b * b) % x
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

    def invert(x, m):
        """Return y such that x*y == 1 (mod m), assuming m is prime.

        Raises ZeroDivisionError if no inverse exists.
        """
        if m == 2:
            y = x%2
        else:
            y = pow(x, m-2, m)
        if y == 0:
            raise ZeroDivisionError

        return y

    def legendre(x, y):
        """Return the Legendre symbol (x|y), assuming y is an odd prime."""
        z = pow(x, (y-1)//2, y)
        if z > 1:  # z == y-1
            z = -1
        return z

    def is_square(x):
        """Return True if x is a perfect square, False otherwise."""
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

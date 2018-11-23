"""This module collects all gmpy2 functions used by MPyC.

Stubs of limited functionality and efficiency are provided
in case the gmpy2 package is not available.
"""

try:
    from gmpy2 import is_prime, next_prime, powmod, invert, legendre
except ImportError:
    import random
    def is_prime(x, n=25):
        """Return True if x is probably prime, else False if x is
        definitely composite, performing up to n Miller-Rabin
        primality tests.
        """
        if x <= 2 or x % 2 == 0:
            return x == 2
        # odd x >= 3
        r, s = 0, x - 1
        while s % 2 == 0:
            r += 1
            s //= 2
        for _ in range(n):
            a = random.randrange(1, x - 1)
            b = pow(a, s, x)
            if b == 1:
                continue
            for _ in range(r):
                if b == x - 1:
                    break
                elif b == 1:
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
            x += 1 + (x % 2)
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
            y = x % 2
        else:
            y = pow(x, m - 2, m)
        if y == 0:
            raise ZeroDivisionError
        return y

    def legendre(x, y):
        """Return the Legendre symbol (x|y), assuming y is an odd prime."""
        z = pow(x, (y - 1) // 2, y)
        if z > 1: # z == y - 1
            z = -1
        return z

"""This module collects all gmpy2 functions used by MPyC.

Stubs of limited functionality and efficiency are provided
in case the gmpy2 package is not available.
"""
try:
    from gmpy2 import is_prime, next_prime, powmod, invert, legendre
except:
    import random
    def is_prime(x, n=25): #Miller-Rabin primality test
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
        if x <= 1:
            return 2
        else:
            x += 1 + (x % 2)
            while not is_prime(x):
                x += 2
            return x

    def powmod(x, y, m):
        return pow(x, y, m)

    def invert(x, y): # y prime
        if y == 2:
            z = x % 2
        else:
            z = pow(x, y - 2, y)
        if z == 0:
            raise ZeroDivisionError
        return z

    def legendre(x, y): # y odd prime
        z = pow(x, (y - 1) // 2, y)
        if 0 <= z <= 1:
            return z
        else: # z == y - 1
            return -1

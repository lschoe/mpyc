"""Demo unit vectors.

A secure length-n unit vector is of the form [0]*a + [1] + [0]*(n-1-a),
0<=a<n, where a is secret. The demo generates all unit vectors of a given
length n, for secure prime fields, secure integers, and secure fixed-point
numbers both using the built-in MPyC iterative method unit_vector() and
the recursive method secret_index() defined below.
"""

import sys
from mpyc.runtime import mpc


def secret_index(a, n):
    """Return ath unit vector of length n, assuming 0 <= a < n."""

    def si1(a, n):
        """Return (a-1)st unit vector of length n-1 (if 1 <= a < n)
        or all-0 vector of length n-1 (if a=0).
        """
        if n == 1:
            x = []
        elif n == 2:
            x = [a]
        else:
            a2, b = divmod(a, 2)
            z = si1(a2, (n+1)//2)
            y = mpc.scalar_mul(b, z)
            x = [b - sum(y)] + [z[i//2] - y[i//2] if i%2 == 0 else y[i//2] for i in range(n-2)]
        return x
    x = si1(a, n)
    return [type(a)(1) - sum(x)] + x


async def xprint(n, f, sectype):
    print(f'Using {f.__name__} with type {sectype.__name__}:')
    async with mpc:
        for i in range(n):
            print(i, await mpc.output(f(sectype(i), n)))


async def main():
    if sys.argv[1:]:
        n = int(sys.argv[1])
    else:
        n = 10
        print('Setting input to default =', n)

    secfld = mpc.SecFld(min_order=max(len(mpc.parties) + 1, n))
    secint = mpc.SecInt()
    secfxp = mpc.SecFxp()

    await xprint(n, mpc.unit_vector, secfld)
    # NB: secret_index does not work with secfld
    await xprint(n, mpc.unit_vector, secint)
    await xprint(n, secret_index, secint)
    await xprint(n, mpc.unit_vector, secfxp)
    await xprint(n, secret_index, secfxp)

if __name__ == '__main__':
    mpc.run(main())

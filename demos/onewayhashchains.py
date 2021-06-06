"""Demo MPC-based One-Way Hash Chains.

This MPyC demo shows how to generate and reverse one-way hash chains in a multiparty setting.
The seed for a hash chain, which serves as the private key, is generated jointly at random such
that no party learns any information about it. Subsequently, the hash chain is built and
reversed securely by the parties. No information about the upcoming elements of the hash
chain is leaked whatsoever.

Optimal binary pebbling is employed to reverse a length-2^k hash chain using k/2 hashes per
output round, storing k hash values. See the accompanying notebook for references.
"""

import argparse
import itertools
from mpyc.runtime import mpc
import aes  # MPyC AES demo operating on 4x4 arrays over GF(256).

f = None  # one-way function


def tS(k, r):
    """Optimal schedule for binary pebbling."""
    if r < 2**(k-1):
        return 0

    return ((k + r)%2 + k+1 - ((2*r) % (2**(2**k - r).bit_length())).bit_length()) // 2


def P(k, x):
    """Recursive optimal binary pebbler outputs {f^i(x)}_{i=0}^{n-1} in reverse, n=2^k."""
    # initial stage
    y = [None]*k + [x]
    i = k
    g = 0
    for r in range(1, 2**k):
        for _ in range(tS(k, r)):
            z = y[i]
            if g == 0:
                i -= 1
                g = 2**i
            y[i] = f(z)
            g -= 1
        yield
    # output stage
    yield y[0]
    for v in itertools.zip_longest(*(P(i-1, y[i]) for i in range(1, k+1))):
        yield next(filter(None, v))


def p(k, x):
    """Iterative optimal binary pebbler generating {f^i(x)}_{i=0}^{n-1} in reverse, n=2^k."""
    # initial stage
    z = []
    y = x
    for h in range(2**k, 1, -1):
        if h & (h-1) == 0:  # h is power of 2
            z.insert(0, y)
        y = f(y)
        yield
    # output stage
    yield y
    a = [None] * (k>>1)
    v = 0
    for r in range(2**k - 1, 0, -1):
        yield z[0]
        c = r
        i = 0
        while ~c&1:
            z[i] = z[i+1]
            i += 1
            c >>= 1
        i += 1
        c >>= 1
        if c&1:
            a[v] = (i, 0)
            v += 1
        u = v
        w = (r&1) + i+1
        while c:
            while ~c&1:
                w += 1
                c >>= 1
            u -= 1
            q, g = a[u]
            for _ in range(w//2):
                y = z[q]
                if not g:
                    q -= 1
                    g = 2**q
                z[q] = f(y)
                g -= 1
            if q:
                a[u] = q, g
            else:
                v -= 1
            w = w&1
            while c&1:
                w += 1
                c >>= 1


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--order', type=int, metavar='K',
                        help='order K of hash chain, length n=2**K')
    parser.add_argument('--recursive', action='store_true',
                        help='use recursive pebbler')
    parser.add_argument('--no-one-way', action='store_true',
                        default=False, help='use dummy one-way function')
    parser.add_argument('--no-random-seed', action='store_true',
                        default=False, help='use fixed seed')
    parser.set_defaults(order=1)
    args = parser.parse_args()

    await mpc.start()

    if args.recursive:
        Pebbler = P
    else:
        Pebbler = p

    IV = [[aes.secfld(3)] * 4] * 4  # IV as 4x4 array of GF(256) elements
    global f
    if args.no_one_way:
        D = aes.circulant_matrix([3, 0, 0, 0])
        f = lambda x: mpc.matrix_prod(D, x)
    else:
        K = aes.key_expansion(IV)
        f = lambda x: mpc.matrix_add(aes.encrypt(K, x), x)

    if args.no_random_seed:
        x0 = IV
    else:
        x0 = [[mpc.random.getrandbits(aes.secfld, 8) for _ in range(4)] for _ in range(4)]

    k = args.order
    print(f'Hash chain of length {2**k}:')
    r = 1
    for v in Pebbler(k, x0):
        if v is None:  # initial stage
            print(f'{r:4}', '-')
            await mpc.throttler(0.1)  # raise barrier roughly every 10 AES calls
        else:  # output stage
            await aes.xprint(f'{r:4} x{2**(k+1) - 1 - r:<4} =', v)
        r += 1

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

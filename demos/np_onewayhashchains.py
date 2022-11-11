"""Demo Threshold One-Way Hash Chains, vectorized.

This demo is an extended reimplementation of the onewayhashchain.py demo for
generating and reversing one-way hash chains in a multiparty setting.

Next to the Matyas-Meyer-Oseas one-way function based on AES, the SHAKE128
one-way function from the SHA3 family is also provided as an option.

Note that in the output stage the hashes pertaining to different pebbles are
evaluated in parallel, without increasing the overall round complexity. Multiple
hashes pertaining to the same pebble, however, are necessarily evaluated in series,
increasing the overall round complexity accordingly.

See demo onewayhashchain.py for more information.
"""

import argparse
import itertools
import numpy as np
from mpyc.runtime import mpc
import np_aes as aes  # vectorized AES demo operating on secure arrays over GF(256)
import sha3  # vectorized SHA3/SHAKE demo operating on secure arrays over GF(2)

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
    parser.add_argument('--sha3', action='store_true',
                        help='use SHAKE128 as one-way function')
    parser.add_argument('--no-one-way', action='store_true',
                        help='use dummy one-way function')
    parser.add_argument('--no-random-seed', action='store_true',
                        help='use fixed seed')
    parser.set_defaults(order=1)
    args = parser.parse_args()

    await mpc.start()

    if args.recursive:
        Pebbler = P
    else:
        Pebbler = p

    secfld = sha3.secfld if args.sha3 else aes.secfld

    IV = np.array([[3] * 4] * 4)  # IV as 4x4 array of bytes
    global f
    if args.no_one_way:
        D = aes.circulant([3, 0, 0, 0])
        f = lambda x: D @ x
    elif args.sha3:
        f = lambda x: sha3.shake(x, 128)
    else:
        K = aes.key_expansion(secfld.array(IV))
        f = lambda x: aes.encrypt(K, x) + x

    if args.no_random_seed:
        if args.sha3:
            # convert 4x4 array of bytes to length-128 array of bits
            IV = np.array([(b >> i) & 1 for b in IV.flat for i in range(8)])
        x0 = secfld.array(IV)
    else:
        x0 = mpc.np_random_bits(secfld, 128)
        if not args.sha3:
            # convert length-128 array of bits to 4x4 array of bytes
            x0 = mpc.np_from_bits(x0.reshape(4, 4, 8))

    xprint = sha3.xprint if args.sha3 else aes.xprint

    k = args.order
    print(f'Hash chain of length {2**k}:')
    r = 1
    for v in Pebbler(k, x0):
        if v is None:  # initial stage
            print(f'{r:4}', '-')
            await mpc.throttler(0.0625)  # raise barrier every 16 calls to one-way f()
        else:  # output stage
            await xprint(f'{r:4} x{2**(k+1) - 1 - r:<4} =', v)
        r += 1
    print(f'Performed {k * 2**(k-1) = } hashes in total.')

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

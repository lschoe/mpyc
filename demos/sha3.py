"""Demo Threshold SHA-3 hash functions.

This demo shows how to implement threshold (multiparty) cryptographic hash
functions using MPyC's secure NumPy arrays over GF(2). The SHA-3 family of
hash functions is selected as these functions are quite MPC-friendly. The
nonlinear part of each internal round consists of 1600 bit multiplications
in parallel.

The demo covers the SHA-3 hash functions with output lengths 224, 256, 384,
and 512 as well as the SHAKE extendable-output functions (XOFs) at security
levels 128 and 256.

The type mpc.SecFld(2) is used to represent the secret-shared bits manipulated
by the Keccak permutation function. Note that one cannot do nontrivial Shamir
secret sharing over GF(2). Therefore, the secure type mpc.SecFld(2) switches
internally to an extension of degree e such that 2^e exceeds the number of
parties. The actual computations are done in the GF(2)-subfield of the extension.

See https://en.wikipedia.org/wiki/SHA-3, which refers to the SHA-3 Standard:
Permutation-Based Hash and Extendable-Output Functions (FIPS 202).
"""

import argparse
from hashlib import sha3_224, sha3_256, sha3_384, sha3_512, shake_128, shake_256
import numpy as np
from mpyc.gfpx import GFpX
from mpyc.runtime import mpc

triangular_numbers = tuple(i*(i+1)//2 % 64 for i in range(1, 25))

round_constants = tuple(tuple(int(GFpX(2).powmod('x', 7*i + j, 'x^8+x^6+x^5+x^4+1')) % 2
                              for j in range(7))
                        for i in range(24))

secfld = mpc.SecFld(2)


def _keccak_f1600(S):
    """Keccak-f[1600] permutation applied to 1600-bit array S.

    Operating over secure GF(2) arrays.
    """
    # Convert S into 3D array A[x, y, z] = S[64(5y + x) + z]
    A = S.reshape(5, 5, 64).transpose(1, 0, 2)

    for r in range(24):
        # Apply θ
        C = A.sum(axis=1)
        D = np.roll(C, 1, axis=0) + np.roll(np.roll(C, -1, axis=0), 1, axis=1)
        A += D[:, np.newaxis, :]

        # Apply ρ and π
        AA = A.copy()  # TODO: check alternatives for use of copy()
        x, y = 1, 0
        for shift in triangular_numbers:
            lane = AA[x, y]
            x, y = y, (2*x + 3*y) % 5
            A = mpc.np_update(A, (x, y), np.roll(lane, shift))

        # Apply χ
        A += (np.roll(A, -1, axis=0) + 1) * np.roll(A, -2, axis=0)

        # Apply ι
        for j in range(7):
            key = (0, 0, (1<<j)-1)
            A = mpc.np_update(A, key,  A[key] + round_constants[r][j])

    S = A.transpose(1, 0, 2).reshape(1600)
    return S


@mpc.coroutine
async def keccak_f1600(S):
    """Keccak-f[1600] permutation applied to 1600-bit array S.

    Slightly optimized version, operating over finite field arrays.
    """
    await mpc.returnType((type(S), S.shape))
    # Convert S into 3D array A[x, y, z] = S[64(5y + x) + z]
    S = await mpc.gather(S)  # NB: S is now a finite field array
    S = S.copy()  # TODO: investigate why needed for SHAKE with d > r
    A = S.reshape(5, 5, 64).transpose(1, 0, 2)

    for r in range(24):
        # Apply θ
        C = A.sum(axis=1)
        D = np.roll(C, 1, axis=0) + np.roll(np.roll(C, -1, axis=0), 1, axis=1)
        A += D[:, np.newaxis, :]

        # Apply ρ and π
        x, y = 1, 0
        lane = A[x, y]
        for shift in triangular_numbers:
            x, y = y, (2*x + 3*y) % 5
            lane, A[x, y] = A[x, y].copy(), np.roll(lane, shift)

        # Apply χ
        A += (np.roll(A, -1, axis=0) + 1) * np.roll(A, -2, axis=0)
        A = await mpc._reshare(A)

        # Apply ι
        for j in range(7):
            A[0, 0, (1<<j)-1] += round_constants[r][j]

    S = A.transpose(1, 0, 2).reshape(1600)
    return S


def sponge(r, N, d):
    """Sponge construction with the Keccak-f[1600] permutation with rate r and output length d."""
    # Pad with 10^*1 to make input length multiple of r.
    P = np.concatenate((N, np.array([1] + [0]*((-(N.size + 2)) % r) + [1])))
    n = P.size // r
    P = P.reshape(n, r)

    # Absorb input P into sponge S.
    S = secfld.array(np.zeros(1600, dtype=object))
    for i in range(n):
        U = P[i] + S[:r]
        S = mpc.np_update(S, slice(r), U)  # S[:r] = U
        S = keccak_f1600(S)

    # Squeeze output Z from sponge S.
    Z = S[:r]
    while len(Z) < d:
        S = keccak_f1600(S)
        Z = np.concatenate((Z, S[:r]))
    return Z[:d]


def keccak(c, N, d):
    """Keccak function with given capacity c and output length d applied to bit string N."""
    r = 1600 - c  # rate r satisfying r + c = b = 1600
    return sponge(r, N, d)


def sha3(M, d=256, c=128):
    """SHA3 hash of the given message M with output length d."""
    # append 01 to M
    N = np.concatenate((M, np.array([0, 1])))
    return keccak(c, N, d)


def shake(M, d, c=256):
    """SHAKE[c//2] of the given message M with output length d."""
    # append 1111 to M
    N = np.concatenate((M, np.array([1, 1, 1, 1])))
    return keccak(c, N, d)


async def xprint(text, s):
    """Print and return bit array s as hex string."""
    s = await mpc.output(s)
    s = np.fliplr(s.reshape(-1, 8)).reshape(-1)  # reverse bits for each byte
    d = len(s)
    s = f'{int("".join(str(int(b)) for b in s), 2):0{d//4}x}'  # bits to hex digits with leading 0s
    print(f'{text} {s}')
    return s


async def main():
    global keccak_f1600

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, metavar='I',
                        help='input string I (default "hello123")')
    parser.add_argument('-n', type=int, metavar='N',
                        help='number of times N (default 1) to repeat input I')
    parser.add_argument('--hash', action='store_true', default=True,
                        help='apply SHA3 hash (default) to input I^N')
    parser.add_argument('--shake', action='store_true', default=False,
                        help='apply SHAKE extendable output function to input I^N')
    parser.add_argument('--no-optimize', action='store_true',
                        help='use slightly slower implementation')
    parser.add_argument('-d', type=int, metavar='D',
                        help='set output length D  (default 256 for SHA3, 512 for SHAKE)')
    parser.add_argument('-c', type=int, metavar='C',
                        help='set capacity C (default 512)')
    parser.set_defaults(i='hello123', n=1, c=None, d=None)
    args = parser.parse_args()

    if args.no_optimize:
        print('No optimization, using high-level f1600')
        keccak_f1600 = _keccak_f1600

    c = args.c
    d = args.d
    if args.hash and not args.shake:
        # SHA3
        if c is None and d is None:
            c = 512
        if d is None:
            d = c//2
        if c is None:
            c = 2*d
        assert c == 2*d
        assert d in (224, 256, 384, 512)
        F = sha3
        f = {224: sha3_224, 256: sha3_256, 384: sha3_384, 512: sha3_512}[d]
        e = ()
    else:
        # SHAKE
        if c is None:
            c = 512
        if d is None:
            d = c
        assert c in (256, 512)
        assert d%8 == 0
        F = shake
        f = {256: shake_128, 512: shake_256}[c]
        e = (d//8,)

    print(f'function {F.__name__} with capacity {c} and output length {d}')

    await mpc.start()

    X = args.i.encode() * args.n
    print(f'Input: {X}')
    x = np.array([(b >> i) & 1 for b in X for i in range(8)])  # bytes to bits
    x = secfld.array(x)  # secret-shared input bits

    y = F(x, d, c)  # secret-shared output bits
    Y = await xprint('Output:', y)
    assert Y == f(X).hexdigest(*e)

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

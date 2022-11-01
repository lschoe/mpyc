"""Demo Threshold AES cipher, vectorized.

This demo is a fully equivalent reimplementation of the aes.py demo.
Secure arrays over GF(256) to perform all computations in a vectorized
manner. For example, in each encryption round the S-Boxes are evaluated
for all 16 bytes of the state in one go; in aes.py this was done by
applying the S-Box to each byte one at a time. Similarly, the 4 S-Boxes
in each round of the key expansion are evaluated in one go.

Apart from reducing the overhead, which makes the vectorized version about
twice as fast as aes.py, the code is rather simple as well.

See demo aes.py for background information.
"""

import sys
import numpy as np
from mpyc.runtime import mpc

secfld = mpc.SecFld(2**8)  # Secure AES field GF(2^8) for secret values.
f256 = secfld.field        # Plain AES field GF(2^8) for public values.


def circulant(r):
    """Circulant matrix with first row r."""
    r = np.stack([np.roll(r, j, axis=0) for j in range(len(r))])
    return f256.array(r)


A = circulant([1, 0, 0, 0, 1, 1, 1, 1])   # 8x8 matrix A over GF(2)
A1 = np.linalg.inv(A)                     # inverse of A
B = f256.array([1, 1, 0, 0, 0, 1, 1, 0])  # vector B over GF(2)
C = circulant([2, 3, 1, 1])               # 4x4 matrix C over GF(2^8)
C1 = np.linalg.inv(C)                     # inverse of C


def sbox(x):
    """AES S-Box."""
    x = mpc.np_to_bits(x**254)
    x = (A @ x[..., np.newaxis]).reshape(*x.shape)
    x += B
    x = mpc.np_from_bits(x)
    return x


def sbox1(x):
    """AES inverse S-Box."""
    x = mpc.np_to_bits(x)
    x += B
    x = (A1 @ x[..., np.newaxis]).reshape(*x.shape)
    x = mpc.np_from_bits(x)**254
    return x


def key_expansion(k):
    """AES key expansion for 128/256-bit keys."""
    w = k
    Nk = k.shape[1]  # Nk is 4 or 8
    Nr = 10 if Nk == 4 else 14
    for i in range(Nk, 4*(Nr+1)):
        t = w[:, -1]
        if i % Nk in (0, 4):
            t = sbox(t)
        if i % Nk == 0:
            t = np.roll(t, -1, axis=0)
            t = mpc.np_update(t, 0, t[0] + (f256(1) << i // Nk - 1))
        t += w[:, -Nk]
        t = t.reshape(4, 1)
        w = np.append(w, t, axis=1)
    K = np.hsplit(w, Nr+1)
    return K


def encrypt(K, s):
    """AES encryption of s given key schedule K."""
    Nr = len(K) - 1  # Nr is 10 or 14
    s += K[0]
    for i in range(1, Nr+1):
        s = sbox(s)
        s = np.stack([np.roll(s[j], -j, axis=0) for j in range(4)])
        if i < Nr:
            s = C @ s
        s += K[i]
    return s


def decrypt(K, s):
    """AES decryption of s given key schedule K."""
    Nr = len(K) - 1  # Nr is 10 or 14
    for i in range(Nr, 0, -1):
        s += K[i]
        if i < Nr:
            s = C1 @ s
        s = np.stack([np.roll(s[j], j, axis=0) for j in range(4)])
        s = sbox1(s)
    s += K[0]
    return s


async def xprint(text, s):
    """Print matrix s transposed and flattened as hex string."""
    s = await mpc.output(s)
    s = s.T.flatten()
    print(f'{text} {bytes(map(int, s)).hex()}')


async def main():
    if sys.argv[1:]:
        full = False
        print('AES-128 encryption only.')
    else:
        full = True
        print('AES-128 en/decryption and AES-256 en/decryption.')

    print('AES polynomial:', f256.modulus)  # x^8 + x^4 + x^3 + x + 1

    await mpc.start()

    p = secfld.array(f256.array([[17 * (4*j + i) for j in range(4)] for i in range(4)]))
    await xprint('Plaintext:  ', p)

    k128 = secfld.array(f256.array([[4*j + i for j in range(4)] for i in range(4)]))
    await xprint('AES-128 key:', k128)
    K = key_expansion(k128)
    c = encrypt(K, p)
    await xprint('Ciphertext: ', c)
    if full:
        p = decrypt(K, c)
        await xprint('Plaintext:  ', p)

        k256 = secfld.array(f256.array([[4*j + i for j in range(8)] for i in range(4)]))
        await xprint('AES-256 key:', k256)
        K = key_expansion(k256)
        c = encrypt(K, p)

        await xprint('Ciphertext: ', c)
        p = decrypt(K, c)
        await xprint('Plaintext:  ', p)

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

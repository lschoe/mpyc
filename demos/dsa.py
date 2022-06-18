"""Demo Threshold DSA (Digital Signature Algorithm) and variants.

See the demo elgamal.py for background information.

The demo shows how to implement threshold (EC)DSA and Schnorr signatures.
Basically, any group with a given generator g of known prime order q can be used.
Schnorr groups are used for DSA and Schnorr signatures, and
elliptic curves are used for ECDSA and also for Schnorr signatures.

Secure field mpc.SecFld(q) is used to perform the secure (threshold) arithmetic
with private key x and the random nonces (k for DSA and u for Schnorr).
As for secure groups, only function repeat_public() is used, namely
for threshold generation of keys and signatures. Signature verification
is done in the clear, using ordinary group operations.

A call like repeat_public(g,x) will generate a public group element g^x,
for a given secret-shared exponent x (and publicly known base g).
The underlying protocol lets the MPC parties compute g^x without exposing
their individual shares of secret x.
"""

import time
import argparse
from hashlib import sha1, sha224, sha256, sha384, sha512
from elgamal import keygen  # reuse key generation from threshold ElGamal cryptosystem
from mpyc.gmpy import invert
from mpyc.fingroups import SchnorrGroup, EllipticCurve, EllipticCurvePoint
from mpyc.runtime import mpc


class DSA:
    """Threshold (EC)DSA with Schnorr groups (or, elliptic curves)."""

    def __init__(self, group):
        """Threshold DSA signer."""
        self.group = group

    async def keygen(self):
        """Threshold DSA key generation."""
        self.x, self.y = await keygen(self.group.generator)

    async def sign(self, M):
        """Threshold DSA signature generation."""
        g = self.group.generator
        q = self.group.order
        H = self.H
        x = self.x
        secgrp = mpc.SecGrp(self.group)
        secfld = mpc.SecFld(q)
        while True:
            k = mpc._random(secfld)
            a = await secgrp.repeat_public(g, k)  # a = g^k
            if a == self.group.identity:
                continue
            r = self.to_int(a) % q
            if r == 0:
                continue
            s = (H(M) + x * r) / k
            s = int(await mpc.output(s))
            if s != 0:
                break
        S = r, s
        return S

    def verify(self, M, S):
        """DSA signature verification."""
        g = self.group.generator
        q = self.group.order
        H = self.H
        y = self.y
        r, s = S

        if not (0 < r < q and 0 < s < q):
            return False

        w = int(invert(s, q))  # s^-1 mod q
        u_1 = H(M)*w % q
        u_2 = r*w % q
        v = self.to_int((g^u_1) @ (y^u_2)) % q
        return v == r

    def H(self, M):
        """Hash message M using a SHA-2 hash function with sufficiently large output length."""
        N = (self.group.order.bit_length() + 7) // 8  # byte length
        N_sha = ((20, sha1), (28, sha224), (32, sha256), (48, sha384))
        sha = next((sha for _, sha in N_sha if _ >= N), sha512)

        h = int.from_bytes(sha(M).digest()[:N], byteorder='big')
        return h

    @staticmethod
    def to_int(a):
        """Map group element a to an integer value."""
        if isinstance(a, EllipticCurvePoint):  # cf. ECDSA
            z = int(a.normalize().x)           # x-coordinate of point a on elliptic curve
        else:           # cf. DSA
            z = int(a)  # Schnorr group element a
        return z


class Schnorr:
    """Threshold Schnorr signatures for groups of prime order."""

    def __init__(self, group):
        """Threshold Schnorr signer."""
        self.group = group

    async def keygen(self):
        """Threshold Schnorr key generation."""
        self.x, self.h = await keygen(self.group.generator)

    async def sign(self, M):
        """Threshold Schnorr signature generation."""
        g = self.group.generator
        H = self.H
        x = self.x
        secgrp = mpc.SecGrp(self.group)
        secfld = mpc.SecFld(self.group.order)

        u = mpc._random(secfld)
        a = await secgrp.repeat_public(g, u)  # a = g^u
        c = H(a, M)
        r = u + c * x
        r = int(await mpc.output(r))
        S = c, r
        return S

    def verify(self, M, S):
        """Schnorr signature verification."""
        g = self.group.generator
        H = self.H
        h = self.h
        c, r = S
        return c == H((g^r) @ (h^-c), M)

    def H(self, a, M):
        """Hash a and M using a SHA-2 hash function with sufficiently large output length."""
        N = (self.group.order.bit_length() + 7) // 8  # byte length
        N_sha = ((20, sha1), (28, sha224), (32, sha256), (48, sha384))
        sha = next((sha for _, sha in N_sha if _ >= N), sha512)

        a = self.to_bytes(a)
        c = int.from_bytes(sha(a + M).digest()[:N], byteorder='big')
        return c

    @staticmethod
    def to_bytes(a):
        """Map group element a to fixed-length byte string."""
        if isinstance(a, EllipticCurvePoint):  # cf. ECDSA
            z = int(a.normalize().x)           # x-coordinate of point a on elliptic curve
        else:           # cf. DSA
            z = int(a)  # Schnorr group element a
        N = (a.field.order.bit_length() + 7) // 8
        return z.to_bytes(length=N, byteorder='big')


async def test_sig(sig, group, M):
    """Keygen-Sign-Verify cycle for message M."""
    dsa = sig(group)
    await dsa.keygen()
    S = await dsa.sign(M)
    assert dsa.verify(M, S)


async def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--group', type=int, metavar='G',
                        help=('1=EC (default), 2=SG'))
    parser.set_defaults(group=1)
    args = parser.parse_args()

    if args.group == 1:
        groups = (EllipticCurve('Ed25519'),
                  EllipticCurve('Ed25519', 'projective'),
                  EllipticCurve('Ed25519', 'extended'),
                  EllipticCurve('secp256k1', 'projective'))
    else:
        groups = (SchnorrGroup(p=9739, q=541),
                  SchnorrGroup(n=160),
                  SchnorrGroup(l=2048))

    M = b'hello there?!'

    await mpc.start()
    print('Sign/verify tests')
    print('-----------------')

    for group in groups:
        print(group.__name__)
        for sig in DSA, Schnorr:
            start = time.process_time()
            await test_sig(sig, group, M)
            print(f'{time.process_time() - start} seconds for {sig.__name__} signature')
            await mpc.barrier()  # synchronize for more accurate timings

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

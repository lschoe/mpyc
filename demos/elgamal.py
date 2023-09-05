"""Demo Threshold ElGamal Cryptosystem.

This demo shows the use of secure groups in MPyC. Four types of groups are used:
quadratic residue groups, Schnorr groups, elliptic curve groups, and class groups.

The parties jointly generate a key pair for the ElGamal cryptosystem, consisting
of a private key x and a public key h=g^x, where g is a generator of the group.
Throughout the entire protocol, private key x will only exist in secret-shared form.

The demo runs a boardroom election, where each party will enter a (random) bit v
representing a (random) yes/no vote. Each party encrypts its vote v using additively
homomorphic ElGamal, hence puts c = (g^u, h^u g^v) as ciphertext for a random nonce u
(generated privately by each party). The parties then broadcast their ciphertexts,
and multiply everything together to obtain a ciphertext representing the sum of their
votes. Finally, the sum is decrypted, and the total number of "yes" votes is obtained.

In this demo we apply the default notation for finite groups in MPyC, writing @ for
the group operation, ~ for group inversion, and ^ for the repeated group operation.

See also the demo dsa.py for threshold signatures from secure groups.
"""

import random
import argparse
from mpyc.gmpy import is_prime, isqrt
from mpyc.runtime import mpc


async def keygen(g):
    """Threshold ElGamal key generation."""
    group = type(g)
    secgrp = mpc.SecGrp(group)
    n = group.order
    if n is not None and is_prime(n):
        secnum = mpc.SecFld(n)
    else:
        l = isqrt(-group.discriminant).bit_length()
        secnum = mpc.SecInt(l)

    while True:
        x = mpc._random(secnum)
        h = await secgrp.repeat_public(g, x)  # g^x
        if h != group.identity:
            # NB: this branch will always be followed unless n is artificially small
            return x, h


def encrypt(g, h, M):
    """ElGamal encryption."""
    group = type(g)
    n = group.order
    if n is None:
        n = isqrt(-group.discriminant)
    u = random.randrange(n)
    c = (g^u, (h^u) @ M)
    return c


async def decrypt(C, x, public_out=True):
    """Threshold ElGamal decryption.

    The given ciphertext C=(A,B) consists of two group elements.
    If public_out is set (default), the decrypted message M will also be a group element;
    otherwise, the decrypted message M will be a secure (secret-shared) group element.
    """
    A, B = C
    group = type(A)
    secgrp = mpc.SecGrp(group)
    if public_out:
        A_x = await secgrp.repeat_public(A, -x)  # A^-x
        assert isinstance(A_x, group)
    else:
        A_x = A^-x
        assert isinstance(A_x, secgrp)
    M = A_x @ B
    return M


async def election(secgrp):
    """Boardroom election between all MPC parties."""
    group = secgrp.group
    # Create ElGamal key pair:
    g = group.generator
    x, h = await keygen(g)

    # Each party encrypts a random vote:
    v = random.randint(0, 1)
    print(f'''My vote: {v} (for {'"yes"' if v else '"no"'})''')
    c = encrypt(g, h, g^v)  # additive homomorphic ElGamal
    c = await mpc.transfer(c)

    # Accumulate all votes:
    C = list(c[0])
    for c_i in c[1:]:
        C[0] @= c_i[0]
        C[1] @= c_i[1]

    # Decrypt using MPC:
    M = await decrypt(C, x, public_out=True)
    T, t = group.identity, 0  # T = g^t
    while T != M:
        T, t = T @ g, t+1
    print(f'Referendum result: {t} "yes" / {len(c) - t} "no"')


async def crypt_cycle(secgrp, M, public_out=True):
    """Encrypt/decrypt cycle for message M."""
    group = secgrp.group
    # Create ElGamal key pair:
    g = group.generator
    x, h = await keygen(g)

    # Party 0 encrypts message M and broadcasts ciphertext C:
    if mpc.pid == 0:
        M, Z = group.encode(M)
        C_M = encrypt(g, h, M)
        C_Z = encrypt(g, h, Z)
        C = C_M, C_Z
    else:
        C = None
    C = await mpc.transfer(C, senders=0)

    # Threshold decrypt C:
    C_M, C_Z = C
    M = await decrypt(C_M, x, public_out=public_out)
    Z = await decrypt(C_Z, x, public_out=public_out)
    if public_out:
        M = group.decode(M, Z)
    else:
        M = secgrp.decode(M, Z)
    return M


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--group', type=int, metavar='G',
                        help=('1=EC (default), 2=QR, 3=SG, 4=Cl'))
    parser.add_argument('-b', '--batch-size', type=int, metavar='B',
                        help='number of messages B in batch, B>=1')
    parser.add_argument('-o', '--offset', type=int, metavar='O',
                        help='offset O for batch of messages, O>=0')
    parser.add_argument('--no-public-output', action='store_true',
                        help='force secure (secret-shared) message upon decryption')
    parser.set_defaults(group=1, batch_size=1, offset=0)
    args = parser.parse_args()

    if args.group == 1:
        secgrp = mpc.SecEllipticCurve('secp256k1', 'projective')
    elif args.group == 2:
        secgrp = mpc.SecQuadraticResidues(l=2048)
    elif args.group == 3:
        secgrp = mpc.SecSchnorrGroup(l=1024)
    elif args.group == 4:
        if args.no_public_output:
            secgrp = mpc.SecClassGroup(l=32)
        else:
            secgrp = mpc.SecClassGroup(l=1024)
    print(f'Using secure group: {secgrp.__name__}')

    await mpc.start()
    print('Boardroom election')
    print('------------------')
    await election(secgrp)
    print()

    print('Encryption/decryption tests')
    print('---------------------------')
    for m in range(args.batch_size):
        m += 1 + args.offset
        print(f'Plaintext sent: {m}')
        p = await crypt_cycle(secgrp, m, not args.no_public_output)
        if args.no_public_output:
            # p is a secure
            p = await mpc.output(p)
        print(f'Plaintext received: {p}')
        assert m == p, (m, p)
    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

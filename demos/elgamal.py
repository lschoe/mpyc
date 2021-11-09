"""Demo Threshold ElGamal.

This demo shows the use of secure groups in MPyC. Three types of groups are used,
namely quadratic residue groups, elliptic curve groups, and class groups.

The parties jointly generate a key pair for the ElGamal cryptosystem, consisting
of a private key x and a public key h=g^x, where g is a generator of the group.
Throughout the entire protocol, private key x will only exist in secret-shared form.

The demo runs a boardroom election, where each party will enter a (random) bit v
representing a (random) yes/no vote. Each party encrypts its vote v using homomorphic
ElGamal, hence puts c = (g^u, h^u g^v) as ciphertext for a random nonce u (generated
privately by each party). The parties then broadcast their ciphertexts, and multiply
everything together to obtain a ciphertext representing the sum of their votes.
Finally, the sum is decrypted, and the total number of "yes" votes is obtained.

In this demo we apply the default notation for finite groups in MPyC, writing @ for
the group operation, ~ for group inversion, and ^ for the repeated group operation.

Demo still under construction.
"""

import math
import random
import argparse
from mpyc.gmpy import is_prime
from mpyc.runtime import mpc
from mpyc.fingroups import QuadraticResidues, EllipticCurve, ClassGroup


async def keygen(g):
    """ElGamal key generation."""
    group = type(g)
    secgrp = mpc.SecGrp(group)
    n = group.order
    if n is not None and is_prime(n):
        # QuadraticResidues/EllipticCurve group
        secnum = mpc.SecFld(modulus=n)
    else:
        # Class group
        secnum = mpc.SecInt((int(math.sqrt(-group.discriminant))).bit_length())
    x = mpc._random(secnum)
    h = await secgrp.repeat_public(g, x)  # g^x
    return x, h


def encrypt(g, h, m):
    """ElGamal encryption."""
    group = type(g)
    n = group.order
    if n is None:
        n = int(math.sqrt(-group.discriminant))
    u = random.randrange(n)
    c = (g^u, (h^u) @ m)
    return c


async def decrypt(c, x,  public_out=True):
    """ElGamal threshold decryption."""
    a, b = c
    group = type(a)
    secgrp = mpc.SecGrp(group)
    if public_out:
        a_x = await secgrp.repeat_public(a, x)
    else:
        a_x = a^x
    # TODO: optimization use -x above to avoid ~a_x below
    m = ~a_x @ b
    return m


async def election(group):
    """Boardroom election between all MPC parties."""
    # Create ElGamal key pair:
    g = group.generator
    x, h = await keygen(g)

    # Each party encrypts a random vote:
    v = random.randint(0, 1)
    print('My vote:', v)
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
    print('Election result:', t)


async def crypt_cycle(group, m, public_out=True):
    """Encrypt/decrypt cycle for message m."""
    # Create ElGamal key pair:
    g = group.generator
    x, h = await keygen(g)

    # Party 0 encrypts message m and broadcasts ciphertext c:
    if mpc.pid == 0:
        m, z = group.encode(m)
        c_m = encrypt(g, h, m)
        c_z = encrypt(g, h, z)
        c = c_m, c_z
    else:
        c = None
    c = await mpc.transfer(c, senders=0)

    # Threshold decrypt c:
    c_m, c_z = c
    M = await decrypt(c_m, x, public_out=public_out)
    Z = await decrypt(c_z, x, public_out=public_out)
    if public_out:
        m = group.decode(M, Z)
    else:
        secgrp = mpc.SecGrp(group)
        m = secgrp.decode(M, Z)
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int, metavar='I',
                        help=('group 0=QR (default) 1=EC 2=Cl'))
    parser.add_argument('-b', '--batch-size', type=int, metavar='B',
                        help='number of messages')
    parser.add_argument('-o', '--offset', type=int, metavar='O',
                        help='offset for batch')
    parser.add_argument('--no-public-output', action='store_true',
                        default=False, help='force secure (secret-shared) output upon decryption')
    parser.set_defaults(index=0, batch_size=1, offset=0)
    args = parser.parse_args()

    if args.index == 0:
        group = QuadraticResidues(l=1024)
    elif args.index == 1:
        group = EllipticCurve('ED25519', 'extended')
    elif args.index == 2:
        group = ClassGroup(l=1024)
    print(f'Group = {group.__name__}')

    mpc.run(mpc.start())
    print('Boardroom election')
    print('------------------')
    mpc.run(election(group))
    print()

    print('Encrytpion/decryption tests')
    print('---------------------------')
    for m in range(args.batch_size):
        m += 1 + args.offset
        print("Plaintext sent: ", m)
        p = mpc.run(crypt_cycle(group, m, not args.no_public_output))
        if args.no_public_output:
            p = mpc.run(mpc.output(p))
        print("Plaintext received: ", p)
        assert m == p, (m, p)
    mpc.run(mpc.shutdown())

"""Demo BNN MNIST classifier, vectorized.

This demo is a fully equivalent reimplementation of the bnnmnist.py demo,
similar to the cnnmnist.py demo and its vectorized reimplementation.

The layers in the bnnmnist.py demo were already vectorized, except that all
data was kept in lists of secure integers. These lists are now replaced
by secure integer arrays, thereby eliminating the overhead per element.

As a result the performance is improved by a factor of about 3 compared to the
fastest mode for the original demo. The secure matrix-vector multiplications
in fully-connected layers 2 and 3 now form the bottleneck (actually, the n^2
multiplications and additions for the local matrix-vector products with say
64-bit Python integers as entries consumes almost all of the time. Note that
n=4096, hence n^2 = 2^24 ~ 16M of 64-bit multiplications and 128-bit additions.
The neural network has not been optimized in this respect.

See bnnmnist.py for background information on the BNN MNIST classifier.
"""

import os
import logging
import random
import argparse
import gzip
import numpy as np
from mpyc.runtime import mpc
import mpyc.gmpy as gmpy2

secint = None


def load_W_b(name):
    """Load signed binary weights W and signed integer bias values b
    for fully connected layer 'name'.
    """
    W = np.load(os.path.join('data', 'bnn', 'W_' + name + '.npy'))
    W = np.unpackbits(W, axis=0).astype(np.int8)
    W = W*2 - 1  # map bit values 0 and 1 to signed binary values -1 and 1, resp.
    b = np.load(os.path.join('data', 'bnn', 'b_' + name + '.npy')).astype(object)
    return secint.array(W), secint.array(b)


@mpc.coroutine
async def bsgn_0(a):
    """Compute binary sign of a securely.

    Binary sign of a (1 if a>=0 else -1) is obtained by securely computing (2a+1 | p).

    Legendre symbols (a | p) for secret a are computed securely by evaluating
    (a s r^2 | p) in the clear for secret random sign s and secret random r modulo p,
    and outputting secret s * (a s r^2 | p).
    """
    stype = type(a)
    await mpc.returnType((stype, a.shape))
    Zp = stype.sectype.field
    legendre = gmpy2.legendre
    p = gmpy2.mpz(Zp.modulus)
    legendre_p = np.vectorize(lambda a: legendre(a, p), otypes='O')

    n = a.size
    s = mpc.np_random_bits(Zp, n, signed=True)  # n random signs
    r2 = mpc._np_randoms(Zp, n)
    if mpc.options.no_prss:
        r2 = await r2
    r2 **= 2  # n random squares modulo p
    r2 = mpc._reshare(r2)
    s, r2 = await mpc.gather(s, r2)
    y = s * r2
    y = await mpc._reshare(y)
    a = await mpc.gather(a)
    y = y * (2*a + 1).reshape(n)
    y = await mpc.output(y, threshold=2*mpc.threshold)
    return (s * legendre_p(y.value)).reshape(a.shape)


@mpc.coroutine
async def bsgn_1(a):
    """Compute binary sign of a securely.

    Binary sign of a (1 if a>=0 else -1) is obtained by securely computing
    (u+v+w - u*v*w)/2 with u=(2a-1 | p), v=(2a+1 | p), and w=(2a+3 | p).
    """
    stype = type(a)
    await mpc.returnType((stype, a.shape))
    Zp = stype.sectype.field
    legendre = gmpy2.legendre
    p = gmpy2.mpz(Zp.modulus)
    legendre_p = np.vectorize(lambda a: legendre(a, p), otypes='O')

    n = a.size
    s = mpc.np_random_bits(Zp, 3*n, signed=True)  # 3n random signs
    r2 = mpc._np_randoms(Zp, 3*n)
    if mpc.options.no_prss:
        r2 = await r2
    r2 **= 2  # 3n random squares modulo p
    r2 = mpc._reshare(r2)
    s, r2 = await mpc.gather(s, r2)
    s = s.reshape(3, n)
    r2 = r2.reshape(3, n)
    s = np.append(s, [s[0]], axis=0)
    r2 = np.append(r2, [s[1]], axis=0)
    z = s * r2
    z = await mpc._reshare(z)
    a = await mpc.gather(a)
    y = 2*a + 1
    y = y.reshape(1, n)
    y = y + np.array([[-2], [0], [2]])  # y.shape = (3, n)
    y = np.append(y, [s[2]], axis=0)
    y = z * y
    y = await mpc.output(y, threshold=2*mpc.threshold)
    y = y.value
    h = legendre_p(y[:3])
    t = (s[:3] * h).value
    z = h[0] * h[1] * h[2] * y[3]
    q = int((p+1) >> 1)  # q = 1/2 mod p
    return Zp.array((t[0] + t[1] + t[2] - z) * q).reshape(a.shape)


@mpc.coroutine
async def bsgn_2(a):
    """Compute binary sign of a securely.

    Binary sign of a (1 if a>=0 else -1) is obtained by securely computing
    (t | p), with t = sum((2a+1+2i | p) for i=-2,-1,0,1,2).
    """
    stype = type(a)
    await mpc.returnType((stype, a.shape))
    Zp = stype.sectype.field
    legendre = gmpy2.legendre
    p = gmpy2.mpz(Zp.modulus)
    legendre_p = np.vectorize(lambda a: legendre(a, p), otypes='O')

    n = a.size
    s = mpc.np_random_bits(Zp, 6*n, signed=True)  # 6n random signs
    r2 = mpc._np_randoms(Zp, 6*n)
    if mpc.options.no_prss:
        r2 = await r2
    r2 **= 2  # 6n random squares modulo p
    r2 = mpc._reshare(r2)
    s, r2 = await mpc.gather(s, r2)
    s = s.reshape(6, n)
    r2 = r2.reshape(6, n)
    z = s * r2
    z = await mpc._reshare(z)
    a = await mpc.gather(a)
    y = 2*a + 1
    y = y.reshape(1, n)
    y = y + np.array([[-4], [-2], [0], [2], [4]])  # y.shape = (5, n)
    y = y * z[:5]
    y = await mpc._reshare(y)
    y = np.append(y, [z[5]], axis=0)
    y = await mpc.output(y, threshold=2*mpc.threshold)
    t = np.sum(s[:5] * legendre_p(y[:5].value), axis=0)
    t = await mpc.output(t * y[5])
    return (s[5] * legendre_p(t.value)).reshape(a.shape)


async def main():
    global secint

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, metavar='B',
                        help='number of images to classify')
    parser.add_argument('-o', '--offset', type=int, metavar='O',
                        help='offset for batch (otherwise random in [0,10000-B])')
    parser.add_argument('-d', '--d-k-star', type=int, metavar='D',
                        help='k=D=0,1,2 for Legendre-based comparison using d_k^*')
    parser.add_argument('--no-legendre', action='store_true',
                        help='disable Legendre-based comparison')
    parser.set_defaults(batch_size=1, offset=-1, d_k_star=1)
    args = parser.parse_args()

    batch_size = args.batch_size
    offset = args.offset
    if args.no_legendre:
        secint = mpc.SecInt(14)  # using vectorized MPyC integer comparison
    else:
        if args.d_k_star == 0:
            secint = mpc.SecInt(14, p=3546374752298322551)  # Legendre-0 range [-134, 134]
            bsgn = bsgn_0
        elif args.d_k_star == 1:
            secint = mpc.SecInt(14, p=9409569905028393239)  # Legendre-1 range [-383, 383]
            bsgn = bsgn_1
        else:
            secint = mpc.SecInt(14, p=15569949805843283171)  # Legendre-2 range [-594, 594]
            bsgn = bsgn_2

    await mpc.start()

    if offset < 0:
        offset = random.randrange(10001 - batch_size) if mpc.pid == 0 else None
        offset = await mpc.transfer(offset, senders=0)

    logging.info('--------------- INPUT   -------------')
    print(f'Type = {secint.__name__}, range = ({offset}, {offset + batch_size})')
    # read batch_size labels and images at given offset
    df = gzip.open(os.path.join('data', 'cnn', 't10k-labels-idx1-ubyte.gz'))
    d = df.read()[8 + offset: 8 + offset + batch_size]
    labels = list(map(int, d))
    print('Labels:', labels)
    df = gzip.open(os.path.join('data', 'cnn', 't10k-images-idx3-ubyte.gz'))
    d = df.read()[16 + offset * 28**2: 16 + (offset + batch_size) * 28**2]
    L = np.frombuffer(d, dtype=np.ubyte).reshape(batch_size, 28**2)
    if batch_size == 1:
        x = np.array(L[0]).reshape(28, 28)
        print(np.array2string(np.vectorize(lambda a: int(bool((a/255))))(x), separator=''))

    L = secint.array(L)

    logging.info('--------------- LAYER 1 -------------')
    W, b = load_W_b('fc1')
    logging.info('- - - - - - - - fc  784 x 4096  - - -')
    L = L @ W + b
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    L = (L >= 0)*2 - 1
    await mpc.barrier('after-layer-1')

    logging.info('--------------- LAYER 2 -------------')
    W, b = load_W_b('fc2')
    logging.info('- - - - - - - - fc 4096 x 4096  - - -')
    L = L @ W + b
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    if args.no_legendre:
        secint.bit_length = 10
        L = (L >= 0)*2 - 1
    else:
        L = bsgn(L)
    await mpc.barrier('after-layer-2')

    logging.info('--------------- LAYER 3 -------------')
    W, b = load_W_b('fc3')
    logging.info('- - - - - - - - fc 4096 x 4096  - - -')
    L = L @ W + b
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    if args.no_legendre:
        secint.bit_length = 10
        L = (L >= 0)*2 - 1
    else:
        L = bsgn(L)
    await mpc.barrier('after-layer-3')

    logging.info('--------------- LAYER 4 -------------')
    W, b = load_W_b('fc4')
    logging.info('- - - - - - - - fc 4096 x 10  - - - -')
    L = L @ W + b

    logging.info('--------------- OUTPUT  -------------')
    if args.no_legendre:
        secint.bit_length = 14
    for i in range(batch_size):
        prediction = await mpc.output(np.argmax(L[i]))
        error = '******* ERROR *******' if prediction != labels[i] else ''
        print(f'Image #{offset+i} with label {labels[i]}: {prediction} predicted. {error}')
        print(await mpc.output(L[i]))

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

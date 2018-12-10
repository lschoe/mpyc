"""Demo Binarized Neural Network (Multilayer Perceptron) MNIST classifier.

MPyC demo accompanying the paper 'Fast Secure Comparison for Medium-Sized
Integers and Its Application in Binarized Neural Networks' by Mark Abspoel,
Niek J. Bouman, Berry Schoenmakers, and Niels de Vreede, to appear at
Cryptographer's Track RSA 2019.

Alternative to Convolutional Neural Network MNIST classifier, see cnnmnist.py.

The binarized Multilayer Perceptron (MLP) classifier consists of four
layers, where all weights are binarized, taking values either -1 or +1.
Also, the output of the activation functions (perceptrons) is -1 if the
input is negative and +1 otherwise.

The operations are performed using secure 14-bit integers. In the first
layer and in the output layer, the secure comparisons are performed
assuming the full 14-bit range for the input values. However, for the second
and third layers, the input values are assumed to be limited to a 10-bit range,
approximately. 

Functions bsgn_1(a) and bsgn_2(a) securely compute binary signs using the primes
p=9409569905028393239 and p=15569949805843283171, respectively, as the modulus 
for the underlying prime field. Function bsgn_1(a) yields the correct result 
for a in [-383, 383], and bsgn_2(a) is correct for a in [-594, 594].

Basically, bsgn_1(a) securely computes:

    (t | p), with t = sum((2(a+i)+1 | p) for i=-1,0,1),

where (t | p) = t^((p-1)/2) mod p denotes the Legendre symbol.
And, bsgn_2(a) securely computes:

    (t | p), with t = sum((2(a+i)+1 | p) for i=-2,-1,0,1,2),

Benchmarking against the built-in MPyC integer comparison is also supported.
The limited 10-bit range for the second and third layers is exploited in this
case by setting the bit length for the comparison inputs to 10.

Additionally, functions vector_bsgn_1(x), vector_bsgn_2(x),and vector_sge(x),
respectively, compute bsgn_1(a), bsgn_2(a), and 2*(a>=0)-1, for all elements a
of x in parallel. This reduces the overall time spent on integer comparisons
by a factor of 2 to 3.
"""

import os
import sys
import logging
import argparse
import gzip
import numpy as np
from mpyc.runtime import mpc
import mpyc.gmpy as gmpy2

def load_W(name):
    """Load signed binary weights for fully connected layer 'name'."""
    W = np.load(os.path.join('data', 'bnn', 'W_' + name + '.npy'))
    W = np.unpackbits(W, axis=0).tolist()
    neg_one, pos_one = secnum(-1), secnum(1)
    for w in W:
        for j in range(len(w)):
# representations neg_one and pos_one of -1 and 1 shared to avoid overhead.
            w[j] = neg_one if w[j] == 0 else pos_one # shared sharings
#            w[j] = secnum(-1) if w[j] == 0 else secnum(1) # fresh sharings
    return W

def load_b(name):
    """Load signed integer bias values for fully connected layer 'name'."""
    b = np.load(os.path.join('data', 'bnn', 'b_' + name + '.npy')).tolist()
    for i in range(len(b)):
        b[i] = secnum(int(b[i]))
    return b

@mpc.coroutine
async def bsgn_1(a):
    """Compute binary sign of a securely.

    Binary sign of a (1 if a>=0 else -1) is obtained by securely computing
    (u+v+w - u*v*w)/2 with u=(2a-1 | p), v=(2a+1 | p), and w=(2a+3 | p).

    Legendre symbols (a | p) for secret a are computed securely by evaluating
    (a s r^2 | p) in the clear for secret random sign s and secret random r modulo p,
    and outputting secret s * (a s r^2 | p).
    """
    stype = type(a)
    await mpc.returnType(stype)
    Zp = stype.field
    p = Zp.modulus
    legendre_p = lambda a: gmpy2.legendre(a.value, p)

    s = mpc.random_bits(Zp, 3, signed=True) # 3 random signs
    r = mpc._randoms(Zp, 3)
    r = mpc.schur_prod(r, r) # 3 random squares modulo p
    a, s, r = await mpc.gather(a, s, r)
    y = [b + 2 * i for b in (2 * a + 1,) for i in (-1, 0, 1)]
    y.append(s[0])
    s.append(s[1])
    r.append(s[2])
    y = await mpc.schur_prod(y, s)
    y = await mpc.output(await mpc.schur_prod(y, r))
    h = [legendre_p(y[i]) for i in range(3)]
    u, v, w = [s[i] * h[i] for i in range(3)]
    uvw = h[0] * h[1] * h[2] * y[3]
    return (u + v + w - uvw) / 2

@mpc.coroutine
async def vector_bsgn_1(x):
    """Compute bsgn_1(a) for all elements a of x in parallel."""
    stype = type(x[0])
    n = len(x)
    await mpc.returnType(stype, n)
    Zp = stype.field
    p = Zp.modulus
    legendre_p = lambda a: gmpy2.legendre(a.value, p)

    s = mpc.random_bits(Zp, 3*n, signed=True) # 3n random signs
    r = mpc._randoms(Zp, 3*n)
    r = mpc.schur_prod(r, r) # 3n random squares modulo p
    x, s, r = await mpc.gather(x, s, r)
    y = [b + 2 * i for b in (2 * a + 1 for a in x) for i in (-1, 0, 1)]
    y.extend(s[:n])
    s.extend(s[n:2*n])
    r.extend(s[-n:])
    y = await mpc.schur_prod(y, s)
    y = await mpc.output(await mpc.schur_prod(y, r))
    h = [legendre_p(y[j]) for j in range(3*n)]
    t = [s[j] * h[j] for j in range(3*n)]
    z = [h[3*j] * h[3*j+1] * h[3*j+2] * y[3*n + j] for j in range(n)]
    q = (p + 1) >> 1 # q = 1/2 mod p
    return [Zp((u.value + v.value + w.value - uvw.value)*q) for u, v, w, uvw in zip(*[iter(t)] * 3, z)]

@mpc.coroutine
async def bsgn_2(a):
    """Compute binary sign of a securely.

    Binary sign of a (1 if a>=0 else -1) is obtained by securely computing
    (t | p), with t = sum((2a+2i+1 | p) for i=-2,-1,0,1,2).
    """
    stype = type(a)
    await mpc.returnType(stype)
    Zp = stype.field
    p = Zp.modulus
    legendre_p = lambda a: gmpy2.legendre(a.value, p)

    s = mpc.random_bits(Zp, 6, signed=True) # 6 random signs
    r = mpc._randoms(Zp, 6)
    r = mpc.schur_prod(r, r) # 6 random squares modulo p
    a, s, r = await mpc.gather(a, s, r)
    y = [b + 2 * i for b in (2 * a + 1,) for i in (-2, -1, 0, 1, 2)]
    y = await mpc.schur_prod(y, s[:-1])
    y.append(s[-1])
    y = await mpc.output(await mpc.schur_prod(y, r))
    t = sum(s[i] * legendre_p(y[i]) for i in range(5))
    t = await mpc.output(t * y[-1])
    return s[-1] * legendre_p(t)

@mpc.coroutine
async def vector_bsgn_2(x):
    """Compute bsgn_2(a) for all elements a of x in parallel."""
    stype = type(x[0])
    n = len(x)
    await mpc.returnType(stype, n)
    Zp = stype.field
    p = Zp.modulus
    legendre_p = lambda a: gmpy2.legendre(a.value, p)

    s = mpc.random_bits(Zp, 6*n, signed=True) # 6n random signs
    r = mpc._randoms(Zp, 6*n)
    r = mpc.schur_prod(r, r) # 6n random squares modulo p
    x, s, r = await mpc.gather(x, s, r)
    y = [b + 2 * i for b in (2 * a + 1 for a in x) for i in (-2, -1, 0, 1, 2)]
    y = await mpc.schur_prod(y, s[:-n])
    y.extend(s[-n:])
    y = await mpc.output(await mpc.schur_prod(y, r))
    t = [sum(s[5*j + i] * legendre_p(y[5*j + i]) for i in range(5)) for j in range(n)]
    t = await mpc.output(await mpc.schur_prod(t, y[-n:]))
    return [c * legendre_p(d) for c, d in zip(s[-n:], t)]

@mpc.coroutine
async def vector_sge(x):
    """Compute binary signs securely for all elements of x in parallel.

    Vectorized version of MPyC's built-in secure comparison.
    Cf. mpc.sgn() with GE=True (and EQ=False).

    NB: mpc.prod() and mpc.is_zero_public() are not (yet) vectorized.
    """
    stype = type(x[0])
    n = len(x)
    await mpc.returnType(stype, n)
    Zp = stype.field
    l = stype.bit_length
    k = mpc.options.security_parameter

    r_bits = await mpc.random_bits(Zp, (l + 1) * n)
    r_bits = [b.value for b in r_bits]
    r_modl = [0] * n
    for j in range(n):
        for i in range(l - 1, -1, -1):
            r_modl[j] <<= 1
            r_modl[j] += r_bits[l*j + i]
    r_divl = mpc._randoms(Zp, n, 1<<k)
    x = await mpc.gather(x)
    x_r = [a + ((1<<l) + b) for a, b in zip(x, r_modl)]
    c = await mpc.output([a + (b.value << l) for a, b in zip(x_r, r_divl)])

    c = [c.value % (1<<l) for c in c]
    e = [[None] * (l + 1) for _ in range(n)]
    for j in range(n):
        s_sign = (r_bits[l*n + j] << 1) - 1
        sumXors = 0
        for i in range(l - 1, -1, -1):
            c_i = (c[j] >> i) & 1
            e[j][i] = Zp(s_sign + r_bits[l*j + i] - c_i + 3 * sumXors)
            sumXors += 1 - r_bits[l*j + i] if c_i else r_bits[l*j + i]
        e[j][l] = Zp(s_sign - 1 + 3 * sumXors)
    e = await mpc.gather([mpc.prod(_) for _ in e])
    g = await mpc.gather([mpc.is_zero_public(stype(_)) for _ in e])
    UF = [1 - b if g else b for b, g in zip(r_bits[-n:], g)]
    z = [(a - (c + (b << l))) / (1 << l - 1) - 1 for a, b, c in zip(x_r, UF, c)]
    return z

def argmax(x):
    a = type(x[0])(0)
    m = x[0]
    for i in range(1, len(x)):
        b = m >= x[i]
        a = b * (a - i) + i
        m = b * (m - x[i]) + x[i]
    return a

async def main():
    global secnum

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, metavar='B',
                        help='number of images to classify')
    parser.add_argument('-o', '--offset', type=int, metavar='O',
                        help='offset for batch (otherwise random in [0,10000-B])')
    parser.add_argument('-d', '--d-k-star', type=int, metavar='D',
                        help='k=D=1,2 for Legendre-based comparison using d_k^*')
    parser.add_argument('--no-legendre', action='store_true',
                        default=False, help='disable Legendre-based comparison')
    parser.add_argument('--no-vectorization', action='store_true',
                        default=False, help='disable vectorization of comparisons')
    parser.set_defaults(batch_size=1, offset=-1, d_k_star=1)
    args = parser.parse_args()

    batch_size = args.batch_size
    offset = args.offset
    if args.no_legendre:
        secnum = mpc.SecInt(14) # using vectorized MPyC integer comparison
    else:
        if args.d_k_star == 1:
            secnum = mpc.SecInt(14, p=9409569905028393239) # Legendre-1 range [-383, 383]
            bsgn = bsgn_1
            vector_bsgn = vector_bsgn_1
        else:
            secnum = mpc.SecInt(14, p=15569949805843283171) # Legendre-2 range [-594, 594]
            bsgn = bsgn_2
            vector_bsgn = vector_bsgn_2

    one_by_one = args.no_vectorization

    await mpc.start()

    if offset < 0:
        import mpyc.random as secrnd
        offset = int(await mpc.output(secrnd.randrange(secnum, 10001 - batch_size)))

    logging.info('--------------- INPUT   -------------')
    print(f'Type = {secnum.__name__}, range = ({offset}, {offset + batch_size})')
    # read batch_size labels and images at given offset
    df = gzip.open(os.path.join('data', 'cnn', 't10k-labels-idx1-ubyte.gz'))
    d = df.read()[8 + offset: 8 + offset + batch_size]
    labels = list(map(int, d))
    print('Labels:', labels)
    df = gzip.open(os.path.join('data', 'cnn', 't10k-images-idx3-ubyte.gz'))
    d = df.read()[16 + offset * 28**2: 16 + (offset + batch_size) * 28**2]
    L = np.array(list(d)).reshape(batch_size, 28**2)
    if batch_size == 1:
        x = np.array(L[0]).reshape(28, 28)
        print(np.vectorize(lambda a: int(bool(a / 255)))(x))

    L = np.vectorize(lambda a: secnum(int(a)))(L).tolist()

    logging.info('--------------- LAYER 1 -------------')
    logging.info('- - - - - - - - fc      - - - - - - -')
    L = mpc.matrix_prod(L, load_W('fc1'))
    L = mpc.matrix_add(L, [load_b('fc1')] * len(L))
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    if one_by_one:
        L = np.vectorize(lambda a: (a >= 0) * 2 - 1)(L).tolist()
    else:
        L = [vector_sge(_) for _ in L]
    await mpc.barrier()

    logging.info('--------------- LAYER 2 -------------')
    logging.info('- - - - - - - - fc      - - - - - - -')
    L = mpc.matrix_prod(L, load_W('fc2'))
    L = mpc.matrix_add(L, [load_b('fc2')] * len(L))
    await mpc.barrier()
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    if not secnum.__name__.endswith(')'):
        secnum.bit_length = 10
        if one_by_one:
            activate = np.vectorize(lambda a: (a >= 0) * 2 - 1)
            L = activate(L).tolist()
        else:
            L = [vector_sge(_) for _ in L]
    else:
        if one_by_one:
            activate = np.vectorize(bsgn)
            L = activate(L).tolist()
        else:
            L = [vector_bsgn(_) for _ in L]
    await mpc.barrier()

    logging.info('--------------- LAYER 3 -------------')
    logging.info('- - - - - - - - fc      - - - - - - -')
    L = mpc.matrix_prod(L, load_W('fc3'))
    L = mpc.matrix_add(L, [load_b('fc3')] * len(L))
    await mpc.barrier()
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    if not secnum.__name__.endswith(')'):
        secnum.bit_length = 10
        if one_by_one:
            activate = np.vectorize(lambda a: (a >= 0) * 2 - 1)
            L = activate(L).tolist()
        else:
            L = [vector_sge(_) for _ in L]
    else:
        if one_by_one:
            activate = np.vectorize(bsgn)
            L = activate(L).tolist()
        else:
            L = [vector_bsgn(_) for _ in L]
    await mpc.barrier()

    logging.info('--------------- LAYER 4 -------------')
    logging.info('- - - - - - - - fc      - - - - - - -')
    L = mpc.matrix_prod(L, load_W('fc4'))
    L = mpc.matrix_add(L, [load_b('fc4')] * len(L))
    await mpc.barrier()

    logging.info('--------------- OUTPUT  -------------')
    if not secnum.__name__.endswith(')'):
        secnum.bit_length = 14
    for i in range(batch_size):
        prediction = await mpc.output(argmax(L[i]))
        error = '******* ERROR *******' if prediction != labels[i] else ''
        print(f'Image #{offset+i} with label {labels[i]}: {prediction} predicted. {error}')
        print(await mpc.output(L[i]))

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

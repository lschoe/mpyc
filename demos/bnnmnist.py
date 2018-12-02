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
or more precisely, to the range [-493, 493]. The function vector_bsgn() is
used to efficiently compare inputs from this range with 0.

Function vector_bsgn(x) securely computes binary signs using the prime
p = 13835556230699448671 as the modulus for the underlying prime field.
Basically, vector_bsgn(x) securely computes, for all a in x:

    (t | p), with t = sum((2(a+i)+1 | p) for i=-2,-1,0,1,2),

where (a | p ) = a^((p-1)/2) mod p denotes the Legendre symbol.

Benchmarking against the built-in MPyC integer comparison is also supported.
The limited 10-bit range for the second and third layers is exploited in this
case by setting the bit length for the comparison inputs to 10. Note, however,
that built-in MPyC comparisons do not (yet;) apply vectorization.
"""

import os
import sys
import logging
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
async def vector_bsgn(x):
    """Compute binary signs securely for all elements of x.

    Binary sign of a (1 if a>=0 else -1) is obtained by securely computing
    (t | p), with t = sum((2(a+i)+1 | p) for i=-2,-1,0,1,2).

    Legendre symbols (a | p) for secret a are computed securely by evaluating
    (a s r^2 | p) in the clear for secret random sign s and secret random r modulo p,
    and outputting secret s * (a s r^2 | p).
    """
    stype = type(x[0])
    n = len(x)
    await mpc.returnType(stype, n)
    p = stype.field.modulus
    legendre_p = lambda a: gmpy2.legendre(a.value, p)

    s = mpc.random_bits(stype, 6*n, signed=True) # 6n random signs
    r = mpc._randoms(stype, 6*n)
    r = mpc.schur_prod(r, r) # 6n random squares modulo p
    y = [b + i for b in [a * 2 + 1 for a in x] for i in (-4, -2, 0, 2, 4)]
    y = mpc.schur_prod(y, s[:-n])
    y.extend(s[-n:])
    y = await mpc.output(mpc.schur_prod(y, r))
    t = [mpc.sum([s[5*j + i] * legendre_p(y[5*j + i]) for i in range(5)]) for j in range(n)]
    t = await mpc.output(mpc.schur_prod(t, y[-n:]))
    return [c * legendre_p(d) for c, d in zip(s[-n:], t)]

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

    k = 1 if len(sys.argv) == 1 else int(sys.argv[1])
    if k < 0:
        secnum = mpc.SecInt(14, p=13835556230699448671) # Legendre-2 range [-493, 493]
    else:
        secnum = mpc.SecInt(14) # using built-in MPyC integer comparison
    batch_size = abs(k)

    await mpc.start()

    if len(sys.argv) <= 2:
        import mpyc.random as secrnd
        offset = await mpc.output(secrnd.randrange(secnum, 10001 - batch_size))
    else:
        offset = sys.argv[2]
    offset = int(offset)

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
    L = np.vectorize(lambda a: (a >= 0) * 2 - 1)(L).tolist()
    await mpc.barrier()

    logging.info('--------------- LAYER 2 -------------')
    logging.info('- - - - - - - - fc      - - - - - - -')
    L = mpc.matrix_prod(L, load_W('fc2'))
    L = mpc.matrix_add(L, [load_b('fc2')] * len(L))
    await mpc.barrier()
    logging.info('- - - - - - - - bsgn    - - - - - - -')
    if not secnum.__name__.endswith(')'):
        secnum.bit_length = 10
        activate = np.vectorize(lambda a: (a >= 0) * 2 - 1)
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
        activate = np.vectorize(lambda a: (a >= 0) * 2 - 1)
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

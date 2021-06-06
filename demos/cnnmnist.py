"""Demo Convolutional Neural Network (CNN) MNIST classifier.

The MNIST dataset of handwritten digits consists of a training set of
60,000 images of a test set of 10,000 images. The training images have been
used in the clear to obtain a highly reliable CNN classifier. The demo
feeds the classifier with random test images keeping both the CNN parameters
(neuron weights and bias for all layers) and the test image secret.

The secure CNN classifier is run either with scaled secure integers or
with secure fixed-point numbers. Barriers are used to throttle the MPyC
secure computation (reduces memory usage).
"""

import os
import sys
import logging
import random
import gzip
import numpy as np
from mpyc.runtime import mpc

secnum = None


def scale_to_int(f):
    if issubclass(secnum, mpc.Integer):
        scale = lambda a: secnum(int(round(a * f)))  # force Python integers
    else:
        scale = lambda a: secnum(float(a))  # force Python floats
    return np.vectorize(scale)


def load(name, f, a=2):
    W = np.load(os.path.join('data', 'cnn', 'W_' + name + '.npy'))
    W = scale_to_int(1 << f)(W)
    b = np.load(os.path.join('data', 'cnn', 'b_' + name + '.npy'))
    b = scale_to_int(1 << (a * f))(b)
    return W, b


def dim(x):  # dimensions of tensor x
    if isinstance(x, np.ndarray):
        s = list(x.shape)
    else:
        s = []
        while isinstance(x, list):
            s.append(len(x))
            x = x[0]
    return s


@mpc.coroutine
async def convolvetensor(x, W, b):
    logging.info('- - - - - - - - conv2d  - - - - - - -')
    # 2D convolutions on m*n sized images from X with s*s sized filters from W.
    # b is of dimension v
    k, r, m, n = dim(x)
    x = x.tolist()
    v, r, s, s = dim(W)
    W = W.tolist()
    stype = type(x[0][0][0][0])
    await mpc.returnType(stype, k, v, m, n)
    x, W, b = await mpc.gather(x, W, b)
    Y = [[[[b[j]]*m for _ in range(n)] for j in range(v)] for _ in range(k)]
    counter = 0
    for i in range(k):
        for j in range(v):
            for l in range(r):
                counter += 1
                if counter % 500 == 0:
                    await mpc.barrier()
                Y[i][j] = mpc.matrix_add(Y[i][j], inprod2D(x[i][l], W[j][l]))
    for i in range(k):
        for j in range(v):
            for im in range(m):
                Y[i][j][im] = mpc._reshare(Y[i][j][im])
    Y = await mpc.gather(Y)
    if issubclass(stype, mpc.FixedPoint):
        l = stype.bit_length
        Y = [[[mpc.trunc(y, f=stype.frac_length, l=l) for y in _] for _ in _] for _ in Y]
    Y = await mpc.gather(Y)
    return Y
    # k, v, m, n = dim(Y)


def inprod2D(X, W):
    m, n = dim(X)
    s = len(W)  # s * s filter W
    s2 = (s-1) // 2
    Y = [None] * m
    for i in range(m):
        Y[i] = [None] * n
        for j in range(n):
            t = 0
            ix = i - s2
            for di in range(s):
                if 0 <= ix < m:
                    jx = j - s2
                    for dj in range(s):
                        if 0 <= jx < n:
                            t += X[ix][jx].value * W[di][dj].value
                        jx += 1
                ix += 1
            Y[i][j] = t
    return Y


def tensormatrix_prod(x, W, b):
    logging.info('- - - - - - - - fc      - - - - - - -')
    W, b = W.tolist(), b.tolist()
    return [mpc.vector_add(mpc.matrix_prod([z.tolist()], W)[0], b) for z in x]


def maxpool(x):
    logging.info('- - - - - - - - maxpool - - - - - - -')
    # maxpooling 2 * 2 squares in images of size m * n with stride 2
    k, r, m, n = dim(x)
    Y = [[[[mpc.max(y[i][j], y[i][j+1], y[i+1][j], y[i+1][j+1])
            for j in range(0, n, 2)] for i in range(0, m, 2)]
          for y in z] for z in x]
    return np.array(Y)


def ReLU(x):
    logging.info('- - - - - - - - ReLU    - - - - - - -')
    return np.vectorize(lambda a: (a >= 0) * a)(x)


async def main():
    global secnum

    k = 1 if len(sys.argv) == 1 else float(sys.argv[1])
    if k - int(k) == 0.5:
        secnum = mpc.SecFxp(10, 4)
    else:
        secnum = mpc.SecInt(37)
    batch_size = round(k - 0.01)

    await mpc.start()

    if len(sys.argv) <= 2:
        offset = random.randrange(10001 - batch_size) if mpc.pid == 0 else None
        offset = await mpc.transfer(offset, senders=0)
    else:
        offset = int(sys.argv[2])

    f = 6

    logging.info('--------------- INPUT   -------------')
    print(f'Type = {secnum.__name__}, range = ({offset}, {offset + batch_size})')
    # read batch_size labels and images at given offset
    df = gzip.open(os.path.join('data', 'cnn', 't10k-labels-idx1-ubyte.gz'))
    d = df.read()[8 + offset: 8 + offset + batch_size]
    labels = list(map(int, d))
    print('Labels:', labels)
    df = gzip.open(os.path.join('data', 'cnn', 't10k-images-idx3-ubyte.gz'))
    d = df.read()[16 + offset * 28**2: 16 + (offset + batch_size) * 28**2]
    x = list(map(lambda a: a/255, d))
    x = np.array(x).reshape(batch_size, 1, 28, 28)
    if batch_size == 1:
        print(np.array2string(np.vectorize(lambda a: int(bool(a)))(x[0, 0]), separator=''))
    x = scale_to_int(1 << f)(x)

    logging.info('--------------- LAYER 1 -------------')
    W, b = load('conv1', f)
    x = convolvetensor(x, W, b)
    await mpc.barrier()
    if issubclass(secnum, mpc.Integer):
        secnum.bit_length = 16
    x = maxpool(x)
    await mpc.barrier()
    x = ReLU(x)
    await mpc.barrier()

    logging.info('--------------- LAYER 2 -------------')
    W, b = load('conv2', f, 3)
    x = convolvetensor(x, W, b)
    await mpc.barrier()
    if issubclass(secnum, mpc.Integer):
        secnum.bit_length = 23
    x = maxpool(x)
    await mpc.barrier()
    x = ReLU(x)
    await mpc.barrier()

    logging.info('--------------- LAYER 3 -------------')
    x = x.reshape(batch_size, 64 * 7**2)
    W, b = load('fc1', f, 4)
    x = tensormatrix_prod(x, W, b)
    if issubclass(secnum, mpc.Integer):
        secnum.bit_length = 30
    x = ReLU(x)
    await mpc.barrier()

    logging.info('--------------- LAYER 4 -------------')
    W, b = load('fc2', f, 5)
    x = tensormatrix_prod(x, W, b)

    logging.info('--------------- OUTPUT  -------------')
    if issubclass(secnum, mpc.Integer):
        secnum.bit_length = 37
    for i in range(batch_size):
        prediction = int(await mpc.output(mpc.argmax(x[i])[0]))
        error = '******* ERROR *******' if prediction != labels[i] else ''
        print(f'Image #{offset+i} with label {labels[i]}: {prediction} predicted. {error}')
        print(await mpc.output(x[i]))

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

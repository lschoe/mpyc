"""Demo CNN MNIST classifier, vectorized.

This demo is a fully equivalent reimplementation of the cnnmnist.py demo.
However, this time using secure number arrays to perform all computations
in a vectorized manner. For example, the ReLU activation at the end of a layer
is done simply by updating state x to (x >= 0) * x. Here, x is a secure
number (integer/fixed-point) array, which is first tested elementwise for
nonnegativity; the resulting array of secret-shared bits is then multiplied
elementwise with x itself.

Apart from the simplicity, the performance for these array operations is
also much better than what one gets for (nested) lists of secure numbers.
Basically, the work and especially the communication is done in large batches,
processing all array elements in one go. In contrast, for (nested) lists
the elements are processed one by one in an uncoordinated manner, each time
doing a small bit of work and communication.

For this demo, we see an 18-fold speedup when run with three parties on
local host (using secure fixed-point arithmetic). Also, the memory consumption
is reduced accordingly, e.g., by a factor of around 10 when run with all
barriers disabled.

See cnnmnist.py for background information on the CNN MNIST classifier.
"""

import os
import sys
import logging
import random
import gzip
import numpy as np
from mpyc.runtime import mpc

secnum = None


def scale_int(x, f2):
    return np.vectorize(round)(x * f2)


def load(name, f, a=2):
    W = np.load(os.path.join('data', 'cnn', 'W_' + name + '.npy'))
    if name.startswith('conv'):
        W = np.flip(W, axis=3)  # for use with np.convolve()
    b = np.load(os.path.join('data', 'cnn', 'b_' + name + '.npy'))
    if issubclass(secnum, mpc.SecureInteger):
        W = secnum.array(scale_int(W, 1 << f))
        b = secnum.array(scale_int(b, 1 << (a * f)))
    else:
        W = secnum.array(W, integral=False)
        b = secnum.array(b, integral=False)
    return W, b


@mpc.coroutine
async def convolvetensor(x, W, b):
    # 2D convolutions on (m,n)-shape images from x with (s,s)-shape filters from W.
    # b is a vector of dimension v
    stype = type(x)
    k, r, m, n = x.shape
    v, r, s, s = W.shape
    shape = (k, v, m, n)
    if issubclass(stype, mpc.SecureIntegerArray):
        rettype = (stype, shape)
    else:
        rettype = (stype, x.integral and W.integral and b.integral, shape)
    await mpc.returnType(rettype)

    x, W, b = await mpc.gather(x, W, b)
    x, W, b = x.value, W.value, b.value
    Y = np.zeros(shape, dtype=object)
    s2 = (s-1) // 2
    for i in range(k):
        for j in range(v):
            Z = np.empty((m, n), dtype=object)
            for l in range(r):
                for I in range(m):
                    i_s2 = I - s2
                    Z[I] = sum(np.convolve(x[i, l, i_d], W[j, l, i_d - i_s2], mode='same')
                               for i_d in range(max(0, i_s2), min(i_s2+s, m)))
                Y[i, j] += Z
    Y += b[:, np.newaxis, np.newaxis]
    Y = stype.sectype.field.array(Y)
    Y = mpc._reshare(Y)
    if issubclass(stype, mpc.SecureFixedPointArray):
        Y = mpc.np_trunc(stype(Y, shape=shape))
    return Y


def maxpool(x):
    # maxpooling (2,2)-squares in (m,n)-shape images from x with stride 2
    x = np.maximum(x[:, :, ::2, :], x[:, :, 1::2, :])  # m //= 2
    x = np.maximum(x[:, :, :, ::2], x[:, :, :, 1::2])  # n //= 2
    return x


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
    x = np.frombuffer(d, dtype=np.ubyte) / 255
    x = np.reshape(x, (batch_size, 1, 28, 28))
    if batch_size == 1:
        print(np.array2string(np.vectorize(lambda a: int(bool(a)))(x[0, 0]), separator=''))
    if issubclass(secnum, mpc.SecureInteger):
        x = secnum.array(scale_int(x, 1 << f))
    else:
        x = secnum.array(x, integral=False)

    logging.info('--------------- LAYER 1 -------------')
    W, b = load('conv1', f)
    logging.info('- - - - - - - - conv2d  - - - - - - -')
    x = convolvetensor(x, W, b)
    if issubclass(secnum, mpc.SecureInteger):
        secnum.bit_length = 16
    logging.info('- - - - - - - - maxpool - - - - - - -')
    x = maxpool(x)
    logging.info('- - - - - - - - ReLU    - - - - - - -')
    x = (x >= 0) * x
    await mpc.barrier('after-layer-1')

    logging.info('--------------- LAYER 2 -------------')
    W, b = load('conv2', f, 3)
    logging.info('- - - - - - - - conv2d  - - - - - - -')
    x = convolvetensor(x, W, b)
    if issubclass(secnum, mpc.SecureInteger):
        secnum.bit_length = 23
    logging.info('- - - - - - - - maxpool - - - - - - -')
    x = maxpool(x)
    logging.info('- - - - - - - - ReLU    - - - - - - -')
    x = (x >= 0) * x
    await mpc.barrier('after-layer-2')

    x = x.reshape(batch_size, 64 * 7**2)

    logging.info('--------------- LAYER 3 -------------')
    W, b = load('fc1', f, 4)
    logging.info('- - - - - - - - fc 3136 x 1024  - - -')
    x = x @ W + b
    if issubclass(secnum, mpc.SecureInteger):
        secnum.bit_length = 30
    logging.info('- - - - - - - - ReLU    - - - - - - -')
    x = (x >= 0) * x
    await mpc.barrier('after-layer-3')

    logging.info('--------------- LAYER 4 -------------')
    W, b = load('fc2', f, 5)
    logging.info('- - - - - - - - fc 1024 x 10  - - - -')
    x = x @ W + b

    logging.info('--------------- OUTPUT  -------------')
    if issubclass(secnum, mpc.SecureInteger):
        secnum.bit_length = 37

    for i in range(batch_size):
        prediction = int(await mpc.output(np.argmax(x[i])))
        error = '******* ERROR *******' if prediction != labels[i] else ''
        print(f'Image #{offset+i} with label {labels[i]}: {prediction} predicted. {error}')
        print(await mpc.output(x[i]))

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

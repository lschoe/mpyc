import os
import logging
import argparse
import asyncio
from mpyc.runtime import mpc

def load(filename):
    attr_ranges = None
    attr_names = None
    transactions = []
    comment_sign = '#'
    sep = ','
    for line in open(os.path.join('data', 'id3', filename), 'r'):
        # strip comments and whitespace and skip empty lines
        line = line.split(comment_sign)[0].strip()
        if line:
            if not attr_ranges:
                attr_ranges = list(map(int, line.split(sep)))
            elif not attr_names:
                attr_names = line.split(sep)
            else:
                transactions.append(list(map(int, line.split(sep))))
    return attr_ranges, attr_names, transactions

def argmax(xs, arg_ge):
    n = len(xs)
    if n == 1:
        return secint(0), xs[0] #ToDo: get rid of Zp here
    i0, max0 = argmax(xs[:n//2], arg_ge)
    i1, max1 = argmax(xs[n//2:], arg_ge)
    b, mx = arg_ge(max0, max1)
    return i0 + b * (n//2 + i1 - i0), mx

def argmax_int(xs):
    def arg_ge_int(x0, x1):
        b = x0 <= x1
        mx = b * (x1 - x0) + x0
        return b, mx
    return argmax(xs, arg_ge_int)

def max_rat(xs):
    def arg_ge_rat(x0, x1):
        (n0, d0) = x0
        (n1, d1) = x1
        b = mpc.in_prod([n0, d0], [d1, -n1]) <= 0
        h = mpc.scalar_mul(b, [n1 - n0, d1 - d0])
        return b, (h[0] + n0, h[1] + d0)
    return argmax(xs, arg_ge_rat)[0]

@mpc.coroutine
async def id3(T, R) -> asyncio.Future:
    sizes = [mpc.in_prod(T, v) for v in S[C]]
    i, mx = argmax_int(sizes)
    sizeT = mpc.sum(sizes)
    stop = (sizeT <= int(0.05*len(T))) + (mx == sizeT)
    if not (R and await mpc.is_zero_public(stop)):
        i = (await mpc.output(i)).value
        logging.info('Leaf node label %d' % i)
        return i
    else:
        T_R = [[mpc.schur_prod(T, v) for v in S[A]] for A in R]
        gains = [GI(mpc.matrix_prod(T_A, S[C], True)) for T_A in T_R]
        k = (await mpc.output(max_rat(gains))).value
        T_Rk = T_R[k]; T_R = gains = None # release memory
        A = list(R)[k]
        logging.info('Attribute node %d' % A)
#        trees = await mpc.gather([id3(t, R.difference([A])) for t in T_Rk])
        trees = [await id3(t, R.difference([A])) for t in T_Rk]
        return attr_names[A], trees

def GI(x):
    y = [8 * s + 1 for s in map(mpc.sum, x)] #s + (s == 0)
    D = mpc.prod(y)
    G = mpc.in_prod(list(map(mpc.in_prod, x, x)), list(map(lambda x: 1/x, y)))
    return (D * G, D)

def main():
    global secint, S, C, attr_names

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Input data.')
    parser.add_argument('options', nargs='*')
    parser.set_defaults(data='tennis')
    args = parser.parse_args(mpc.args)

    secint = mpc.SecInt()

    attr_ranges, attr_names, transactions = load(args.data + '.csv')
    n = len(attr_ranges)
    S = [[[secint(t[i] == j) for t in transactions]
          for j in range(attr_ranges[i])]
         for i in range(n)]
    C = n - 1
    T = [secint(1)] * len(transactions)

    mpc.start()

    tree = mpc.run(id3(T, frozenset(list(range(n))).difference([C])))
    print('Tree =', tree)

    height = lambda t: max(list(map(height, t[1])))+1 if isinstance(t, tuple) else 0
    print('Tree height =', height(tree))

    size = lambda t: sum(map(size, t[1]))+1 if isinstance(t, tuple) else 1
    print('Tree size =', size(tree))

    mpc.shutdown()

if __name__ == '__main__':
    main()

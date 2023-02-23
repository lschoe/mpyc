"""Demo decision tree learning using ID3, vectorized.

This demo is a fully equivalent reimplementation of the id3gini.py demo,
using secure integer arrays for NumPy-based vectorized computation.

Performance improvement of over 9x speedup when run with three parties
on local host. Memory consumption is reduced accordingly.

See demo id3gini.py for more information on decision tree learning and ID3.
"""

import os
import logging
import argparse
import csv
import asyncio
import numpy as np
from mpyc.runtime import mpc


@mpc.coroutine
async def id3(T, R) -> asyncio.Future:
    sizes = S[C] @ T
    i, mx = sizes.argmax(arg_unary=False)
    sizeT = sizes.sum()
    stop = (sizeT <= int(args.epsilon * len(T))) + (mx == sizeT)
    if not (R and await mpc.is_zero_public(stop)):
        i = await mpc.output(i)
        logging.info(f'Leaf node label {i}')
        tree = i
    else:
        T_SC = (T * S[C]).T
        CT = np.stack(tuple(GI(S[A] @ T_SC) for A in R))
        k = CT.argmax(key=SecureFraction, arg_unary=False, arg_only=True)
        A = list(R)[await mpc.output(k)]
        logging.info(f'Attribute node {A}')
        T_SA = T * S[A]
        if args.parallel_subtrees:
            subtrees = await mpc.gather([id3(Tj, R.difference([A])) for Tj in T_SA])
        else:
            subtrees = [await id3(Tj, R.difference([A])) for Tj in T_SA]
        tree = A, subtrees
    return tree


def GI(x):
    """Gini impurity for contingency table x."""
    y = args.alpha * np.sum(x, axis=1) + 1  # NB: alternatively, use s + (s == 0)
    D = mpc.prod(y.tolist())
    G = np.sum(np.sum(x * x, axis=1) / y)
    return mpc.np_fromlist([D * G, D])  # numerator, denominator


class SecureFraction:

    size = 2  # __lt__() assumes last dimension of size 2

    def __init__(self, a):
        self.a = a  # numerator, denominator

    def __lt__(self, other):  # NB: __lt__() is basic comparison as in Python's list.sort()
        return self.a[..., 0] * other.a[..., 1] < self.a[..., 1] * other.a[..., 0]


depth = lambda tree: 0 if isinstance(tree, int) else max(map(depth, tree[1])) + 1

size = lambda tree: 1 if isinstance(tree, int) else sum(map(size, tree[1])) + 1


def pretty(prefix, tree, names, ranges):
    """Convert raw tree into multiline textual representation, using attribute names and values."""
    if isinstance(tree, int):  # leaf
        return ranges[C][tree]

    A, subtrees = tree
    s = ''
    for a, t in zip(ranges[A], subtrees):
        s += f'\n{prefix}{names[A]} == {a}: {pretty("|   " + prefix, t, names, ranges)}'
    return s


async def main():
    global args, secint, C, S

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset', type=int, metavar='I',
                        help=('dataset 0=tennis (default), 1=balance-scale, 2=car, '
                              '3=SPECT, 4=KRKPA7, 5=tic-tac-toe, 6=house-votes-84'))
    parser.add_argument('-l', '--bit-length', type=int, metavar='L',
                        help='override preset bit length for dataset')
    parser.add_argument('-e', '--epsilon', type=float, metavar='E',
                        help='minimum fraction E of samples for a split, 0.0<=E<=1.0')
    parser.add_argument('-a', '--alpha', type=int, metavar='A',
                        help='scale factor A to prevent division by zero, A>=1')
    parser.add_argument('--parallel-subtrees', action='store_true',
                        help='process subtrees in parallel (rather than in series)')
    parser.add_argument('--no-pretty-tree', action='store_true',
                        help='print raw flat tree instead of pretty tree')
    parser.set_defaults(dataset=0, bit_length=0, alpha=8, epsilon=0.05)
    args = parser.parse_args()

    settings = [('tennis', 32), ('balance-scale', 77), ('car', 95),
                ('SPECT', 42), ('KRKPA7', 69), ('tic-tac-toe', 75), ('house-votes-84', 62)]
    name, bit_length = settings[args.dataset]
    if args.bit_length:
        bit_length = args.bit_length
    secint = mpc.SecInt(bit_length)
    print(f'Using secure integers: {secint.__name__}')

    with open(os.path.join('data', 'id3', name + '.csv')) as file:
        reader = csv.reader(file)
        attr_names = next(reader)
        C = 0 if attr_names[0].lower().startswith('class') else len(attr_names)-1  # class attribute
        transactions = list(reader)
    n, d = len(transactions), len(attr_names)
    attr_ranges = [sorted(set(t[i] for t in transactions)) for i in range(d)]
    # one-hot encoding of attributes:
    S = [secint.array(np.array([[t[i] == j for t in transactions] for j in attr_ranges[i]]))
         for i in range(d)]
    T = secint.array(np.ones(n, dtype='O'))
    print(f'dataset: {name} with {n} samples and {d-1} attributes')

    await mpc.start()
    tree = await id3(T, frozenset(range(d)).difference([C]))
    await mpc.shutdown()

    print(f'Decision tree of depth {depth(tree)} and size {size(tree)}: ', end='')
    if args.no_pretty_tree:
        print(tree)
    else:
        print(pretty('if ', tree, attr_names, attr_ranges))


if __name__ == '__main__':
    mpc.run(main())

"""Demo decision tree learning using ID3.

This demo implements a slight variant of Protocol 4.1 from the paper
'Practical Secure Decision Tree Learning in a Teletreatment Application'
by Sebastiaan de Hoogh, Berry Schoenmakers, Ping Chen, and Harm op den Akker,
which appeared at the 18th International Conference on Financial Cryptography
and Data Security (FC 2014), LNCS 8437, pp. 179-194, Springer.
See https://doi.org/10.1007/978-3-662-45472-5_12 (or,
see https://fc14.ifca.ai/papers/fc14_submission_103.pdf or,
see https://www.researchgate.net/publication/295148009).

ID3 is a recursive algorithm for generating decision trees from a database
of samples (or, transactions). The first or last attribute of each sample
is assumed to be the class attribute, according to which the samples are
classified. Gini impurity is used to determine the best split. See also
https://en.wikipedia.org/wiki/ID3_algorithm.

The samples are assumed to be secret, while the resulting decision tree is
public. All calculations are performed using secure integer arithmetic only.

The demo includes the following datasets (datasets 1-4 are used in the paper):

  0=tennis: classic textbook example
  1=balance-scale: balance scale weight & distance database
  2=car: car evaluation database
  3=SPECT: Single Proton Emission Computed Tomography train+test heart images
  4=KRKPA7: King+Rook versus King+Pawn-on-A7 chess endgame
  5=tic-tac-toe: Tic-Tac-Toe endgame
  6=house-votes-84: 1984 US congressional voting records

The numbers 0-6 can be used with the command line option -i of the demo.

Three command line options are provided for controlling accuracy:

  -l L, --bit-length: overrides the preset bit length for a dataset
  -e E, --epsilon: minimum number of samples E required for attribute nodes,
                   represented as a fraction of all samples, 0.0<=E<=1.0
  -a A, --alpha: scale factor A to avoid division by zero when calculating
                 Gini impurities, basically, by adding 1/A to denominators, A>=1

Setting E=1.0 yields a trivial tree consisting of a single leaf node, while
setting E=0.0 yields a large (overfitted) tree. Default value E=0.05 yields
trees of reasonable complexity. Default value A=8 is sufficiently large for
most datasets. Note that if A is increased, L should be increased accordingly.

Finally, the command line option --parallel-subtrees can be used to let the
computations of the subtrees of an attribute node be done in parallel. The
default setting is to compute the subtrees one after another (in series).
The increase in the level of parallelism, however, comes at the cost of
a larger memory footprint.

Interestingly, one will see that the order in which the nodes are added to
the tree will change if the --parallel-subtrees option is used. The resulting
tree is still the same, however. Of course, this only happens if the MPyC
program is actually run with multiple parties (or, if the -M1 option is used).
"""

import os
import logging
import argparse
import csv
import asyncio
from mpyc.runtime import mpc


@mpc.coroutine
async def id3(T, R) -> asyncio.Future:
    sizes = [mpc.in_prod(T, v) for v in S[C]]
    i, mx = mpc.argmax(sizes)
    sizeT = mpc.sum(sizes)
    stop = (sizeT <= int(args.epsilon * len(T))) + (mx == sizeT)
    if not (R and await mpc.is_zero_public(stop)):
        i = await mpc.output(i)
        logging.info(f'Leaf node label {i}')
        tree = i
    else:
        T_R = [[mpc.schur_prod(T, v) for v in S[A]] for A in R]
        gains = [GI(mpc.matrix_prod(T_A, S[C], True)) for T_A in T_R]
        k = await mpc.output(mpc.argmax(gains, key=SecureFraction)[0])
        T_Rk = T_R[k]
        del T_R, gains  # release memory
        A = list(R)[k]
        logging.info(f'Attribute node {A}')
        if args.parallel_subtrees:
            subtrees = await mpc.gather([id3(Tj, R.difference([A])) for Tj in T_Rk])
        else:
            subtrees = [await id3(Tj, R.difference([A])) for Tj in T_Rk]
        tree = A, subtrees
    return tree


def GI(x):
    """Gini impurity for contingency table x."""
    y = [args.alpha * s + 1 for s in map(mpc.sum, x)]  # NB: alternatively, use s + (s == 0)
    D = mpc.prod(y)
    G = mpc.in_prod(list(map(mpc.in_prod, x, x)), list(map(lambda x: 1/x, y)))
    return [D * G, D]  # numerator, denominator


class SecureFraction:
    def __init__(self, a):
        self.n, self.d = a  # numerator, denominator

    def __lt__(self, other):  # NB: __lt__() is basic comparison as in Python's list.sort()
        return mpc.in_prod([self.n, -self.d], [other.d, other.n]) < 0


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
                        default=False, help='process subtrees in parallel (rather than in series)')
    parser.add_argument('--no-pretty-tree', action='store_true',
                        default=False, help='print raw flat tree instead of pretty tree')
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
    S = [[[secint(t[i] == j) for t in transactions] for j in attr_ranges[i]] for i in range(d)]
    T = [secint(1)] * n  # bit vector representation
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

"""This module currently provides alternative implementations for two
functions in Python's itertools and functools modules, respectively.

The alternative implementations can be used as drop-in replacements, however,
potentially enhancing the performance when used in secure computations. More
specifically, these implementations are aimed at reducing the overall round
complexity, possibly at the expense of increasing overall space complexity,
time complexity, and communication complexity.
"""

import operator

runtime = None

_no_value = type('mpyc.mpctools.NoValueType', (object,), {'__repr__': lambda self: '<no value>'})()
_no_value.__doc__ = 'Represents "empty" value, different from any other object including None.'


def reduce(f, x, initial=_no_value):
    """Apply associative function f of two arguments to the items of iterable x.

    The applications of f are arranged in a binary tree of logarithmic depth,
    thus limiting the overall round complexity of the secure computation.

    In contrast, Python's functools.reduce() higher-order function arranges
    the applications of f in a linear chain (a binary tree of linear depth),
    and in this case f is not required to be associative; the arguments to f
    may even be of different types.

    If initial is provided (possibly equal to None), it is placed before the
    items of x (hence effectively serves as a default when x is empty). If no
    initial value is given and x contains only one item, that item is returned.
   """
    x = list(x)
    if initial is not _no_value:
        x.insert(0, initial)
    if not x:
        raise TypeError('reduce() of empty sequence with no initial value')

    while len(x) > 1:
        x[len(x)%2:] = (f(x[i], x[i+1]) for i in range(len(x)%2, len(x), 2))
    return x[0]


def accumulate(x, f=operator.add, initial=_no_value):
    """For associative function f of two arguments, make an iterator that returns
    the accumulated results over all (nonempty) prefixes of the given iterable x.

    The applications of f are arranged such that the maximum depth is logarithmic
    in the number of elements of x, potentially at the cost of increasing the total
    number of applications of f by a logarithmic factor as well.

    In contrast, Python's itertools.accumulate() higher-order function arranges
    the applications of f in a linear fashion, as in general it cannot be assumed
    that f is associative (and that the arguments to f are even of the same type).

    If initial is provided (possibly equal to None), the accumulation leads off
    with this initial value so that the output has one more element than the input
    iterable. Otherwise, the number of elements output matches the input iterable x.
    """
    x = list(x)
    if initial is not _no_value:
        x.insert(0, initial)
    n = len(x)
    if runtime.options.no_prss and n >= 32:
        # Minimize f-complexity of acc(0, n) a la Brent-Kung.
        # For n=2^k, k>=0: f-complexity=2n-2-k calls, f-depth=max(2k-2, k) rounds.
        def acc(i, j):
            h = (i + j)//2
            if i < h:
                acc(i, h)
                a = x[h-1]
                if i:
                    x[h-1] = f(x[i-1], a)
                acc(h, j)
                x[j-1] = f(a, x[j-1])
    else:
        # Minimize f-depth of acc(0, n) a la Sklansky.
        # For n=2^k, k>=0: f-complexity=(n/2)k calls, f-depth=k rounds.
        def acc(i, j):
            h = (i + j)//2
            if i < h:
                acc(i, h)
                a = x[h-1]
                acc(h, j)
                x[h:j] = (f(a, b) for b in x[h:j])

    acc(0, n)
    return iter(x)

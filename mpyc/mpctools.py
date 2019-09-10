"""This module currently provides alternative implementations for two
functions in Python's itertools and functools modules, respectively.

The alternative implementations can be used as drop-in replacements, however,
potentially enhancing the performance when used in secure computations. More
specifically, these implementations are aimed at reducing the overall round
complexity, possible at the expense of increasing overall space complexity,
time complexity, and communication complexity.
"""
import operator


def reduce(f, x, iv=None):
    """Apply associative function f of two arguments to the items of iterable x.

    The applications of f are arranged in a binary tree of logarithmic depth,
    thus limiting the overall round complexity of the secure computation.

    In contrast, Python's functools.reduce() higher-order function arranges
    the applications of f in a linear chain (a binary tree of linear depth),
    and in this case f is not required to be associative; the arguments to f
    may even be of different types.

    If iv is provided, it is placed before the items of x (hence effectively
    serves as a default when x is empty). If iv is not given and x contains
    only one item, that item is returned.
   """
    x = list(x)
    if iv is not None:
        x.insert(0, iv)
    while len(x) > 1:
        x[len(x)%2:] = (f(x[i], x[i+1]) for i in range(len(x)%2, len(x), 2))
    return x[0]


def accumulate(x, f=operator.add, iv=None):
    """For associative function f of two arguments, make an iterator that returns
    the accumulated results over all (nonempty) prefixes of the given iterable x.

    The applications of f are arranged such that the maximum depth is logarithmic
    in the number of elements of x, at the cost of increasing the total number of
    applications of f by a logarithmic factor as well.

    In contrast, Python's itertools.accumulate() higher-order function arranges
    the applications of f in a linear fashion, as in general it cannot be assumed
    that f is associative (and that the arguments to f are even of the same type).

    If iv is provided, the accumulation leads off with this initial value so that
    the output has one more element than the input iterable. Otherwise, the number
    of elements output matches the input iterable x.
    """
    x = list(x)
    if iv is not None:
        x.insert(0, iv)

    def acc(i, j):
        if j == i+1:
            return x[i:j]

        h = (i + j)//2
        y = acc(i, h)
        a = y[-1]
        y.extend(f(a, b) for b in acc(h, j))
        return y

    yield from acc(0, len(x))

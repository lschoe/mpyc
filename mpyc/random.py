"""This module provides secure versions of several functions for
generating pseudorandom numbers, cf. the random module of Python's
standard library. Each function behaves like its Python counterpart,
except that a secure type is required as additional (first) argument.

Additionally, random_unit_vector() generates a random bit vector
with exactly one bit set to 1, using approximately log_2 n secure
random bits for a bit vector of length n.

Also, random_permutation() is provided as a convenience function.

Main concern for the implementations is to minimize the randomness
complexity, that is, to limit the usage of secure random bits as
provided by runtime.random_bits(). Other than this, the code is
relatively simple for now.

NB: runtime._random(sectype, n) cannot be used as an alternative to
_randbelow(sectype, n) as its output is not uniformly random, except
when n is equal to the order of sectype's finite field.
"""

import math
import functools
import itertools
from mpyc import asyncoro

runtime = None

def getrandbits(sectype, k):
    """Uniformly random nonnegative k-bit integer value."""
    x = runtime.random_bits(sectype, k)
    return runtime.from_bits(x)

@asyncoro.mpc_coro
async def _randbelow(sectype, n):
    """Uniformly random secret integer in range(n).

    Expected number of secret random bits needed is log_2 n + c,
    with c a small constant, c < 3.

    Special case: if sectype is a secure finite field and
    n is equal to the order of the finite field, then the
    uniformly random output can be generated directly.
    """
    await runtime.returnType((sectype, True))
    if sectype.__name__.startswith('SecFld') and n == sectype.field.order:
        return runtime._random(sectype)

    b = n - 1
    k = b.bit_length()
    x = runtime.random_bits(sectype, k)
    h = 1
    i = k - 1
    while i >= 0:
        if (b >> i) & 1:
            h -= h * (1 - x[i])
            i -= 1
        else:
            if await runtime.output(h * x[i]): # todo: mul_public
                # restart, keeping unused secret random bits x[:i]
                x = x[:i] + runtime.random_bits(sectype, k - i)
                i = k - 1
            else:
                i -= 1
    return runtime.from_bits(x)

@asyncoro.mpc_coro
async def random_unit_vector(sectype, n):
    """Uniformly random secret permutation of [1] + [0]*(n-1).

    Expected number of secret random bits needed is log_2 n + c,
    with c a small constant, c < 3.
    """
    await runtime.returnType((sectype, True), n)
    b = n - 1
    k = b.bit_length()
    x = runtime.random_bits(sectype, k)
    u = [sectype(1)]
    i = k - 1
    while i >= 0:
        if (b >> i) & 1:
            v = runtime.scalar_mul(x[i], u)
            u = v + runtime.vector_sub(u, v)
            i -= 1
        else:
            if await runtime.output(u[0] * x[i]): # todo: mul_public
                # restart, keeping unused secret random bits x[:i]
                x = x[:i] + runtime.random_bits(sectype, k - i)
                u = [sectype(1)]
                i = k - 1
            else:
                v = runtime.scalar_mul(x[i], u[1:])
                u = u[:1] + v + runtime.vector_sub(u[1:], v)
                i -= 1
    return u

def randrange(sectype, start, stop=None, step=1):
    """Uniformly random secret integer in range(start, stop[, step])."""
    if stop is None:
        stop = start
        start = 0
    n = len(range(start, stop, step))
    if not n:
        raise ValueError('empty range for randrange()')
    return start + _randbelow(sectype, n) * step

def randint(sectype, a, b):
    """Uniformly random secret integer between a and b, incl. both endpoints."""
    return randrange(sectype, a, b + 1)

def choice(sectype, seq):
    """Uniformly random secret element chosen from seq.

    Given seq may contain public and/or secret elements.

    If seq is empty, raises IndexError.
    """
    if not seq:
        raise IndexError('cannot choose from an empty sequence')
    u = random_unit_vector(sectype, len(seq))
    s = 0
    for i in range(len(seq)):
        s += u[i] * seq[i]
    return s

def choices(sectype, population, weights=None, *, cum_weights=None, k=1):
    """List of k uniformly random secret elements chosen from population.

    Choices are made with replacement.

    Given population may contain public and/or secret elements.

    If the relative weights or cumulative weights are not specified,
    the choices are made with equal probability.
    """
    if cum_weights is None:
        if weights is None:
            return [choice(sectype, population) for _ in range(k)]
        cum_weights = list(itertools.accumulate(weights))
    elif weights is not None:
        raise TypeError('cannot specify both weights and cumulative weights')
    if len(cum_weights) != len(population):
        raise ValueError('number of weights does not match the population')
    # assume weights are integers
    g = functools.reduce(math.gcd, cum_weights)
    cum_weights = [a // g for a in cum_weights]
    z = []
    for _ in range(k):
        r = _randbelow(sectype, cum_weights[-1])
        h = [r < a for a in cum_weights[:-1]]
        u = runtime.vector_sub(h + [1], [0] + h)
        s = 0
        for i in range(len(u)):
            s += u[i] * population[i]
        z.append(s)
    return z

def shuffle(sectype, x):
    """Shuffle list x secretly in place, and return None.

    Given list x may contain public or secret elements.
    """
    n = len(x)
    if not isinstance(x[0], sectype): # assume same type for all elts of x
        for i in range(len(x)):
            x[i] = sectype(x[i])
    for i in range(n - 1):
        u = random_unit_vector(sectype, n - i)
        x_u = runtime.in_prod(x[i:], u)
        d = runtime.scalar_mul(x[i] - x_u, u)
        x[i] = x_u
        x[i:] = runtime.vector_add(x[i:], d)

def random_permutation(sectype, x):
    """Uniformly random permutation of given sequence x or range(x)."""
    if isinstance(x, int):
        x = range(x)
    x = list(x)
    shuffle(sectype, x)
    return x

@asyncoro.mpc_coro
async def sample(sectype, population, k):
    """List of k uniformly random secret elements chosen from population.

    Choices are made without replacement.

    Given population may contain public and/or secret elements.

    If the population contains repeats, then each occurrence is a
    possible selection in the sample.

    To choose a sample in a range of integers, use range as an argument.
    This is especially fast and space efficient for sampling from a
    large population, e.g.: sample(sectype, range(10000000), 60).
    """
    await runtime.returnType(sectype, k)
    n = len(population)
    if not 0 <= k <= n:
        raise ValueError('sample larger than population or size is negative')
    elif not isinstance(population, range):
        x = list(population)
        if not isinstance(x[0], sectype): # assume same type for all elts of x
            for i in range(len(x)):
                x[i] = sectype(x[i])
        for i in range(k):
            u = random_unit_vector(sectype, n - i)
            x_u = runtime.in_prod(x[i:], u)
            d = runtime.scalar_mul(x[i] - x_u, u)
            x[i] = x_u
            x[i:] = runtime.vector_add(x[i:], d)
        return x[:k]
    else:
        x = []
        while len(x) < k:
            r = randrange(sectype, population.start, population.stop, population.step)
            if x:
                t = runtime.prod([r - a for a in x])
                if await runtime.is_zero_public(t):
                    continue
            x.append(r)
        return x

def random(sectype):
    """Uniformly random secret fixed-point number in the range [0.0, 1.0)."""
    f = sectype.field.frac_length
    if not f:
        raise TypeError('secure fixed-point type required')
    x = runtime.random_bits(sectype, f)
    return runtime.from_bits(x) * (2 ** -f)

def uniform(sectype, a, b):
    """Uniformly random secret fixed-point number N such that
    a <= N <= b for a <= b and b <= N <= a for b < a.
    """
    f = sectype.field.frac_length
    if not f:
        raise TypeError('secure fixed-point type required')
    s = math.copysign(1, b - a)
    return a + _randbelow(sectype, round(abs(a - b) * 2**f)) * s * (2**-f)

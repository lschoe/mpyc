"""This module provides secure versions of common mathematical statistics functions.
The module is modeled after the statistics module in the Python standard library, and
as such aimed at small scale use ("at the level of graphing and scientific calculators").

Functions mean, median, median_low, median_high, and mode are provided for calculating
averages (measures of central location). Functions variance, stdev, pvariance, pstdev
are provided for calculating variability (measures of spread).

Most of these functions work best with secure fixed-point numbers, but some effort is
done to support the use of secure integers as well. For instance, the mean of a sample
of integers is rounded to the nearest integer, which may still be useful. The variance
of a sample of integers is also rounded to the nearest integer, but this will only be
useful if the sample is properly scaled.

A baseline implementation is provided, favoring simplicity over efficiency. Also, the
current implementations of mode and median favor a small privacy leak over a strict but
less efficient approach.

If these functions are called with plain data, the call is relayed to the corresponding
function in Python's statistics module.
"""

import statistics
from mpyc import sectypes
from mpyc import asyncoro
from mpyc import random
from mpyc.mpctools import reduce

runtime = None


def mean(data):
    """Return the sample mean (average) of data which can be a sequence or an iterable.

    If the data points are secure integers or secure fixed-point numbers, the mean
    value returned is of the same secure type, rounded to the nearest number.

    If data is empty, StatisticsError will be raised.
    """
    if iter(data) is data:
        x = list(data)
    else:
        x = data
    n = len(x)
    if not n:
        raise statistics.StatisticsError('mean requires at least one data point')

    stype = type(x[0])  # all elts of x assumed of same type
    if issubclass(stype, sectypes.SecureFiniteField):
        raise TypeError('secure fixed-point or integer type required')

    if issubclass(stype, sectypes.SecureInteger):
        s = runtime.sum(x)
        return (s + n//2) // n  # round to nearest integer

    if issubclass(stype, sectypes.SecureFixedPoint):
        s = runtime.sum(x)
        e = n.bit_length()-1  # 1/2 < 2**e / n <= 1
        return s * (2**e / n) * 2**-e

    return statistics.mean(x)


def variance(data, xbar=None):
    """Return the sample variance of data, an iterable of at least two numbers.

    If the optional second argument xbar is given, it should be the mean of data.
    If it is missing or None (the default), the mean is automatically calculated.

    Use this function when your data is a sample from a population. To calculate
    the variance from the entire population, see pvariance().

    Raises StatisticsError if data has fewer than two values.
    """
    return _var(data, xbar, 1)


def stdev(data, xbar=None):
    """Return the sample standard deviation (square root of the sample variance).

    See variance() for arguments and other details.
    """
    return _std(data, xbar, 1)


def pvariance(data, mu=None):
    """Return the population variance of data, an iterable of at least two numbers.

    If the optional second argument mu is given, it is typically the mean of the data.
    It can also be used to compute the second moment around a point that is not the mean.
    If it is missing or None (the default), the arithmetic mean is automatically calculated.

    Use this function to calculate the variance from the entire population. To estimate
    the variance from a sample, the variance() function is usually a better choice.

    Raises StatisticsError if data is empty.
    """
    return _var(data, mu, 0)


def pstdev(data, mu=None):
    """Return the population standard deviation (square root of the population variance).

    See pvariance() for arguments and other details.
    """
    return _std(data, mu, 0)


def _var(data, m, correction):
    if iter(data) is data:
        x = list(data)
    else:
        x = data
    n = len(x)
    if n < 1 + correction:
        if correction:
            e = 'variance requires at least two data points'
        else:
            e = 'pvariance requires at least one data point'
        raise statistics.StatisticsError(e)

    stype = type(x[0])  # all elts of x assumed of same type
    if issubclass(stype, sectypes.SecureFiniteField):
        raise TypeError('secure fixed-point or integer type required')

    if issubclass(stype, sectypes.SecureInteger):
        if m is None:
            s = runtime.sum(x)
            y = [a * n - s for a in x]  # TODO: runtime.scalar_mul(n,x) for public (int) n
            d = n**2 * (n - correction)
        else:
            y = runtime.vector_sub(x, [m] * n)  # TODO: runtime.vector_sub(x,y) for scalar y
            d = n - correction
        return (runtime.in_prod(y, y) + d//2) // d

    if issubclass(stype, sectypes.SecureFixedPoint):
        if m is None:
            m = mean(x)
        y = runtime.vector_sub(x, [m] * n)
        d = n - correction
        return runtime.in_prod(y, y) / d

    if correction:
        return statistics.variance(x, m)

    return statistics.pvariance(x, m)


def _std(data, m, correction):
    if iter(data) is data:
        x = list(data)
    else:
        x = data
    n = len(x)
    if n < 1 + correction:
        if correction:
            e = 'stdev requires at least two data points'
        else:
            e = 'pstdev requires at least one data point'
        raise statistics.StatisticsError(e)

    stype = type(x[0])  # all elts of x assumed of same type
    if issubclass(stype, sectypes.SecureFiniteField):
        raise TypeError('secure fixed-point or integer type required')

    if issubclass(stype, sectypes.SecureInteger):
        return _isqrt(_var(x, m, correction))

    if issubclass(stype, sectypes.SecureFixedPoint):
        return _fsqrt(_var(x, m, correction))

    if correction:
        return statistics.stdev(x, m)

    return statistics.pstdev(x, m)


def _isqrt(a):
    """Return integer square root of nonnegative a.

    Simple secure version of bitwise algorithm for integer square roots,
    cf. function mpyc.gmpy.isqrt(). One comparison per bit of the output
    is quite costly though.
    """
    stype = type(a)
    e = (stype.bit_length - 1) // 2
    r, r2 = stype(0), stype(0)  # r2 = r**2
    j = 1 << e
    for _ in range(e+1):
        h, h2 = r + j, r2 + (2*r + j) * j
        r, r2 = runtime.if_else(h2 <= a, [h, h2], [r, r2])
        j >>= 1
    return r


def _fsqrt(a):
    """Return square root of nonnegative fixed-point number a.

    See function _isqrt(a).
    """
    stype = type(a)
    f = stype.frac_length
    e = (stype.bit_length + f-1) // 2  # (l+f)/2 - f = (l-f)/2 in [0..l/2]
    r = stype(0)
    j = 2**(e - f)
    for _ in range(e+1):
        h = r + j
        r = runtime.if_else(h * h <= a, h, r)
        j /= 2
    return r


def median(data):
    """Return the median of numeric data, using the common “mean of middle two” method.

    If data is empty, StatisticsError is raised. data can be a sequence or iterable.

    When the number of data points is even, the median is interpolated by taking the average of
    the two middle values.
    """
    return _med(data)


def median_low(data):
    """Return the low median of numeric data.

    If data is empty, StatisticsError is raised. data can be a sequence or iterable.

    The low median is always a member of the data set. When the number of data points is odd, the
    middle value is returned. When it is even, the smaller of the two middle values is returned.
    """
    return _med(data, med='low')


def median_high(data):
    """Return the high median of numeric data.

    If data is empty, StatisticsError is raised. data can be a sequence or iterable.

    The high median is always a member of the data set. When the number of data points is odd, the
    middle value is returned. When it is even, the larger of the two middle values is returned.
    """
    return _med(data, med='high')


def _med(data, med=None):
    if iter(data) is data:
        x = list(data)
    else:
        x = data[:]
    n = len(x)
    if not n:
        raise statistics.StatisticsError('median requires at least one data point')

    stype = type(x[0])  # all elts of x assumed of same type
    if issubclass(stype, sectypes.SecureFiniteField):
        raise TypeError('secure fixed-point or integer type required')

    if not issubclass(stype, sectypes.SecureObject):
        return statistics.median(x)

    if n%2:
        return _quickselect(x, (n-1)/2)

    if med == 'low':
        return _quickselect(x, (n-2)/2)

    if med == 'high':
        return _quickselect(x, n/2)

    # average two middle values
    s = _quickselect(x, (n-2)/2) + _quickselect(x, n/2)

    if issubclass(stype, sectypes.SecureInteger):
        return s//2

    return s/2


@asyncoro.mpc_coro
async def _quickselect(x, k):
    """Return kth order statistic, 0 <= k < n with n=len(x).

    If all elements of x are distinct, no information on x is leaked.
    If x contains duplicate elements, ties in comparisons are broken evenly, which
    ultimately leaks some information on the distribution of duplicate elements.

    Average running time (dominated by number of secure comparisons, and number of
    conversions of integer indices to unit vectors) is linear in n.
    """
    stype = type(x[0])
    await runtime.returnType(stype)
    n = len(x)
    if n == 1:
        return x[0]

    f = stype.frac_length
    y = runtime.random_bits(stype, n)
    p = runtime.in_prod(x, random.random_unit_vector(stype, n))  # random pivot
    z = [(x[i] - p)*2 < y[i] * 2**-f for i in range(n)]  # break ties x[i] == p uniformly at random
    s = int(await runtime.output(runtime.sum(z)))
    if k >= s:  # take complement
        k = k - s
        z = [1-a for a in z]
        s = n - s
    # 0 <= k < s
    w = [stype(0)] * s
    for i in range(n):
        j = runtime.sum(z[:i+1])  # 0 <= j <= s
        u = runtime.unit_vector(j, s)  # TODO: exploit that j <= i+1 to shorten u
        v = runtime.scalar_mul(z[i] * x[i], u)
        w = runtime.vector_add(w, v)
    return _quickselect(w, k)


def mode(data):
    """Return the mode, the most common data point from discrete or nominal data.

    If there are multiple modes with the same frequency, the first one encountered
    in data is returned.

    If data is empty, StatisticsError is raised.

    To speed up the computation, the bit length of the sample range max(data) - min(data)
    is revealed, provided this range is not too small.
    """
    if iter(data) is data:
        x = list(data)
    else:
        x = data[:]
    n = len(x)
    if not n:
        raise statistics.StatisticsError('mode requires at least one data point')

    if isinstance(x[0], sectypes.SecureObject):
        return _mode(x, PRIV=runtime.options.sec_param//6)

    return statistics.mode(x)  # NB: raises StatisticsError in Python < 3.8 if x is multimodal


@asyncoro.mpc_coro
async def _mode(x, PRIV=0):
    stype = type(x[0])  # all elts of x assumed of same type
    if issubclass(stype, sectypes.SecureFiniteField):
        raise TypeError('secure fixed-point or integer type required')

    if issubclass(stype, sectypes.SecureFixedPoint) and not x[0].integral:  # TODO: allow fractions
        raise ValueError('integral values required')

    await runtime.returnType(stype)

    f = stype.frac_length
    m, M = runtime.min_max(x)
    b = runtime.to_bits(M - m)
    e = len(b) - f
    while e > PRIV and not await runtime.output(b[e-1 + f]):
        e -= 1
    if not e:
        # m = M, x is constant
        return m

    # e <= PRIV or e = (M - m).bit_length()
    freqs = reduce(runtime.vector_add, (runtime.unit_vector(a - m, 2**e) for a in x))
    return m + runtime.argmax(freqs)[0]

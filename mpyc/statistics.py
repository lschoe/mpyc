"""This module provides secure versions of common mathematical statistics functions.
The module is modeled after the statistics module in the Python standard library, and
as such aimed at small scale use ("at the level of graphing and scientific calculators").

Functions mean, median, median_low, median_high, quantiles, and mode are provided
for calculating averages (measures of central location). Functions variance, stdev,
pvariance, pstdev are provided for calculating variability (measures of spread).
Functions covariance, correlation, linear_regression are provided for calculating
statistics regarding relations between two sets of data.

Most of these functions work best with secure fixed-point numbers, but some effort is
done to support the use of secure integers as well. For instance, the mean of a sample
of integers is rounded to the nearest integer, which may still be useful. The variance
of a sample of integers is also rounded to the nearest integer, but this will only be
useful if the sample is properly scaled.

A baseline implementation is provided, favoring simplicity over efficiency. Also, the
current implementations of mode, median, and quantiles favor a small privacy leak over
a strict but less efficient approach.

If these functions are called with plain data, the call is relayed to the corresponding
function in Python's statistics module.
"""

import sys
from math import fsum, sqrt
import statistics
from mpyc.sectypes import SecureObject, SecureInteger, SecureFixedPoint
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

    sectype = type(x[0])  # all elts of x assumed of same type
    if not issubclass(sectype, SecureObject):
        return statistics.mean(x)

    if issubclass(sectype, SecureFixedPoint):
        s = runtime.sum(x)
        e = n.bit_length()-1  # 1/2 < 2**e / n <= 1
        return s * (2**e / n) * 2**-e

    if issubclass(sectype, SecureInteger):
        s = runtime.sum(x)
        return (s + n//2) // n  # round to nearest integer

    raise TypeError('secure fixed-point or integer type required')


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

    sectype = type(x[0])  # all elts of x assumed of same type
    if not issubclass(sectype, SecureObject):
        if correction:
            return statistics.variance(x, m)

        return statistics.pvariance(x, m)

    if issubclass(sectype, SecureFixedPoint):
        if m is None:
            m = mean(x)
        y = runtime.vector_sub(x, [m] * n)
        d = n - correction
        return runtime.in_prod(y, y) / d

    if issubclass(sectype, SecureInteger):
        if m is None:
            s = runtime.sum(x)
            y = [a * n - s for a in x]  # TODO: runtime.scalar_mul(n,x) for public (int) n
            d = n**2 * (n - correction)
        else:
            y = runtime.vector_sub(x, [m] * n)  # TODO: runtime.vector_sub(x,y) for scalar y
            d = n - correction
        return (runtime.in_prod(y, y) + d//2) // d

    raise TypeError('secure fixed-point or integer type required')


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

    sectype = type(x[0])  # all elts of x assumed of same type
    if not issubclass(sectype, SecureObject):
        if correction:
            return statistics.stdev(x, m)

        return statistics.pstdev(x, m)

    if issubclass(sectype, SecureFixedPoint):
        return _fsqrt(_var(x, m, correction))

    if issubclass(sectype, SecureInteger):
        return _isqrt(_var(x, m, correction))

    raise TypeError('secure fixed-point or integer type required')


def _isqrt(a):
    """Return integer square root of nonnegative a.

    Simple secure version of bitwise algorithm for integer square roots,
    cf. function mpyc.gmpy.isqrt(). One comparison per bit of the output
    is quite costly though.
    """
    sectype = type(a)
    e = (sectype.bit_length - 1) // 2
    r, r2 = sectype(0), sectype(0)  # r2 = r**2
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
    sectype = type(a)
    f = sectype.frac_length
    e = (sectype.bit_length + f-1) // 2  # (l+f)/2 - f = (l-f)/2 in [0..l/2]
    r = sectype(0)
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

    sectype = type(x[0])  # all elts of x assumed of same type
    if not issubclass(sectype, SecureObject):
        return statistics.median(x)

    if not issubclass(sectype, (SecureFixedPoint, SecureInteger)):
        raise TypeError('secure fixed-point or integer type required')

    if n%2:
        return _quickselect(x, [(n-1)//2])[0]

    if med == 'low':
        return _quickselect(x, [(n-2)//2])[0]

    if med == 'high':
        return _quickselect(x, [n//2])[0]

    s = sum(_quickselect(x, [(n-2)//2, n//2]))  # average two middle values

    if issubclass(sectype, SecureFixedPoint):
        return s/2

    return s//2


@asyncoro.mpc_coro
async def _quickselect(x, ks):
    """Return kth order statistics for k in ks, where 0 <= k < n with n=len(x).

    If all elements of x are distinct, no information on x is leaked.
    If x contains duplicate elements, ties in comparisons are broken evenly, which
    ultimately leaks some information on the distribution of duplicate elements.

    Average running time (dominated by number of secure comparisons, and number of
    conversions of integer indices to unit vectors) is linear in n, for fixed ks.
    """
    # TODO: consider adding case ks is an int instead of a list
    # TODO: try to make implementation below competitive with straightforward "sort and pick"
    # approach; slowness due to expensive computation of w_left (and w_right). Also note
    # advantage of sorting that there is no privacy leakage.
    if len(ks) >= 3:
        y = runtime.sorted(x)
        return [y[k] for k in ks]

    if not ks:
        return []

    n = len(x)
    if n == 1:
        return [x[0]]

    sectype = type(x[0])
    await runtime.returnType(sectype, len(ks))

    f = sectype.frac_length
    while True:
        y = runtime.random_bits(sectype, n)
        p = runtime.in_prod(x, random.random_unit_vector(sectype, n))  # random pivot
        z = [2*(x[i] - p) < y[i] * 2**-f for i in range(n)]  # break ties x[i] == p evenly
        s = int(await runtime.output(runtime.sum(z)))
        if 0 < s < n:
            break

    ks_left = [k for k in ks if k < s]
    ks_right = [k - s for k in ks if k >= s]
    if not ks_left:
        ks_left = ks_right
        ks_right = []
        z = [1-a for a in z]
        s = n - s
    zx = runtime.schur_prod(z, x)
    sectype_0 = sectype(0)
    w_left = [sectype_0] * s
    if ks_right:
        w_right = [sectype_0] * (n - s)
    for i in range(n):
        j = runtime.sum(z[:i+1])  # 0 <= j <= i+1
        m = min(i+2, s)  # i+2 to avoid wrap around when i+1 < s still
        u_left = runtime.unit_vector(j, m)
        v_left = runtime.scalar_mul(zx[i], u_left)
        v_left.extend([sectype_0] * (s - m))
        w_left = runtime.vector_add(w_left, v_left)
        if ks_right:  # TODO: save some work by computing w_left and w_right together
            j = i+1 - j
            m = min(i+2, n - s)  # i+2 to avoid wrap around when i+1 < n - s still
            u_right = runtime.unit_vector(j, m)
            v_right = runtime.scalar_mul(x[i] - zx[i], u_right)
            v_right.extend([sectype_0] * (n - s - m))
            w_right = runtime.vector_add(w_right, v_right)
    w = _quickselect(w_left, ks_left)
    if ks_right:
        w.extend(_quickselect(w_right, ks_right))
    return w


def quantiles(data, *, n=4, method='exclusive'):
    """Divide data into n continuous intervals with equal probability.

    Returns a list of n-1 cut points separating the intervals.

    Set n to 4 for quartiles (the default). Set n to 10 for deciles.
    Set n to 100 for percentiles which gives the 99 cuts points that
    separate data into 100 equal sized groups.

    The data can be any iterable containing samples.
    The cut points are linearly interpolated between data points.

    If method is set to 'inclusive', data is treated as population data.
    The minimum value is treated as the 0th percentile (lowest quantile) and
    the maximum value is treated as the 100th percentile (highest quantile).
    """
    if n < 1:
        raise statistics.StatisticsError('n must be at least 1')

    if iter(data) is data:
        x = list(data)
    else:
        x = data
    ld = len(x)
    if ld < 2:
        raise statistics.StatisticsError('must have at least two data points')

    sectype = type(x[0])  # all elts of x assumed of same type
    if not issubclass(sectype, SecureObject):
        return statistics.quantiles(x, n=n, method=method)

    if issubclass(sectype, SecureFixedPoint):
        div_n = lambda a: a / n
    elif issubclass(sectype, SecureInteger):
        div_n = lambda a: (a + n//2) // n
    else:
        raise TypeError('secure fixed-point or integer type required')

    if method == 'inclusive':
        m = ld - 1
        # Determine which kth order statistics will actually be used.
        data = {}
        for i in range(1, n):
            j, delta = divmod(i * m, n)
            data[j] = None
            if delta:
                data[j+1] = None
        points = _quickselect(x, list(data))
        data = dict(zip(data, points))

        # Compute the n-1 cut points for the n quantiles.
        result = []
        for i in range(1, n):
            j, delta = divmod(i * m, n)
            interpolated = data[j]
            if delta:
                interpolated += div_n((data[j+1] - data[j]) * delta)
            result.append(interpolated)
        return result

    if method == 'exclusive':
        m = ld + 1
        # Determine which kth order statistics will actually be used.
        data = {}
        for i in range(1, n):
            j = i * m // n
            j = 1 if j < 1 else ld-1 if j > ld-1 else j  # clamp to 1 .. ld-1
            delta = i*m - j*n
            if n - delta:
                data[j-1] = None
            if delta:
                data[j] = None
        points = _quickselect(x, list(data))
        data = dict(zip(data, points))

        # Compute the n-1 cut points for the n quantiles.
        result = []
        for i in range(1, n):
            j = i * m // n
            j = 1 if j < 1 else ld-1 if j > ld-1 else j
            delta = i*m - j*n
            if delta == 0:
                interpolated = data[j-1]
            elif delta == n:
                interpolated = data[j]
            else:  # NB: possibly delta<0 or delta>n
                interpolated = data[j-1] + div_n((data[j] - data[j-1]) * delta)
            result.append(interpolated)
        return result

    raise ValueError(f'Unknown method: {method!r}')


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

    if isinstance(x[0], SecureObject):
        return _mode(x, PRIV=runtime.options.sec_param//6)

    return statistics.mode(x)


@asyncoro.mpc_coro
async def _mode(x, PRIV=0):
    sectype = type(x[0])  # all elts of x assumed of same type
    if not issubclass(sectype, (SecureFixedPoint, SecureInteger)):
        raise TypeError('secure fixed-point or integer type required')

    if issubclass(sectype, SecureFixedPoint) and not x[0].integral:  # TODO: allow fractions
        raise ValueError('integral values required')

    await runtime.returnType(sectype)

    f = sectype.frac_length
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


def covariance(x, y):
    """Return the sample covariance of x and y."""
    n = len(x)
    if len(y) != n:
        raise statistics.StatisticsError('covariance requires that both inputs '
                                         'have same number of data points')

    if n < 2:
        raise statistics.StatisticsError('covariance requires at least two data points')

    sectype = type(x[0])  # all elts of x assumed of same type
    if not issubclass(sectype, SecureObject):
        if sys.version_info.minor >= 10:
            return statistics.covariance(x, y)

        # inline code of statistics.covariance() copied from Python 3.10.0:
        xbar = fsum(x) / n
        ybar = fsum(y) / n
        sxy = fsum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
        return sxy / (n - 1)

    if issubclass(sectype, SecureFixedPoint):
        xbar = runtime.sum(x) / n
        ybar = runtime.sum(y) / n
        sxy = runtime.in_prod([xi - xbar for xi in x], [yi - ybar for yi in y])
        return sxy / (n - 1)

    if issubclass(sectype, SecureInteger):
        sx = runtime.sum(x)
        sy = runtime.sum(y)
        sxy = runtime.in_prod([xi * n - sx for xi in x], [yi * n - sy for yi in y])
        d = n**2 * (n - 1)
        return (sxy + d//2) // d

    raise TypeError('secure fixed-point or integer type required')


def correlation(x, y):
    """Return Pearson's correlation coefficient for x and y.

    Pearson's correlation coefficient takes values between -1 and +1.
    It measures the strength and direction of the linear relationship
    between x and y, where +1 means very strong, positive linear relationship,
    -1 very strong, negative linear relationship, and 0 no linear relationship.
    """
    n = len(x)
    if len(y) != n:
        raise statistics.StatisticsError('covariance requires that both inputs '
                                         'have same number of data points')

    if n < 2:
        raise statistics.StatisticsError('covariance requires at least two data points')

    sectype = type(x[0])  # all elts of x assumed of same type
    if not issubclass(sectype, SecureObject):
        if sys.version_info.minor >= 10:
            return statistics.correlation(x, y)

        # inline code of statistics.correlation() copied from Python 3.10.0:
        xbar = fsum(x) / n
        ybar = fsum(y) / n
        sxy = fsum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
        sxx = fsum((xi - xbar) ** 2.0 for xi in x)
        syy = fsum((yi - ybar) ** 2.0 for yi in y)
        try:
            return sxy / sqrt(sxx * syy)

        except ZeroDivisionError:
            raise statistics.StatisticsError('at least one of the inputs is constant') from None

    if issubclass(sectype, SecureFixedPoint):
        xbar = runtime.sum(x) / n
        ybar = runtime.sum(y) / n
        xxbar = [xi - xbar for xi in x]
        yybar = [yi - ybar for yi in y]
        sxy = runtime.in_prod(xxbar, yybar)
        sxx = runtime.in_prod(xxbar, xxbar)
        syy = runtime.in_prod(yybar, yybar)
        return sxy / (_fsqrt(sxx) * _fsqrt(syy))

    raise TypeError('secure fixed-point type required')


if sys.version_info.minor >= 10:
    LinearRegression = statistics.LinearRegression
else:
    from collections import namedtuple
    LinearRegression = namedtuple('LinearRegression', ('slope', 'intercept'))


def linear_regression(x, y):
    """Return a (simple) linear regression model for x and y.

    The parameters of the model are returned as a named LinearRegression tuple,
    with two fields called "slope" and "intercept", respectively.

    A linear regression model describes the relationship between independent
    variable x and dependent variable y in terms of a linear function:

        y = slope * x + intercept + noise

    Here, slope and intercept are the regression parameters estimated using
    ordinary least squares, and noise represents the variability of the data
    not explained by the linear regression (it is equal to the difference
    between predicted and actual values of the dependent variable).
    """
    n = len(x)
    if len(y) != n:
        raise statistics.StatisticsError('covariance requires that both inputs '
                                         'have same number of data points')

    if n < 2:
        raise statistics.StatisticsError('covariance requires at least two data points')

    sectype = type(x[0])  # all elts of x assumed of same type
    if not issubclass(sectype, SecureObject):
        if sys.version_info.minor >= 10:
            return statistics.linear_regression(x, y)

        # inline code of statistics.linear_regression() adapted from Python 3.10.0:
        xbar = fsum(x) / n
        ybar = fsum(y) / n
        sxy = fsum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
        sxx = fsum((xi - xbar) ** 2.0 for xi in x)
        try:
            slope = sxy / sxx   # equivalent to:  covariance(x, y) / variance(x)
        except ZeroDivisionError:
            raise statistics.StatisticsError('x is constant') from None

        intercept = ybar - slope * xbar
        return LinearRegression(slope=slope, intercept=intercept)

    if issubclass(sectype, SecureFixedPoint):
        xbar = runtime.sum(x) / n
        ybar = runtime.sum(y) / n
        xxbar = [xi - xbar for xi in x]
        yybar = [yi - ybar for yi in y]
        sxy = runtime.in_prod(xxbar, yybar)
        sxx = runtime.in_prod(xxbar, xxbar)
        slope = sxy / sxx
        intercept = ybar - slope * xbar
        return LinearRegression(slope=slope, intercept=intercept)

    # TODO: implement for secure integers as well
    raise TypeError('secure fixed-point type required')

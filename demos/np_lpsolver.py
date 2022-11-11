"""Demo Linear Programming (LP) solver, using secure integer arithmetic.

Vectorized.

See demo lpsolver.py for background information.

... work in progress for MPyC version 0.9
"""

import os
import logging
import argparse
import csv
import math
import numpy as np
from mpyc.runtime import mpc


# TODO: unify approaches for argmin etc. with secure NumPy arrays (also see lpsolverfxp.py)


def argmin_int(x):
    secintarr = type(x)
    n = len(x)
    if n == 1:
        return (secintarr(np.array([1])), x[0])

    if n == 2:
        b = x[0] < x[1]
        arg = mpc.np_fromlist([b, 1 - b])
        min = b * (x[0] - x[1]) + x[1]
        return arg, min

    a = x[:n//2]   ## split even odd?  start at n%2 as in reduce in mpctools
    b = x[(n+1)//2:]
    c = a < b
    m = c * (a - b) + b
    if n%2 == 1:
        m = np.concatenate((m, x[n//2:(n+1)//2]))
    ag, mn = argmin_int(m)
    if n%2 == 1:
        ag_1 = ag[-1:]
        ag = ag[:-1]
    arg1 = ag * c
    arg2 = ag - arg1
    if n%2 == 1:
        arg = np.concatenate((arg1, ag_1, arg2))
    else:
        arg = np.concatenate((arg1, arg2))
    return arg, mn


def argmin_rat(nd):
    secintarr = type(nd)
    N = nd.shape[1]
    if N == 1:
        return (secintarr(np.array([1])), (nd[0, 0], nd[1, 0]))

    if N == 2:
        b = mpc.in_prod([nd[0, 0], -nd[1, 0]], [nd[1, 1], nd[0, 1]]) < 0
        arg = mpc.np_fromlist([b, 1 - b])
        min = b * (nd[:, 0] - nd[:, 1]) + nd[:, 1]
        return arg, (min[0], min[1])

    a = nd[:, :N//2]
    b = nd[:, (N+1)//2:]
    c = a[0] * b[1] < b[0] * a[1]
    m = c * (a - b) + b
    if N%2 == 1:
        m = np.concatenate((m, nd[:, N//2:(N+1)//2]), axis=1)
    ag, mn = argmin_rat(m)
    if N%2 == 1:
        ag_1 = ag[-1:]
        ag = ag[:-1]
    arg1 = ag * c
    arg2 = ag - arg1
    if N%2 == 1:
        arg = np.concatenate((arg1, ag_1, arg2))
    else:
        arg = np.concatenate((arg1, arg2))
    return arg, mn


def pow_list(a, x, n):
    """Return [a,ax, ax^2, ..., ax^(n-1)].

    Runs in O(log n) rounds using minimal number of n-1 secure multiplications.
    """
    if n == 1:
        powers = mpc.np_fromlist([a])
    elif n == 2:
        powers = mpc.np_fromlist([a, a * x])
    else:
        even_powers = pow_list(a, x * x, (n+1)//2)
        if n%2:
            d = even_powers[-1]
            even_powers = even_powers[:-1]
        odd_powers = x * even_powers
        powers = np.stack((even_powers, odd_powers))
        powers = powers.T.reshape(n - (n%2))
        if n%2:
            powers = np.append(powers, mpc.np_fromlist([d]))
    return powers


# TODO: consider next version of unit_vector() as alternative for current mpc.unit_vector()
@mpc.coroutine
async def unit_vector(a, n):
    """Length-n unit vector [0]*a + [1] + [0]*(n-1-a) for secret a, assuming 0 <= a < n.

    NB: If a = n, unit vector [1] + [0]*(n-1) is returned. See mpyc.statistics.
    """
    await mpc.returnType(type(a), n)
    # TODO: add conversion from prime field GF(p) to secint, 0<=a<n, n < p/2 needed?
    u = await mpc.gather(mpc.random.random_unit_vector(type(a), n))
    r = sum(a * b for a, b in zip(u, range(n)))
    R = 1 + mpc._random(type(a).field, 1<<mpc.options.sec_param)
    c = await mpc.output(a - r + R * n)
    c %= n
    # rotate u over c positions to the right
    v = u[:n-c]
    u = u[n-c:]
    u.extend(v)
    return u


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset', type=int, metavar='I',
                        help=('dataset 0=uvlp (default), 1=wiki, 2=tb2x2, 3=woody, '
                              '4=LPExample_R20, 5=sc50b, 6=kb2, 7=LPExample'))
    parser.add_argument('-l', '--bit-length', type=int, metavar='L',
                        help='override preset bit length for dataset')
    parser.set_defaults(dataset=0, bit_length=0)
    args = parser.parse_args()

    settings = [('uvlp', 8, 1, 2),
                ('wiki', 6, 1, 2),
                ('tb2x2', 6, 1, 2),
                ('woody', 8, 1, 3),
                ('LPExample_R20', 70, 1, 5),
                ('sc50b', 104, 10, 55),
                ('kb2', 536, 100000, 106),
                ('LPExample', 110, 1, 178)]
    name, bit_length, scale, n_iter = settings[args.dataset]
    if args.bit_length:
        bit_length = args.bit_length

    T = np.genfromtxt(os.path.join('data', 'lp', name + '.csv'), dtype=float, delimiter=',')
    m, n = T.shape[0] - 1, T.shape[1] - 1
    secint = mpc.SecInt(bit_length, n=m + n)  # force existence of Nth root of unity, N>=m+n
    print(f'Using secure {bit_length}-bit integers: {secint.__name__}')
    print(f'dataset: {name} with {m} constraints and {n} variables (scale factor {scale})')
    T[0, -1] = 0.0  # initialize optimal value
    T = np.vectorize(int, otypes='O')(T * scale)
    g = np.gcd.reduce(T[1:], axis=1, keepdims=True)
    T[1:] //= np.maximum(g, 1)  # remove common factors per row (skipping cost row)
    T = secint.array(T)
    c, A, b = -T[0, :-1], T[1:, :-1], T[1:, -1]  # maximize c.x subject to A.x <= b, x >= 0

    Zp = secint.field
    N = Zp.nth
    w = Zp.root  # w is an Nth root of unity in Zp, where N >= m + n
    w_powers = [Zp(1)]
    for _ in range(N-1):
        w_powers.append(w_powers[-1] * w)
    assert w_powers[-1] * w == 1

    await mpc.start()

    cobasis = Zp.array(np.array([w_powers[-j].value for j in range(n)]))
    basis = Zp.array(np.array([w_powers[-(i + n)].value for i in range(m)]))
    previous_pivot = secint(1)

    iteration = 0
    while True:
        # find index of pivot column
        p_col_index, minimum = argmin_int(T[0, :-1])
        if await mpc.output(minimum >= 0):
            break  # maximum reached

        # find index of pivot row
        p_col = T[:, :-1] @ p_col_index
        den = p_col[1:]
        num = T[1:, -1] + (den <= 0)
        p_row_index, (_, pivot) = argmin_rat(np.stack((num, den)))

        # reveal progress a bit
        iteration += 1
        mx, cd, p = await mpc.output([T[0, -1], previous_pivot, pivot])
        logging.info(f'Iteration {iteration}/{n_iter}: {mx / cd} pivot={p / cd}')

        # swap basis entries
        delta = basis @ p_row_index - cobasis @ p_col_index
        cobasis += delta * p_col_index
        basis -= delta * p_row_index

        # update tableau Tij = Tij*Tkl/Tkl' - (Til/Tkl' - bool(i==k)) * (Tkj + bool(j==l)*Tkl')
        p_col_index = np.concatenate((p_col_index, np.array([0])))
        p_row_index = np.concatenate((np.array([0]), p_row_index))

        pp_inv = 1 / previous_pivot
        p_col = p_col * pp_inv - p_row_index
        p_row = p_row_index @ T + previous_pivot * p_col_index
        T = T * (pivot * pp_inv) - np.outer(p_col, p_row)
        previous_pivot = pivot

    mx = await mpc.output(T[0, -1])
    cd = await mpc.output(previous_pivot)  # common denominator for all entries of T
    print(f'max = {mx} / {cd} / {scale} = {mx / cd / scale} in {iteration} iterations')

    logging.info('Solution x')
    coefs = Zp.array([[w_powers[(j * k) % N].value for k in range(N)] for j in range(n)])   ## TODO: vandermonde ff array
    sum_powers = np.sum(np.fromiter((pow_list(T[i+1][-1] / N, basis[i], N) for i in range(m)), 'O'))
    x = coefs @ sum_powers
    Ax_bounded_by_b = mpc.all((A @ x <= b * cd).tolist())      ## TODO: np.all
    x_nonnegative = mpc.all((x >= 0).tolist())

    logging.info('Dual solution y')
    coefs = Zp.array([[w_powers[((n + i) * k) % N].value for k in range(N)] for i in range(m)])
    sum_powers = np.sum(np.fromiter((pow_list(T[0][j] / N, cobasis[j], N) for j in range(n)), 'O'))
    y = coefs @ sum_powers
    yA_bounded_by_c = mpc.all((y @ A >= c * cd).tolist())
    y_nonnegative = mpc.all((y >= 0).tolist())

    cx_eq_yb = c @ x == y @ b
    check = mpc.all([cx_eq_yb, Ax_bounded_by_b, x_nonnegative, yA_bounded_by_c, y_nonnegative])
    check = bool(await mpc.output(check))
    print(f'verification c.x == y.b, A.x <= b, x >= 0, y.A >= c, y >= 0: {check}')

    x = await mpc.output(x)
    print(f'solution = {[a / cd for a in x]}')

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

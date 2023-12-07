"""Demo LP solver, using secure integer arithmetic, vectorized.

This demo is a fully equivalent reimplementation of the lpsolver.py demo,
using secure integer arrays for NumPy-based vectorized computation.

Performance improvement of over 2x speedup when run with three parties
on local host. Memory consumption is also reduced.

See demo lpsolver.py for more information.
"""

import os
import logging
import argparse
import numpy as np
from mpyc.runtime import mpc


class SecureFraction:

    size = 2  # __lt__() assumes last dimension of size 2

    def __init__(self, a):
        self.a = a  # numerator, denominator

    def __lt__(self, other):  # NB: __lt__() is basic comparison as in Python's list.sort()
        return self.a[..., 0] * other.a[..., 1] < self.a[..., 1] * other.a[..., 0]


def np_pwlst(a, x, n):
    """Return array with a,ax,ax^2,...,ax^(n-1).

    Runs roughly in 2log_2(n)-1 rounds using n-1 + log_2(n) secure multiplications.
    """
    if n == 1:
        powers = mpc.np_fromlist([a])
    elif n == 2:
        powers = mpc.np_fromlist([a, a * x])
    else:
        even_powers = np_pwlst(a, x * x, (n+1)//2)
        if n%2:
            even_powers, d = even_powers[:-1], even_powers[-1:]
        powers = np.stack((even_powers, x * even_powers))  # add odd powers
        powers = powers.reshape(n - (n%2), order='F')  # mix even-odd powers
        if n%2:
            powers = np.append(powers, d)
    return powers


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
                ('wiki', 6, 1, 1),
                ('tb2x2', 6, 1, 2),
                ('woody', 8, 1, 3),
                ('LPExample_R20', 70, 1, 9),
                ('sc50b', 104, 10, 55),
                ('kb2', 560, 100000, 154),
                ('LPExample', 110, 1, 175)]
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
    w_powers = Zp.array(np.empty(N), check=False)
    w_powers[0] = 1
    for i in range(1, N):
        w_powers[i] = w_powers[i - 1] * w
    assert w_powers[-1] * w == 1

    await mpc.start()

    cobasis = w_powers[:n]
    basis = w_powers[N - m:N]
    previous_pivot = secint(1)

    iteration = 0
    while await mpc.output((arg_min := T[0, :-1].argmin())[1] < 0):
        # find index of pivot column
        p_col_index = arg_min[0]

        # find index of pivot row
        p_col = T[:, :-1] @ p_col_index
        denominator = p_col[1:]
        constraints = np.column_stack((T[1:, -1] + (denominator <= 0), denominator))
        p_row_index, (_, pivot) = constraints.argmin(key=SecureFraction)

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

    logging.info('Solution x')  # TODO: support np.vander() for finite field Vandermonde arrays
    coefs = w_powers[[[(-j * k) % N for k in range(N)] for j in range(n)]]
    sum_powers = np.sum(np.fromiter((np_pwlst(T[i+1][-1] / N, basis[i], N) for i in range(m)),
                                    'O', count=m))
    x = coefs @ sum_powers
    Ax_bounded_by_b = np.all(A @ x <= b * cd)
    x_nonnegative = np.all(x >= 0)

    logging.info('Dual solution y')
    coefs = w_powers[[[(-i * k) % N for k in range(N)] for i in range(N - m, N)]]
    sum_powers = np.sum(np.fromiter((np_pwlst(T[0][j] / N, cobasis[j], N) for j in range(n)),
                                    'O', count=n))
    y = coefs @ sum_powers
    yA_bounded_by_c = np.all(y @ A >= c * cd)
    y_nonnegative = np.all(y >= 0)

    cx_eq_yb = c @ x == y @ b
    check = mpc.all([cx_eq_yb, Ax_bounded_by_b, x_nonnegative, yA_bounded_by_c, y_nonnegative])
    check = bool(await mpc.output(check))
    print(f'verification c.x == y.b, A.x <= b, x >= 0, y.A >= c, y >= 0: {check}')

    x = await mpc.output(x)
    print(f'solution = {[a / cd for a in x]}')

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

"""Demo LP solver, using secure fixed-point arithmetic, vectorized.

This demo is a fully equivalent reimplementation of the lpsolverfxp.py demo,
using secure fixed-point arrays for NumPy-based vectorized computation.

Performance improvement of over 2x speedup when run with three parties
on local host. Memory consumption is also reduced.

See demo lpsolverfxp.py for more information.
"""

import os
import logging
import argparse
import numpy as np
from mpyc.runtime import mpc


class SecureFraction:

    size = 3  # __lt__() assumes last dimension of size 3

    def __init__(self, a):
        self.a = a  # numerator, denominator, pos

    def __lt__(self, other):  # NB: __lt__() is basic comparison as in Python's list.sort()
        b = self.a[..., 0] * other.a[..., 1] < other.a[..., 0] * self.a[..., 1]
        c0 = self.a[..., 2]
        c0.integral = True
        b *= c0               # b = b if c0 else 0
        c1 = other.a[..., 2]
        c1.integral = True
        b = c1 * (b - 1) + 1  # b = b if c1 else 1
        return b


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset', type=int, metavar='I',
                        help=('dataset 0=uvlp (default), 1=wiki, 2=tb2x2, 3=woody, '
                              '4=LPExample_R20, 5=sc50b, 6=kb2, 7=LPExample'))
    parser.add_argument('-l', '--bit-length', type=int, metavar='L',
                        help='override preset bit length for dataset')
    parser.set_defaults(dataset=0, bit_length=0)
    args = parser.parse_args()

    settings = [('uvlp', 24, 37/3),
                ('wiki', 24, 20),
                ('tb2x2', 18, 10.5),
                ('woody', 36, 540),
                ('LPExample_R20', 52, 3.441176),
                ('sc50b', 52, 70),
                ('kb2', 96, 1749.9204734889486),
                ('LPExample', 96, 1188806595)]
    name, bit_length, exact_max = settings[args.dataset]
    if args.bit_length:
        bit_length = args.bit_length

    T = np.genfromtxt(os.path.join('data', 'lp', name + '.csv'), dtype=float, delimiter=',')
    m, n = T.shape[0] - 1, T.shape[1] - 1
    secfxp = mpc.SecFxp(bit_length)
    print(f'Using secure {bit_length}-bit fixed-point numbers: {secfxp.__name__}')
    print(f'dataset: {name} with {m} constraints and {n} variables')
    T[0, -1] = 0.0  # initialize optimal value
    T = secfxp.array(T, integral=False)
    c, A, b = -T[0, :-1], T[1:, :-1], T[1:, -1]  # maximize c.x subject to A.x <= b, x >= 0

    await mpc.start()

    cobasis = secfxp.array(np.arange(n))
    basis = secfxp.array(np.arange(n, n + m))

    iteration = 0
    while await mpc.output((arg_min := T[0, :-1].argmin())[1] < 0):
        # find index of pivot column
        p_col_index = arg_min[0]

        # find index of pivot row
        p_col = T[:, :-1] @ p_col_index
        denominator = p_col[1:]
        constraints = np.column_stack((T[1:, -1], denominator, denominator > 0.0001))
        p_row_index, (_, pivot, _) = constraints.argmin(key=SecureFraction)

        # reveal progress a bit
        iteration += 1
        mx = await mpc.output(T[0, -1])
        p = await mpc.output(pivot)
        logging.info(f'Iteration {iteration}: {mx} pivot={p}')

        # swap basis entries
        delta = basis @ p_row_index - cobasis @ p_col_index
        cobasis += delta * p_col_index
        basis -= delta * p_row_index

        # update tableau Tij = Tij - (Til - bool(i==k))/Tkl *outer (Tkj + bool(j==l))
        p_col_index = np.concatenate((p_col_index, np.array([0])))
        p_row_index = np.concatenate((np.array([0]), p_row_index))
        p_col = (p_col - p_row_index) / pivot
        p_row = p_row_index @ T + p_col_index
        T -= np.outer(p_col, p_row)

    mx = await mpc.output(T[0, -1])
    rel_error = (mx - exact_max) / exact_max
    print(f'max = {mx} (error {rel_error:.3%}) in {iteration} iterations')

    logging.info('Solution x')
    x = np.sum(np.fromiter((T[i+1, -1] * mpc.np_unit_vector(basis[i], n + m)[:n] for i in range(m)),
                           'O', count=m))
    Ax_bounded_by_b = np.all(A @ x <= 1.01 * b + 0.0001)
    x_nonnegative = np.all(x >= 0)

    logging.info('Dual solution y')
    y = np.sum(np.fromiter((T[0, j] * mpc.np_unit_vector(cobasis[j], n + m)[n:] for j in range(n)),
                           'O', count=n))
    yA_bounded_by_c = np.all(y @ A >= np.where(c > 0, 1/1.01, 1.01) * c - 0.0001)
    y_nonnegative = np.all(y >= 0)

    cx_eq_yb = abs((cx := c @ x) - y @ b) <= 0.01 * abs(cx)
    check = mpc.all([cx_eq_yb, Ax_bounded_by_b, x_nonnegative, yA_bounded_by_c, y_nonnegative])
    check = bool(await mpc.output(check))
    print(f'verification c.x == y.b, A.x <= b, x >= 0, y.A >= c, y >= 0: {check}')

    x = await mpc.output(x)
    print(f'solution = {x}')

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

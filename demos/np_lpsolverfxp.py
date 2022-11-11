"""Demo Linear Programming (LP) solver, using secure fixed-point arithmetic.

Vectorized.

See demo lpsolverfxp.py for background information.

... work in progress for MPyC version 0.9
"""

import os
import logging
import argparse
import csv
import numpy as np
from mpyc.runtime import mpc


# TODO: unify approaches for argmin etc. with secure NumPy arrays (also see lpsolver.py)

# def argmin_int(xs):
    # a, m = mpc.argmin(xs)
    # u = mpc.unit_vector(a, len(xs))
    # u = mpc.np_fromlist(u)
    # u.integral = True
    # return u, m


def argmin_int(x):
    secarr = type(x)
    n = len(x)
    if n == 1:
        return (secarr(np.array([1])), x[0])

    if n == 2:
        b = x[0] < x[1]
        arg = mpc.np_fromlist([b, 1 - b])
        min = b * (x[0] - x[1]) + x[1]
        arg.integral = True
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
    arg.integral = True
    return arg, mn


def argmin_rat(nd, p):
    secarr = type(nd)
    n = nd.shape[1]
    if n == 1:
        return secarr(np.array([1])), (nd[0, 0], nd[1, 0])

    if n == 2:
        b = mpc.in_prod([nd[0, 0], -nd[1, 0]], [nd[1, 1], nd[0, 1]]) < 0
        c0 = p[0]
#        c0.integral = True
        b = mpc.if_else(c0, b, 0)
        c1 = p[1]
#        c1.integral = True
        b = mpc.if_else(c1, b, 1)
#        b.integral = True
        assert b.integral
        arg = mpc.np_fromlist([b, 1 - b])
        min = b * (nd[:, 0] - nd[:, 1]) + nd[:, 1]
        return arg, min

    a = nd[:, :n//2]
    b = nd[:, (n+1)//2:]
    aa = np.stack((-a[1], a[0]))
    aa.integral = False
    b.integral = False
    aa = aa.T.reshape(n//2, 1, 2)
    c = aa @ b.T.reshape(n//2, 2, 1) < 0  # c = a[0] * b[1] < b[0] * a[1]
    c = c.reshape(len(c))
    assert c.integral
    assert p.integral
#    c = a[2] * c
    a2 = p[:n//2]
    c *= a2
    assert c.integral
#    c = b[2] * (c - 1) + 1
    b2 = p[(n+1)//2:]
    c = b2 * (c - 1) + 1
    assert c.integral
    m = c * (a - b) + b
    mp = c * (a2 - b2) + b2
    assert mp.integral
    if n%2 == 1:
        m = np.concatenate((m, nd[:, n//2:(n+1)//2]), axis=1)
        mp = np.concatenate((mp, p[n//2:(n+1)//2]))
        mp.integral = True
    ag, mn = argmin_rat(m, mp)
    if n%2 == 1:
        ag_1 = ag[-1:]
        ag = ag[:-1]
    arg1 = ag * c
    arg2 = ag - arg1
    if n%2 == 1:
        arg = np.concatenate((arg1, ag_1, arg2))
    else:
        arg = np.concatenate((arg1, arg2))
    arg.integral = True
    return arg, mn


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
    basis = secfxp.array(n + np.arange(m))

    iteration = 0
    while True:
        # find index of pivot column
        p_col_index, minimum = argmin_int(T[0, :-1])
        if await mpc.output(minimum >= 0):
            break  # maximum reached

        # find index of pivot row
        assert p_col_index.integral
        p_col =  T[:, :-1] @ p_col_index
        p_col1 = p_col[1:]
        pos = p_col1 > 0.0001
        p_row_index, (_, pivot) = argmin_rat(np.stack((T[1:, -1], p_col1)), pos)

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
        p_col_index.integral = True
        p_row_index.integral = True
        p_col = (p_col - p_row_index) / pivot
        p_row = p_row_index @ T + p_col_index
        T -= np.outer(p_col, p_row)

    mx = await mpc.output(T[0,  -1])
    rel_error = (mx - exact_max) / exact_max
    print(f'max = {mx} (error {rel_error:.3%}) in {iteration} iterations')

    logging.info('Solution x')
    x = np.sum(np.fromiter((T[i+1, -1] * mpc.np_fromlist(mpc.unit_vector(basis[i], m + n)[:n]) for i in range(m)), 'O'))
    cx = c @ x
    Ax = A @ x
    Ax_bounded_by_b = mpc.all((Ax <= 1.01 * b + 0.0001).tolist())
    x_nonnegative = mpc.all((x >= 0).tolist())

    logging.info('Dual solution y')
    y = np.sum(np.fromiter((T[0, j] * mpc.np_fromlist(mpc.unit_vector(cobasis[j], m + n)[n:]) for j in range(n)), 'O'))
    yb = y @ b
    yA = y @ A
    yA_bounded_by_c = mpc.all((yA >= np.where(c > 0, 1/1.01, 1.01) * c - 0.0001).tolist())
    y_nonnegative = mpc.all((y >= 0).tolist())

    cx_eq_yb = abs(cx - yb) <= 0.01 * abs(cx)
    check = mpc.all([cx_eq_yb, Ax_bounded_by_b, x_nonnegative, yA_bounded_by_c, y_nonnegative])
    check = bool(await mpc.output(check))
    print(f'verification c.x == y.b, A.x <= b, x >= 0, y.A >= c, y >= 0: {check}')

    x = await mpc.output(x)
    print(f'solution = {x}')

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

"""Demo Linear Programming (LP) solver, using secure fixed-point arithmetic.

See the demo lpsolver.py for background and explanation.

Unlike the lpsolver.py demo which is based on exact (hence deterministic) secure
integer arithmetic, the results for this demo will vary for each run due to the
use of stochastic rounding in secure fixed-point arithmetic. In particular, the
number of iterations for the Simplex algorithm may vary a bit.

The maximum value found for the objective function is compared to the known exact
value. Since no exact solutions are found for the vectors x and y either, small
deviations are allowed in the verification of the results. The error tolerance
levels have been set an ad hoc fashion (1% relative deviation and 0.0001 absolute
deviation).
"""

import os
import logging
import argparse
import csv
from mpyc.runtime import mpc


def argmin_int(xs):
    a, m = mpc.argmin(xs)
    return mpc.unit_vector(a, len(xs)), m


def argmin_rat(xs):
    a, m = mpc.argmin(xs, key=SecureFraction)
    return mpc.unit_vector(a, len(xs)), m


class SecureFraction:
    def __init__(self, a):  # a = (numerator, denominator, denominator > +0)
        self.n, self.d, self.pos = a
        self.pos.integral = True  # TODO: fix integral attr for mpc.if_else()

    def __lt__(self, other):  # NB: __lt__() is basic comparison as in Python's list.sort()
        c = mpc.in_prod([self.n, -self.d], [other.d, other.n]) < 0
        c = mpc.if_else(self.pos, c, 0)
        c = mpc.if_else(other.pos, c, 1)
        return c


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

    with open(os.path.join('data', 'lp', name + '.csv')) as file:
        T = list(csv.reader(file))
    m = len(T) - 1
    n = len(T[0]) - 1
    secfxp = mpc.SecFxp(bit_length)
    print(f'Using secure {bit_length}-bit fixed-point numbers: {secfxp.__name__}')
    print(f'dataset: {name} with {m} constraints and {n} variables')
    T[0][-1] = '0'  # initialize optimal value
    for i in range(m+1):
        for j in range(n+1):
            T[i][j] = secfxp(float(T[i][j]), integral=False)

    c = T[0][:-1]  # maximize c.x subject to A.x <= b, x >= 0
    A = [T[i+1][:-1] for i in range(m)]
    b = [T[i+1][-1] for i in range(m)]

    await mpc.start()

    cobasis = [secfxp(j) for j in range(n)]
    basis = [secfxp(n + i) for i in range(m)]

    iteration = 0
    while True:
        # find index of pivot column
        p_col_index, minimum = argmin_int(T[0][:-1])

        if await mpc.output(minimum >= 0):
            break  # maximum reached

        # find index of pivot row
        p_col = mpc.matrix_prod([p_col_index], T, True)[0]
        constraints = [[T[i][-1], p_col[i], p_col[i] > 0.0001] for i in range(1, m+1)]
        p_row_index, (_, pivot, _) = argmin_rat(constraints)

        # reveal progress a bit
        iteration += 1
        mx = await mpc.output(T[0][-1])
        p = await mpc.output(pivot)
        logging.info(f'Iteration {iteration}: {mx} pivot={p}')

        # swap basis entries
        delta = mpc.in_prod(basis, p_row_index) - mpc.in_prod(cobasis, p_col_index)
        cobasis = mpc.vector_add(cobasis, mpc.scalar_mul(delta, p_col_index))
        basis = mpc.vector_sub(basis, mpc.scalar_mul(delta, p_row_index))

        # update tableau Tij = Tij - (Til - bool(i==k))/Tkl * (Tkj + bool(j==l))
        p_col_index.append(secfxp(0))
        p_row_index.insert(0, secfxp(0))
        p_col = mpc.vector_sub(p_col, p_row_index)
        p_col = mpc.scalar_mul(1 / pivot, p_col)
        p_row = mpc.matrix_prod([p_row_index], T)[0]
        p_row = mpc.vector_add(p_row, p_col_index)
        T = mpc.gauss(T, secfxp(1), p_col, p_row)

    mx = await mpc.output(T[0][-1])
    rel_error = (mx - exact_max) / exact_max
    print(f'max = {mx} (error {rel_error:.3%}) in {iteration} iterations')

    logging.info('Solution x')
    x = [secfxp(0) for _ in range(n)]
    for i in range(m):
        u = mpc.unit_vector(basis[i], m + n)[:n]
        v = mpc.scalar_mul(T[i+1][-1], u)
        x = mpc.vector_add(x, v)
    cx = mpc.in_prod(c, x)
    Ax = mpc.matrix_prod([x], A, True)[0]
    approx = lambda a: 1.01 * a + 0.0001
    Ax_bounded_by_b = mpc.all(Ax[i] <= approx(b[i]) for i in range(m))
    x_nonnegative = mpc.all(x[j] >= 0 for j in range(n))

    logging.info('Dual solution y')
    y = [secfxp(0) for _ in range(m)]
    for j in range(n):
        u = mpc.unit_vector(cobasis[j], m + n)[n:]
        v = mpc.scalar_mul(T[0][j], u)
        y = mpc.vector_sub(y, v)
    yb = mpc.in_prod(y, b)
    yA = mpc.matrix_prod([y], A)[0]
    approx = lambda a: mpc.if_else(a < 0, 1/1.01, 1.01) * a + 0.0001
    yA_bounded_by_c = mpc.all(yA[j] <= approx(c[j]) for j in range(n))
    y_nonpositive = mpc.all(y[i] <= 0 for i in range(m))

    cx_eq_yb = abs(cx - yb) <= 0.01 * abs(cx)
    check = mpc.all([cx_eq_yb, Ax_bounded_by_b, x_nonnegative, yA_bounded_by_c, y_nonpositive])
    check = bool(await mpc.output(check))
    print(f'verification c.x == y.b, A.x <= b, x >= 0, y.A <= c, y <= 0: {check}')

    x = await mpc.output(x)
    print(f'solution = {x}')

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

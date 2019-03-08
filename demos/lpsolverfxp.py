"""Demo Linear Programming (LP) solver, using fixed-point arithmetic.

See lpsolver.py.
"""

import os
import logging
import argparse
from distutils.version import StrictVersion
from mpyc.runtime import mpc

assert StrictVersion(mpc.version) >= StrictVersion('0.3.1')


def load_tableau(filename):
    T = []
    comment_sign = '#'
    sep = ','
    for line in open(os.path.join('data', 'lp', filename + '.csv'), 'r'):
        # strip comments and whitespace and skip empty lines
        line = line.split(comment_sign)[0].strip()
        if line:
            T.append(list(map(int, line.split(sep))))
    T[-1].append(0)
    return T


def argmin(x, arg_le):
    n = len(x)
    if n == 1:
        return ([type(x[0])(1)], x[0])
    if n == 2:
        b, m = arg_le(x[0], x[1])
        return ([1 - b, b], m)
    b2 = [None] * (n//2)
    m2 = [None] * ((n+1)//2)
    for i in range(n//2):
        b2[i], m2[i] = arg_le(x[2*i], x[2*i+1])
    if n % 2 == 1:
        m2[-1] = x[-1]
    a2, m = argmin(m2, arg_le)
    a = [None] * n
    if n % 2 == 1:
        a[-1] = a2.pop()
    for i in range(n//2):
        a[2*i+1] = b2[i] * a2[i]
        a[2*i] = a2[i] - a[2*i+1]
    return a, m


def argmin_int(xs):
    def arg_le_int(x0, x1):
        a = x0 >= x1
        m = mpc.if_else(a, x1, x0)
        return a, m
    return argmin(xs, arg_le_int)


def argmin_rat(xs):
    def arg_le_rat(x0, x1):
        n0, d0 = x0
        n1, d1 = x1
        a = mpc.in_prod([n0, d0], [d1, -n1]) >= 0
        m = mpc.if_else(a, [n1, d1], [n0, d0])
        return a, m
    return argmin(xs, arg_le_rat)


@mpc.coroutine
async def index_matrix_prod(x, A, tr=False):
    """Secure index-matrix product of unit vector x with (transposed) A."""
    x, A = x[:], [r[:] for r in A]
    stype = type(x[0])
    m = len(A) if tr else len(A[0])
    n = len(A) if not tr else len(A[0])
    await mpc.returnType(stype, m)
    x, A = await mpc.gather(x, A)
    # avoid fxp truncation for integral x
    f1 = 1 / stype.field(1<<stype.field.frac_length)
    x = [a.value * f1 for a in x]
    y = [None] * m
    for i in range(m):
        s = 0
        for j in range(n):
            s += x[j].value * (A[i][j] if tr else A[j][i]).value
        y[i] = stype.field(s)
    y = await mpc.gather(mpc._reshare(y))
    return y


def unit_vector(a, n):
    """Return a-th unit vector of length n, assuming 0 <= a < n."""
    stype = type(a)

    def si1(a, n):
        """Return (a-1)-st unit vector of length n-1 (if 1 <= a < n)
        or all-0 vector of length n-1 (if a=0).
        """
        if n == 1:
            x = []
        elif n == 2:
            x = [a]
        else:
            b = mpc.lsb(a / 2**stype.field.frac_length)
            z = si1((a - b) / 2, (n + 1) // 2)
            y = mpc.scalar_mul(b, z)
            x = [b - sum(y)] + [z[i//2] - y[i//2] if i % 2 == 0 else y[i//2] for i in range(n-2)]
        return x
    x = si1(a, n)
    return [stype(1) - sum(x)] + x


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='filename for tableau')
    parser.add_argument('options', nargs='*')
    parser.set_defaults(data='default')
    args = parser.parse_args()

    if not args.options:
        certificate_filename = f'c{mpc.pid}.cert'
        logging.info('Setting certificate file to default = %s', certificate_filename)
    else:
        certificate_filename = args.options[0]
    T = load_tableau(args.data)
    m = len(T) - 1
    n = len(T[0]) - 1
    l = mpc.options.bit_length
    secfxp = mpc.SecFxp(l)
    for i in range(m + 1):
        for j in range(n + 1):
            T[i][j] = secfxp(T[i][j])

    basis = [secfxp(i + n) for i in range(m)]
    cobasis = [secfxp(j) for j in range(n)]

    await mpc.start()

    iteration = 0
    logging.info('%d Termination?...', iteration)
    p_col_index, minimum = argmin_int(T[-1][:-1])

    while await mpc.output(minimum < 0):
        iteration += 1

        logging.info('%d Determining pivot...', iteration)
        p_col = index_matrix_prod(p_col_index + [secfxp(0)], T, True)
        constraints = [(T[i][-1] + (p_col[i] <= 0), p_col[i]) for i in range(m)]
        p_row_index, _ = argmin_rat(constraints)
        pivot = mpc.in_prod(p_row_index, p_col)

        logging.info('%d Updating tableau...', iteration)
        h = mpc.scalar_mul(1/pivot, [(p_row_index[i] if i < m else 0) - p_col[i] for i in range(m + 1)])
        p_row = index_matrix_prod(p_row_index, T[:-1])
        v = mpc.vector_add(p_row, p_col_index + [0])
        for i in range(m + 1):
            T[i] = mpc.vector_add(T[i], mpc.scalar_mul(h[i], v))

        # swap basis entries
        delta = mpc.in_prod(basis, p_row_index) - mpc.in_prod(cobasis, p_col_index)
        p_row_index = mpc.scalar_mul(delta, p_row_index)
        basis = mpc.vector_sub(basis, p_row_index)
        p_col_index = mpc.scalar_mul(delta, p_col_index)
        cobasis = mpc.vector_add(cobasis, p_col_index)

        logging.info('%d Termination?...', iteration)
        p_col_index, minimum = argmin_int(T[-1][:-1])

    logging.info('Termination...')
    mx = await mpc.output(T[-1][-1])
    print(' max(f) =', mx)

    logging.info('Computing solution...')
    solution = [secfxp(0) for _ in range(n)]
    for i in range(m):
        x = unit_vector(basis[i], m + n)[:n]
        y = mpc.scalar_mul(T[i][-1], x)
        solution = mpc.vector_add(solution, y)
    solution = await mpc.output(solution)

    logging.info('Computing dual solution...')
    dual_solution = [secfxp(0) for _ in range(m)]
    for j in range(n):
        x = unit_vector(cobasis[j], m + n)[n:]
        y = mpc.scalar_mul(T[-1][j], x)
        dual_solution = mpc.vector_add(dual_solution, y)
    dual_solution = await mpc.output(dual_solution)

    await mpc.shutdown()

    logging.info('Writing output to %s.', certificate_filename)
    with open(os.path.join('data', 'lp', certificate_filename), 'w') as f:
        f.write('# tableau = \n' + args.data + '\n')
        f.write('# bit-length = \n' + str(mpc.options.bit_length) + '\n')
        f.write('# security parameter = \n' + str(mpc.options.sec_param) + '\n')
        f.write('# threshold = \n' + str(mpc.threshold) + '\n')
        f.write('# solution = \n')
        f.write('\t'.join(x.__repr__() for x in solution) + '\n')
        f.write('# dual solution = \n')
        f.write('\t'.join(x.__repr__() for x in dual_solution) + '\n')

if __name__ == '__main__':
    mpc.run(main())

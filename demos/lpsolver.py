"""Demo Linear Programming (LP) solver.

The LP solver returns a solution to the following problem.

Given m x n matrix A, length-m vector b >=0, and length-n vector c.
Find a length-n vector x minimizing c.x subject to A x <= b and x >= 0.

The small (or, condensed) tableau variant of the Simplex algorithm is used.
The entries of A, b, and c are all assumed to be integral. Since b >= 0,
the all-zero vector x=0 is a feasible solution. It is also assumed that
a solution exists (hence that the problem is not unbounded).

The solution x is in general not integral. As a certificate of optimality,
a solution y to the dual problem is computed as well, that is, y is a
length-m vector maximizing b.y subject to y A >= c and y >= 0. The solutions
are represented by integer vectors, and one additional number, which is the
common denominator of all entries of the solution vectors.
"""

import os
import logging
import argparse
from mpyc.runtime import mpc

def load_tableau(filename):
    T = []
    comment_sign = '#'
    sep = ','
    with open(os.path.join('data', 'lp', filename + '.csv'), 'r') as f:
        for line in f:
            # strip comments and whitespace and skip empty lines
            line = line.split(comment_sign)[0].strip()
            if line:
                T.append(list(map(int, line.split(sep))))
    T[-1].append(0)
    return T

def pow_list(a, x, n):
    if n == 1:
        return [a]
    even = pow_list(a, x**2, (n+1)//2)
    if n%2 == 1:
        d = even.pop()
    odd = mpc.scalar_mul(x, even)
    xs = [None] * n
    for i in range(n//2):
        xs[2*i] = even[i]
        xs[2*i+1] = odd[i]
    if n%2 == 1:
        xs[-1] = d
    return xs

def argmin(x, arg_le):
    n = len(x)
    if n == 1:
        return ([1], x[0])
    if n == 2:
        b, m = arg_le(x[0], x[1])
        return ([1 - b, b], m)
    b2 = [None] * (n//2)
    m2 = [None] * ((n+1)//2)
    for i in range(n//2):
        b2[i], m2[i] = arg_le(x[2*i], x[2*i+1])
    if n%2 == 1:
        m2[-1] = x[-1]
    a2, m = argmin(m2, arg_le)
    a = [None] * n
    if n%2 == 1:
        a[-1] = a2.pop()
    b2 = mpc.schur_prod(b2, a2)
    for i in range(n//2):
        a[2*i] = a2[i] - b2[i]
        a[2*i+1] = b2[i]
    return a, m

def argmin_int(xs):
    def arg_le_int(x0, x1):
        a = x0 >= x1
        m = x0 + a * (x1 - x0)
        return a, m
    return argmin(xs, arg_le_int)

def argmin_rat(xs):
    def arg_le_rat(x0, x1):
        (n0, d0) = x0
        (n1, d1) = x1
        a = mpc.in_prod([n0, d0], [d1, -n1]) >= 0
        h = mpc.scalar_mul(a, [n1 - n0, d1 - d0])
        m = (h[0] + n0, h[1] + d0)
        return a, m
    return argmin(xs, arg_le_rat)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Filename for tableau.')
    parser.add_argument('options', nargs='*')
    parser.set_defaults(data='default')
    args = parser.parse_args(mpc.args)

    if not args.options:
        certificate_filename = 'c' + str(mpc.id) + '.cert'
        logging.info('Setting certificate file to default = %s', certificate_filename)
    else:
        certificate_filename = args.options[0]
    T = load_tableau(args.data)
    l = mpc.options.bit_length
    m = len(T) - 1
    n = len(T[0]) - 1
    secint = mpc.SecInt(l, m + n)
    for i in range(len(T)):
        for j in range(len(T[0])):
            T[i][j] = secint(T[i][j])

    Zp = secint.field
    p = Zp.modulus
    N = Zp.nth
    w = Zp.root
    w_powers = [Zp(1)]
    for _ in range(N - 1):
        w_powers.append(w_powers[-1] * w)
    assert w_powers[-1] * w == 1

    basis = [secint(w_powers[-(i+n)]) for i in range(m)]
    cobasis = [secint(w_powers[-j]) for j in range(n)]
    prev_pivot = secint(1)

    mpc.start()

    iteration = 0
    logging.info('%d Termination?...', iteration)
    p_col_index, minimum = argmin_int(T[-1][:-1])
    while mpc.run(mpc.output(minimum < 0)):
        iteration += 1

        logging.info('%d Determining pivot...', iteration)
        p_col = mpc.matrix_prod([p_col_index], T, True)[0]
        constraints = [(T[i][-1] + (p_col[i] <= 0), p_col[i]) for i in range(m)]
        p_row_index, (_, pivot) = argmin_rat(constraints)

        logging.info('%d Updating tableau...', iteration)
        #  T[i,j] = T[i,j]*p/p' - (C[i]/p' - p_row_index[i])*(R[j] + p * p_col_index[j])
        p_row = mpc.matrix_prod([p_row_index], T)[0]
        delta_row = mpc.scalar_mul(prev_pivot, p_col_index)
        delta_row.append(secint(0))
        p_row = mpc.vector_add(p_row, delta_row)
        prev_p_inv = 1 / prev_pivot
        p_col = mpc.scalar_mul(prev_p_inv, p_col)
        p_col = mpc.vector_sub(p_col, p_row_index + [secint(0)])
        T = mpc.gauss(T, pivot * prev_p_inv, p_col, p_row)
        prev_pivot = pivot
        # swap basis entries
        delta = mpc.in_prod(basis, p_row_index) - mpc.in_prod(cobasis, p_col_index)
        p_row_index = mpc.scalar_mul(delta, p_row_index)
        basis = mpc.vector_sub(basis, p_row_index)
        p_col_index = mpc.scalar_mul(delta, p_col_index)
        cobasis = mpc.vector_add(cobasis, p_col_index)

        logging.info('%d Termination?...', iteration)
        p_col_index, minimum = argmin_int(T[-1][:-1])

        mpc.run(mpc.barrier())

    logging.info('Termination...')
    mx = mpc.run(mpc.output(T[-1][-1]))
    cd = mpc.run(mpc.output(prev_pivot))
    print(' max(f) = %d / %d = %f' % (mx.value, cd.value, float(mx.value)/cd.value))

    logging.info('Computing solution...')
    sum_x_powers = [secint(0) for _ in range(N)]
    for i in range(m):
        x_powers = pow_list(T[i][-1] / N, basis[i], N)
        sum_x_powers = mpc.vector_add(sum_x_powers, x_powers)
    solution = [None] * n
    for j in range(n):
        coefs = [w_powers[(j*k)%N] for k in range(N)]
        solution[j] = mpc.lin_comb(coefs, sum_x_powers)
    solution = mpc.run(mpc.output(solution))

    logging.info('Computing dual solution...')
    sum_x_powers = [secint(0) for _ in range(N)]
    for j in range(n):
        x_powers = pow_list(T[-1][j] / N, cobasis[j], N)
        sum_x_powers = mpc.vector_add(sum_x_powers, x_powers)
    dual_solution = [None] * m
    for i in range(m):
        coefs = [w_powers[((n+i)*k)%N] for k in range(N)]
        dual_solution[i] = mpc.lin_comb(coefs, sum_x_powers)
    dual_solution = mpc.run(mpc.output(dual_solution))

    mpc.shutdown()

    logging.info('Writing output to %s.', certificate_filename)
    with open(os.path.join('data', 'lp', certificate_filename), 'w') as f:
        f.write('# tableau = \n' + args.data + '\n')
        f.write('# modulus = \n' + str(p) + '\n')
        f.write('# bit-length = \n' + str(mpc.options.bit_length) + '\n')
        f.write('# security parameter = \n' + str(mpc.options.security_parameter) + '\n')
        f.write('# threshold = \n' + str(mpc.threshold) + '\n')
        f.write('# common denominator = \n' + str(cd.value) + '\n')
        f.write('# solution = \n')
        f.write('\t'.join(str(x.value) for x in solution) + '\n')
        f.write('# dual solution = \n')
        f.write('\t'.join(str(x.value) for x in dual_solution) + '\n')

if __name__ == '__main__':
    main()

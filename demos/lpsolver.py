"""Demo Linear Programming (LP) solver, using secure integer arithmetic.

The LP solver returns a solution to the following problem:

    Given m x n matrix A, length-m vector b >= 0, and length-n vector c.
    Find a length-n vector x maximizing c.x subject to A x <= b and x >= 0.

The entries of A, b, and c are all assumed to be integral (dataset is scaled
if necessary). Since b >= 0, the all-zero vector x=0 is a feasible solution.
It is also assumed that a solution exists (hence that the LP problem is bounded),
but note that a solution x is in general not integral.

As a certificate of optimality, a solution y to the dual problem is computed
as well, that is, y is a length-m vector minimizing b.y subject to y A >= c
and y <= 0. The solutions are represented by integer vectors, and one additional
integer, which is the common denominator of all (rational) entries of the
solution vectors x and y.

See https://en.wikipedia.org/wiki/Linear_programming for more background.

History of this demo goes back quite a while. When Tomas Toft joined TUE (and
CWI) as a postdoc in late 2007, he finalized the work for his paper "Solving
Linear Programs Using Multiparty Computation", which appeared at the 13th
International Conference on Financial Cryptography and Data Security (FC 2009),
LNCS 5628, pp. 90-107, Springer. The main idea of using secure integer arithmetic
only still forms the basis for the demo.

With Sebastiaan de Hoogh joining TUE as a PhD student in 2008 and continuing as
a postdoc from 2012, the work was extended in various directions. Most notably,
an alternative approach based on secure fixed-point arithmetic was developed,
for a large part in joint work with Octavian Catrina. See the paper "Secure
Multiparty Linear Programming Using Fixed-Point Arithmetic" by Catrina and de
Hoogh, which appeared at the 15th European Symposium on Research in Computer
Security (ESORICS 2010), LNCS 6345, pp. 134-150, Springer.

And with Meilof Veeningen joining as a postdoc at TUE in 2014, the results were
extended to achieve secure verifiable linear programming. See the paper
"Certificate Validation in Secure Computation and Its Use in Verifiable Linear
Programming" by Sebastiaan de Hoogh, Berry Schoenmakers and Meilof Veeningen,
which appeared in the proceedings of Progress in Cryptology (AFRICACRYPT 2016),
LNCS 9646, pp. 265-284, Springer. See also the book chapter based on this paper:
"Universally Verifiable Outsourcing and Application to Linear Programming" by
the same authors, Chapter 10 in "Applications of Secure Multiparty Computation",
editors Peeter Laud and Lina Kamm, Cryptology and Information Security Series 13,
IOS Press (2015).

In conjunction with these papers, the implementation included in this demo and
the accompanying demo lpsolverfxp.py (which uses secure fixed-point arithmetic)
have been developed. Starting with Tomas Toft's implementation in VIFF, later
versions were based on TUeVIFF, a local branch of VIFF developed at TUE. TUeVIFF
ran on Python 2 using Twisted for asynchronous evaluation like VIFF, but with
several notable extensions (e.g., basic fixed-point arithmetic by Sebastiaan de
Hoogh and introduction of inline callbacks by Meilof Veeningen). With all this
and much more incorporated in MPyC, efficient programs for secure LP can now be
rendered with relative ease.

The LP demos include the following m x n datasets (m constraints, n variables):

  0=uvlp (2 x 3): example from AFRICACRYPT 2016 paper on secure verifiable LP
  1=wiki (2 x 3): https://en.wikipedia.org/wiki/Simplex_algorithm#Example
  2=tb2x2 (2 x 2): tiny example
  3=woody (3 x 2): wooden tables & chairs manufacturer classic text book example
  4=LPExample_R20 (20 x 20): randomly generated example from SecureSCM project
  5=sc50b (70 x 48): netlib.org/lp/data test problem, obtained via Mathematica
  6=kb2 (68 x 41): netlib.org/lp/data test problem, obtained via Mathematica
  7=LPExample (202 x 288): large example from SecureSCM project (*)

(*) Slight variant of problem and solution given in Figures 2 and 5, respectively,
of "Centralised and decentralised supply chain planning" by Richard Pibernik and
Eric Sucky, International Journal of Integrated Supply Management (IJISM), Vol. 2,
Nos. 1/2, pp. 6-27, Inderscience, 2006.

The demos use the small (or, condensed) tableau variant of the simplex algorithm,
combined with Dantzig's pivoting rule. The pivot column is selected by picking
any column l, 0 <=l < n, for which the c-entry in tableau T is minimal, where
index l is represented as a unit vector (1 at position l and 0 elsewhere). Next,
the pivot row k, 0 <= k < m, is selected by determining the most restrictive
constraint for the pivot column: only rows with positive entries in the pivot
column pose restrictions, and any such row k can be picked for which the ratio
between the b-entry in tableau T and the candidate pivot in row k, column l of T
is minimal.

Some special techniques to improve the efficiency are included from the above
paper by Catrina and de Hoogh, in particular to obliviously select the pivot row
and update the tableau. A further interesting technique, briefly explained in the
abovementioned book chapter, is used to speed up the oblivious extraction of the
solution vectors x and y. To this end, each (co-)basis element is represented as
a power of an Nth root of unity w in the prime field underlying the MPyC secure
integers used. Here, N >= m+n, hence sufficiently large to fit all basis and
cobasis elements. Together, the basis and cobasis elements partition the set
{w^i: 0 <= i < m+n}. This FFT-like technique is more efficient than the obvious
alternative using conversions of (co-)basis elements to unit vectors, as shown
in the other demo based on secure fixed-point arithmetic.

Overall, computational savings are favored over reductions in round complexity,
opting for logarithmic round complexity where applicable.
"""

import os
import logging
import argparse
import csv
import math
from mpyc.runtime import mpc


def pow_list(a, x, n):
    """Return [a,ax, ax^2, ..., ax^(n-1)].

    Runs in O(log n) rounds using minimal number of n-1 secure multiplications.
    """
    if n == 1:
        powers = [a]
    elif n == 2:
        powers = [a, a * x]
    else:
        even_powers = pow_list(a, x * x, (n+1)//2)
        if n%2:
            d = even_powers.pop()
        odd_powers = mpc.scalar_mul(x, even_powers)
        powers = [t for _ in zip(even_powers, odd_powers) for t in _]
        if n%2:
            powers.append(d)
    return powers


def argmin_int(xs):
    a, m = mpc.argmin(xs)
    return mpc.unit_vector(a, len(xs)), m


def argmin_rat(xs):
    a, m = mpc.argmin(xs, key=SecureFraction)
    return mpc.unit_vector(a, len(xs)), m


class SecureFraction:
    def __init__(self, a):
        self.n, self.d = a  # numerator, denominator

    def __lt__(self, other):  # NB: __lt__() is basic comparison as in Python's list.sort()
        return mpc.in_prod([self.n, -self.d], [other.d, other.n]) < 0


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

    with open(os.path.join('data', 'lp', name + '.csv')) as file:
        T = list(csv.reader(file))
    m = len(T) - 1
    n = len(T[0]) - 1
    secint = mpc.SecInt(bit_length, n=m + n)  # force existence of Nth root of unity, N>=m+n
    print(f'Using secure {bit_length}-bit integers: {secint.__name__}')
    print(f'dataset: {name} with {m} constraints and {n} variables (scale factor {scale})')
    T[0][-1] = '0'  # initialize optimal value
    for i in range(m+1):
        g = 0
        for j in range(n+1):
            T[i][j] = int(scale * float(T[i][j]))  # scale to integer
            g = math.gcd(g, T[i][j])
        g = max(g, 1) if i else 1  # skip cost row
        for j in range(n+1):
            T[i][j] = secint(T[i][j] // g)

    c = T[0][:-1]  # maximize c.x subject to A.x <= b, x >= 0
    A = [T[i+1][:-1] for i in range(m)]
    b = [T[i+1][-1] for i in range(m)]

    Zp = secint.field
    N = Zp.nth
    w = Zp.root  # w is an Nth root of unity in Zp, where N >= m + n
    w_powers = [Zp(1)]
    for _ in range(N-1):
        w_powers.append(w_powers[-1] * w)
    assert w_powers[-1] * w == 1

    await mpc.start()

    cobasis = [secint(w_powers[-j]) for j in range(n)]
    basis = [secint(w_powers[-(i + n)]) for i in range(m)]
    previous_pivot = secint(1)

    iteration = 0
    while True:
        # find index of pivot column
        p_col_index, minimum = argmin_int(T[0][:-1])
        if await mpc.output(minimum >= 0):
            break  # maximum reached

        # find index of pivot row
        p_col = mpc.matrix_prod([p_col_index], T, True)[0]
        constraints = [[T[i][-1] + (p_col[i] <= 0), p_col[i]] for i in range(1, m+1)]
        p_row_index, (_, pivot) = argmin_rat(constraints)

        # reveal progress a bit
        iteration += 1
        mx = await mpc.output(T[0][-1])
        cd = await mpc.output(previous_pivot)
        p = await mpc.output(pivot)  # NB: no await in f-strings in Python 3.6
        logging.info(f'Iteration {iteration}/{n_iter}: {mx / cd} pivot={p / cd}')

        # swap basis entries
        delta = mpc.in_prod(basis, p_row_index) - mpc.in_prod(cobasis, p_col_index)
        cobasis = mpc.vector_add(cobasis, mpc.scalar_mul(delta, p_col_index))
        basis = mpc.vector_sub(basis, mpc.scalar_mul(delta, p_row_index))

        # update tableau Tij = Tij*Tkl/Tkl' - (Til/Tkl' - bool(i==k)) * (Tkj + bool(j==l)*Tkl')
        p_col_index.append(secint(0))
        p_row_index.insert(0, secint(0))
        pp_inv = 1 / previous_pivot
        p_col = mpc.scalar_mul(pp_inv, p_col)
        p_col = mpc.vector_sub(p_col, p_row_index)
        p_row = mpc.matrix_prod([p_row_index], T)[0]
        p_row = mpc.vector_add(p_row, mpc.scalar_mul(previous_pivot, p_col_index))
        T = mpc.gauss(T, pivot * pp_inv, p_col, p_row)
        previous_pivot = pivot

    mx = await mpc.output(T[0][-1])
    cd = await mpc.output(previous_pivot)  # common denominator for all entries of T
    print(f'max = {mx} / {cd} / {scale} = {mx / cd / scale} in {iteration} iterations')

    logging.info('Solution x')
    sum_x_powers = [secint(0) for _ in range(N)]
    for i in range(m):
        x_powers = pow_list(T[i+1][-1] / N, basis[i], N)
        sum_x_powers = mpc.vector_add(sum_x_powers, x_powers)
    x = [None] * n
    for j in range(n):
        coefs = [w_powers[(j * k) % N] for k in range(N)]
        x[j] = mpc.in_prod(coefs, sum_x_powers)
    cx = mpc.in_prod(c, x)
    Ax = mpc.matrix_prod([x], A, True)[0]
    Ax_bounded_by_b = mpc.all(Ax[i] <= b[i] * cd for i in range(m))
    x_nonnegative = mpc.all(x[j] >= 0 for j in range(n))

    logging.info('Dual solution y')
    sum_x_powers = [secint(0) for _ in range(N)]
    for j in range(n):
        x_powers = pow_list(T[0][j] / N, cobasis[j], N)
        sum_x_powers = mpc.vector_add(sum_x_powers, x_powers)
    y = [None] * m
    for i in range(m):
        coefs = [w_powers[((n + i) * k) % N] for k in range(N)]
        y[i] = mpc.in_prod(coefs, sum_x_powers)
        y[i] = -y[i]
    yb = mpc.in_prod(y, b)
    yA = mpc.matrix_prod([y], A)[0]
    yA_bounded_by_c = mpc.all(yA[j] <= c[j] * cd for j in range(n))
    y_nonpositive = mpc.all(y[i] <= 0 for i in range(m))

    cx_eq_yb = cx == yb
    check = mpc.all([cx_eq_yb, Ax_bounded_by_b, x_nonnegative, yA_bounded_by_c, y_nonpositive])
    check = bool(await mpc.output(check))
    print(f'verification c.x == y.b, A.x <= b, x >= 0, y.A <= c, y <= 0: {check}')

    x = await mpc.output(x)
    print(f'solution = {[a / cd for a in x]}')

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

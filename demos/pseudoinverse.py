"""Demo Moore-Penrose pseudoinverse.

This demo implements the protocol for the pseudoinverse from the paper "A Practical
Approach to the Secure Computation of the Moore-Penrose Pseudoinverse over the
Rationals" by Niek J. Bouman and Niels de Vreede, Applied Cryptography and Network
Security (ACNS) 2020, LNCS 12146, pp. 398-417, Springer.
See https://doi.org/10.1007/978-3-030-57808-4_20 (or, https://eprint.iacr.org/2019/470).

The Moore-Penrose pseudoinverse is available in NumPy as numpy.linalg.pinv(). We use
this function to check our results. The demo generates a random matrix A of the given
dimensions m x n and rank r (by default a 5x5 matrix of full rank 5). The entries of
matrix A are limited to 4-bit integers, by default.

The bit length for the secure integers is set using Springer's bound in terms of the
Frobenius norm of A and the rank r of A. See Lemma 5 of the paper above.
"""

import math
import argparse
import logging
import itertools
import asyncio
import numpy as np
from mpyc.runtime import mpc


@mpc.coroutine
async def random_square_matrix(secarray, m) -> asyncio.Future:
    """Return random m x m matrix of given type secarray."""
    field = secarray.sectype.field
    if mpc.pid == 0:
        # NB: limit random integers to 64 bits as dtype=object is not allowed:
        U = np.random.randint(min(field.order, 2**64), size=(m, m), dtype=np.uint64)
        U = field.array(U)  # TODO: generate random field elts/arrays over full range
    else:
        U = None
    U = await mpc.transfer(U, senders=0)
    return U


def reflexive_generalized_inverse(A):
    """Return reflexive generalized inverse of given m x m matrix A.

    Also return the determinant of A.
    """
    m = A.shape[0]
    if m == 1:
        d = A[0, 0]
        b = d == 0
        Z = np.block([[(1/(d + b) - b)]])
    else:
        t = m // 2
        E, F, FT, H = A[:t, :t], A[:t, t:], A[t:, :t], A[t:, t:]
        X, d1 = reflexive_generalized_inverse(E)
        FTX = FT @ X
        Y, d2 = reflexive_generalized_inverse(H - FTX @ F)
        XFY = FTX.T @ Y
        Z = np.block([[X + XFY @ FTX, -XFY], [-XFY.T, Y]])
        d = d1 * d2
    return Z, d


@mpc.coroutine
async def pseudo_inverse(A):
    """Return n x m pseudoinverse of given m x n matrix A. """
    m, n = A.shape
    if m > n:
        return pseudo_inverse(A.T).T

    await mpc.returnType((type(A), (n, m)))

    U = await random_square_matrix(type(A), m)  # U is a public preconditioner
    A_AT = A @ A.T
    X = U @ (A_AT @ A_AT) @ U.T
    X = reflexive_generalized_inverse(X)[0]
    X = U.T @ X @ U
    X = A.T @ (X @ A_AT)
    return X


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, metavar='M',
                        help='number of matrix rows M > 0 (default=N or R or 5)')
    parser.add_argument('-n', type=int, metavar='N',
                        help='number of matrix columns N > 0 (default=M or R or 5)')
    parser.add_argument('-r', type=int, metavar='R',
                        help='(max.) matrix rank R >= 0 (default=min(M,N) or 5')
    parser.add_argument('-b', type=int, metavar='B',
                        help='(max.) bit length B > 0 of matrix entries (default=4)')
    parser.set_defaults(b=4)
    args = parser.parse_args()

    m = args.m or args.n or args.r or 5
    n = args.n or m
    r = args.r if args.r is not None else min(m, n)

    if mpc.pid == 0:
        if r == 0:
            A = np.zeros((m, n), dtype=int)
        else:
            while True:
                beta = 1 + math.floor(math.sqrt((2**(args.b - 1) - 1) / r))
                A = np.random.randint(-beta, beta, (m, r)) @ np.random.randint(-beta, beta, (r, n))
                if np.linalg.matrix_rank(A) == r:
                    break
    else:
        A = None

    await mpc.start()

    A = await mpc.transfer(A, senders=0)
    print(f'Matrix A, {m}x{n} of rank {r}, entries up to bit length {args.b}:\n {A}')
    A1 = np.linalg.pinv(A)  # store pseudoinverse to check results

    # Set bit length using Springer's bound in terms of r and the Frobenius-norm of A.
    l = 0 if r == 0 else math.ceil(r * math.log(np.linalg.norm(A, 'fro')**2 / r, 2))
    l = 1 + l  # add one bit for signed integers
    secint = mpc.SecInt(l)
    print(f'Using secure integers: {secint.__name__}')
    A = secint.array(A)

    logging.info('Compute pseudoinverse X of A (numerator)')
    X = pseudo_inverse(A)
    X = await mpc.output(X, raw=True)

    logging.info('Set D = I + A(A^T - X)')
    D = np.eye(m, dtype='O') + A @ (A.T - X)
    # Compute det(D) in two ways, for the purpose of demonstration.
    logging.info('Compute determinant using reflexive_generalized_inverse()')
    d = reflexive_generalized_inverse(D)[1]
    d = await mpc.output(d, raw=True)
    logging.info('Compute determinant using FiniteFieldArray.gauss.det()')
    d_ = np.linalg.det(D)
    d_ = await mpc.output(d_, raw=True)
    assert d == d_
    print(f'Common denominator vol^2(A): {d}')

    logging.info('Check result.')
    A = await mpc.output(A, raw=True)
    Penrose_eqs = ((A @ X @ A, A), (X @ A @ X, X), ((A @ X).T, A @ X), ((X @ A).T, X @ A))
    check = all(itertools.starmap(np.array_equal, Penrose_eqs))
    print(f'Penrose equations AXA=A, XAX=X, (AX)^T=AX, (XA)^T=XA: {check}')
    X = secint.field.array.intarray(d * X) / int(d)  # d X is integer valued, with d = vol^2(A)
    print(f'Pseudoinverse X of A:\n {X}')
    assert np.allclose(A1, X.astype(float))

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

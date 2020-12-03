"""
In this mpc we have n players with points (y_i) on an unknown polynomial f: R -> R of degree POLYNOMIAL_DEGREE.
This function secretly computes the polynomial and check whether f(i) = y_i for all i.
The result is only a boolean: whether this polynomial exists or not.
"""
from typing import List, Dict

from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint

POLYNOMIAL_DEGREE = 2


def get_my_secret_value(my_index):
    secret_polynomial = lambda x: 3 * (x ** POLYNOMIAL_DEGREE) + 2 * x + 11
    return secret_polynomial(my_index)


async def main():
    await mpc.start()
    n = len(mpc.parties)
    assert n >= POLYNOMIAL_DEGREE + 2, "Not enough parties to validate"

    value = get_my_secret_value(mpc.pid)  # Each party know his value and not the polynomial.

    shareable_secret = mpc.SecFxp()(value)
    inputs = mpc.input(shareable_secret, senders=list(range(n)))  # gather inputs from all parties
    points_x_to_y: Dict[int, SecureFixedPoint] = {i: inputs[i] for i in range(n)}

    # We will use the polynomial interpolation method as described here:
    # https://en.wikipedia.org/wiki/Polynomial_interpolation#Constructing_the_interpolation_polynomial
    matrix_x_points = list(points_x_to_y)[:POLYNOMIAL_DEGREE+1]
    matrix: List[List[float]] = [[x ** col for col in range(POLYNOMIAL_DEGREE + 1)] for x in matrix_x_points]
    a_col: List[SecureFixedPoint] = [points_x_to_y[x] for x in matrix_x_points]
    for col in range(POLYNOMIAL_DEGREE + 1):
        # assume lower cols are 0, and eliminate all values in this column for rows != col
        row_factor = matrix[col][col] ** -1
        matrix[col] = [row_factor * c for c in matrix[col]]
        a_col[col] = mpc.mul(a_col[col], row_factor)
        for index, row in enumerate(matrix):
            if index != col:
                matrix[index] = [c1 + (-1 * row[col] * c2) for c1, c2 in zip(row, matrix[col])]
                a_col[index] += mpc.mul(a_col[col], -1 * row[col])

    # validate that all the other parties follow this polynomial
    total = mpc.SecFxp()(0)
    for x in set(points_x_to_y) - set(matrix_x_points):
        expected_y = sum([mpc.mul(val, (x ** index)) for index, val in enumerate(a_col)], mpc.SecFxp()(0))
        total += mpc.abs(expected_y - points_x_to_y[x])
    if await mpc.is_zero_public(total):
        print("Same polynomial")
    else:
        print("Different polynomial")

    await mpc.shutdown()


if __name__ == '__main__':
    mpc.run(main())

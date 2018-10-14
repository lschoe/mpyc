"""Demo solution to the Secret Santa problem.

The Secret Santa problem is about generating secret random permutations
without fixed points. A fixed point of a permuation p is a point i for
which p(i)=i, hence the point is mapped to itself. Permutations without
fixed points are also called 'derangements'.
"""

import sys
from mpyc.runtime import mpc

@mpc.coroutine
async def random_unit_vector(n, sectype):
    """Random permutation of [sectype(1)] + [sectype(0) for i in range(n-1)]."""
    await mpc.returnType((sectype, True), n)
    if n == 1:
        return [sectype(1)]
    b = mpc.random_bit(sectype)
    x = random_unit_vector((n + 1) // 2, sectype)
    if n % 2 == 0:
        y = mpc.scalar_mul(b, x)
        return y + mpc.vector_sub(x, y)
    elif await mpc.eq_public(b * x[0], 1):
        return random_unit_vector(n, sectype)
    else:
        y = mpc.scalar_mul(b, x[1:])
        return x[:1] + y + mpc.vector_sub(x[1:], y)

def random_permutation(n, sectype):
    """Random permutation of [sectype(i) for i in range(n)]. """
    p = [sectype(i) for i in range(n)]
    for i in range(n - 1):
        x_r = random_unit_vector(n - i, sectype)
        p_r = mpc.in_prod(p[i - n:], x_r)
        d_r = mpc.scalar_mul(p[i] - p_r, x_r)
        p[i] = p_r
        for j in range(n - i):
            p[i + j] += d_r[j]
    return p

@mpc.coroutine
async def random_derangement(n, sectype):
    """Random permutation of [sectype(i) for i in range(n)] without fixed point."""
    await mpc.returnType((sectype, True), n)
    p = random_permutation(n, sectype)
    t = mpc.prod([p[i] - i for i in range(n)])
    if await mpc.is_zero_public(t):
        p = random_derangement(n, sectype)
    return p

def main():
    if sys.argv[1:]:
        N = int(sys.argv[1])
    else:
        N = 8
        print('Setting input to default =', N)

    mpc.start()

    secint = mpc.SecInt()
    print('Using secure integers:', secint)
    for n in range(2, N + 1):
        print(n, mpc.run(mpc.output(random_derangement(n, secint))))

    secfxp = mpc.SecFxp()
    print('Using secure fixed-point numbers:', secfxp)
    for n in range(2, N + 1):
        print(n, mpc.run(mpc.output(random_derangement(n, secfxp))))

    secpfld = mpc.SecFld(l=max(len(mpc.parties), (N - 1)).bit_length())
    print('Using secure prime fields:', secpfld)
    for n in range(2, N + 1):
        print(n, mpc.run(mpc.output(random_derangement(n, secpfld))))

    secbfld = mpc.SecFld(char2=True, l=max(len(mpc.parties), (N - 1)).bit_length())
    print('Using secure binary fields:', secbfld)
    for n in range(2, N + 1):
        print(n, mpc.run(mpc.output(random_derangement(n, secbfld))))

    mpc.shutdown()

if __name__ == '__main__':
    main()

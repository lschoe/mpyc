from mpyc.runtime import mpc

@mpc.coroutine
async def random_unit_vector(n, sectype):
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
        return y + mpc.vector_sub(x[1:], y) + x[:1]

def random_permutation(n, sectype):
    a = [sectype(i) for i in range(n)]
    for i in range(n - 1):
        x = random_unit_vector(n - i, sectype)
        a_x = mpc.in_prod(a[i - n:], x)
        d = mpc.scalar_mul(a[i] - a_x, x)
        a[i] = a_x
        for j in range(n - i):
            a[i + j] += d[j]
    return a

@mpc.coroutine
async def random_derangement(n, sectype):
    await mpc.returnType((sectype, True), n)
    a = random_permutation(n, sectype)
    t = mpc.prod([a[i] - i for i in range(n)])
    if await mpc.is_zero_public(t):
        return random_derangement(n, sectype)
    else:
        return a

def main():
    if not mpc.args:
        m = 8
        print('Setting input to default =', m)
    else:
        m = int(mpc.args[0])

    mpc.start()

    secfld = mpc.SecFld(l=max(len(mpc.parties), (m - 1)).bit_length() + 1)
    print('Using secure fields:', secfld)
    for n in range(2, m + 1):
        print(n, mpc.run(mpc.output(random_derangement(n, secfld))))

    secint = mpc.SecInt()
    print('Using secure integers:', secint)
    for n in range(2, m + 1):
        print(n, mpc.run(mpc.output(random_derangement(n, secint))))

    secfxp = mpc.SecFxp()
    print('Using secure fixed-point numbers:', secfxp)
    for n in range(2, m + 1):
        print(n, mpc.run(mpc.output(random_derangement(n, secfxp))))

    mpc.shutdown()

if __name__ == '__main__':
    main()

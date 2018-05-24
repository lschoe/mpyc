from mpyc.runtime import mpc

def secret_index(x, n):
    """Return x-th unit vector of length n, assuming 0 <= x < n."""

    def si1(x, n):
        """Return all-0 vector of length n-1 (if x=0) and (x-1)-st unit vector of length n-1 (if 1 <= x < n)."""
        if n==1:
            return []
        elif n==2:
            return [x]
        else:
            b = mpc.lsb(x)
            v = si1((x - b) / 2, (n + 1) // 2)
            w = mpc.scalar_mul(b, v)
            return [b-sum(w)] + [v[i//2]-w[i//2] if i%2==0 else w[i//2] for i in range(n-2)]
    v = si1(x, n)
    return [secint(1) - sum(v)] + v

secint = mpc.SecInt()

if mpc.args:
    t = int(mpc.args[0])
else:
    t = 12
    print('Setting input to default =', t)

mpc.start()
for i in range(t):
    print(i, mpc.run(mpc.output(secret_index(secint(i), t))))
mpc.shutdown()

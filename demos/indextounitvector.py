import sys
from mpyc.runtime import mpc


def secret_index(a, n):
    """Return a-th unit vector of length n, assuming 0 <= a < n."""

    def si1(a, n):
        """Return (a-1)-st unit vector of length n-1 (if 1 <= a < n)
        or all-0 vector of length n-1 (if a=0).
        """
        if n == 1:
            x = []
        elif n == 2:
            x = [a]
        else:
            a2, b = divmod(a, 2)
            z = si1(a2, (n + 1) // 2)
            y = mpc.scalar_mul(b, z)
            x = [b-sum(y)] + [z[i//2]-y[i//2] if i % 2 == 0 else y[i//2] for i in range(n-2)]
        return x
    x = si1(a, n)
    return [secint(1) - sum(x)] + x


secint = mpc.SecInt()

if sys.argv[1:]:
    t = int(sys.argv[1])
else:
    t = 12
    print('Setting input to default =', t)

mpc.run(mpc.start())
for i in range(t):
    print(i, mpc.run(mpc.output(secret_index(secint(i), t))))
mpc.run(mpc.shutdown())

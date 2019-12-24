import sys
from mpyc.runtime import mpc


def bsort(x):

    def bitonic_sort(lo, n, up=True):
        if n > 1:
            m = n//2
            bitonic_sort(lo, m, not up)
            bitonic_sort(lo + m, n - m, up)
            bitonic_merge(lo, n, up)

    def bitonic_merge(lo, n, up):
        if n > 1:
            # set m as the greatest power of 2 less than n:
            m = 2**((n-1).bit_length() - 1)
            for i in range(lo, lo + n - m):
                bitonic_compare(i, i + m, up)
            bitonic_merge(lo, m, up)
            bitonic_merge(lo + m, n - m, up)

    def bitonic_compare(i, j, up):
        b = (x[i] > x[j]) ^ (not up)
        d = b * (x[j] - x[i])
        x[i], x[j] = x[i] + d, x[j] - d

    bitonic_sort(0, len(x))
    return x


async def main():
    if sys.argv[1:]:
        n = int(sys.argv[1])
    else:
        n = 5
        print('Setting input to default =', n)

    s = [(-1)**i * (i + n//2)**2 for i in range(n)]

    secnum = mpc.SecInt()
    print('Using secure integers:', secnum)
    x = list(map(secnum, s))
    async with mpc:
        print('Array:', await mpc.output(x))
        print('Sorted array:', await mpc.output(bsort(x)))

    secnum = mpc.SecFxp()
    print('Using secure fixed-point numbers:', secnum)
    x = list(map(secnum, s))
    async with mpc:
        print('Input array:', await mpc.output(x))
        print('Sorted array:', await mpc.output(bsort(x)))

if __name__ == '__main__':
    mpc.run(main())

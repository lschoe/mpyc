"""Demo parallel sorting, with no secrecy.

This demo shows how parallel computation can be seen as a special case of
secure multiparty computation. By setting the threshold for the number of
corrupt parties equal to 0, each share of a secret will effectively be a
copy of the secret. Hence, the MPyC protocols will not provide any secrecy.
However, the parties are still able to perform joint computations on their
combined inputs and achieve essential speedups by distributing the work.

We use comparison-based sorting as a simple example. To sort m numbers one
needs O(m log m) comparisons, hence a 1-party computation, limited to
sequential processing, needs O(m log m) time to complete the task. With an
m-party computation we can reduce this to O(m) time, letting the parties
P[0],...,P[m-1] work on the task in parallel as follows:

1. P[i] broadcasts its input value x[i] to all parties.
2. P[i] sets p[i] to the rank of its own x[i] w.r.t. all m values.
3. P[i] broadcasts p[i] to all other parties.
4. P[i] computes the length-m list with x[i] in position p[i].

The rank of x[i] is computed using O(m) comparisons by counting how many
values are below x[i]. Hence, each step clearly takes O(m) time. Assuming
unique input values x[i] for now, p will be a permutation on {0,...,m-1}.
So, each party will obtain the sorted list using O(m) time overall.

This idea can be generalized to let m parties sort n numbers in O(n) time
provided n is not too large compared to m, that is, if n = O(m 2^m). With
each party holding k input values, hence n = m*k numbers in total, the
protocol between parties P[0],...,P[m-1] runs as follows:

1. P[i] broadcasts its input values x[i][0..k] in sorted order to all parties.
2. P[i] sets p[i][u] to the rank of its own x[i][u] w.r.t. all n=m*k values.
3. P[i] broadcasts p[i][0..k] to all other parties.
4. P[i] computes the length-n list with x[i][u] in position p[i][u].

Sorting in step 1 takes O(k log k) = O(n/m log n/m) = O(2^m log 2^m) = O(n)
comparisons. The ranks in step 2 can be computed in O(n) comparisons as shown
below in function quickranks(). Broadcasting k values to m parties also takes
O(n) time, hence overall we need O(n) time.

Functions parsort1() and parsort2() below implement these two protocols.
Ties are broken such that p will always be a permutation, even when there
are duplicates among the input values. Two ways to handle the actual exchange
of numbers between the parties are demonstrated:

- Using mpc.output(mpc.input(a)). Since we set mpc.threshold=0, mpc.input(a)
will send a copy of a to all parties, and mpc.output will simply output this
copy. To use this approach, however, the type of a must be an MPyC secure number.

- Using mpc.transfer(a). This will send (a copy of) a to all parties as long as
the type of a can be pickled using Python's pickle module.

Note that the rank values are exchanged in the same way as the input values.
"""

import random
from mpyc.runtime import mpc

mpc.threshold = 0  # No secrecy.
m = len(mpc.parties)
k = 2**(m-1)  # any k=O(2^m) allowed, ensuring that n = m*k = O(m 2^m)

secint = mpc.SecInt()
secfxp = mpc.SecFxp()
secflt = mpc.SecFlt()

flatten = lambda s: [a for _ in s for a in _]


def rank(x, i):
    """Return rank of x[i] w.r.t. all values in x.

    Rank of x[i] is the number of elements below x[i], that is,
    elements smaller than x[i], or equal to x[i] but left from it in x.
    """
    a = x[i]
    return sum(int(b < a or (b == a and j < i)) for j, b in enumerate(x))


def quickranks(x, i):
    """Return ranks of all elements of x[i] w.r.t. all values in x.

    NB: input x is a length-m list of length-k lists x[j], 0<=j<m.
    """
    p = [0] * k
    for j in range(m):
        u = v = 0
        while u < k and v < k:
            a, b = x[i][u], x[j][v]
            if b < a or (b == a and j*k + v < i*k + u):
                p[u] += 1
                v += 1
            else:
                u += 1
    for u in range(k-1):
        p[u+1] += p[u]  # cumulative counts
    return p


def inverse(p):
    """Return inverse of permutation p."""
    q = [None] * len(p)
    for i, a in enumerate(p):
        q[a] = i
    return q


async def parsort1(a):
    """Sort m numbers in O(m) time, each party providing one number a."""
    if isinstance(a, (secint, secfxp, secflt)):
        x = await mpc.output(mpc.input(a))
    else:
        x = await mpc.transfer(a)
    print('Random inputs, one per party: ', x)

    p = rank(x, mpc.pid)  # x[mpc.pid] = a
    if isinstance(a, (secint, secfxp, secflt)):
        p = await mpc.output(mpc.input(type(a)(p)))
        p = [int(a) for a in p]
    else:
        p = await mpc.transfer(p)
    y = [x[a] for a in inverse(p)]
    assert y == sorted(x)
    print('Sorted outputs, one per party:', y)


async def parsort2(z):
    """Sort n=m*k numbers in O(n) time, each party providing length-k list of numbers z."""
    x = await mpc.transfer(sorted(z))
    p = quickranks(x, mpc.pid)  # x[mpc.pid] = sorted(z)
    x = flatten(x)
    print('Random inputs,', k, '(sorted) per party:', x)

    p = await mpc.transfer(p)
    p = flatten(p)
    y = [x[a] for a in inverse(p)]
    assert y == sorted(x)
    print('Sorted outputs,', k, 'per party:        ', y)


mpc.run(mpc.start())

print(f'====== Using MPyC integers {secint}')
mpc.run(parsort1(secint(random.randint(0, 99))))
print(' * * *')

print('====== Using Python integers')
mpc.run(parsort1(random.randint(0, 99)))
mpc.run(parsort2([random.randint(0, 999) for _ in range(k)]))
print(' * * *')

print(f'====== Using MPyC fixed-point numbers {secfxp}')
mpc.run(parsort1(secfxp(0.5 - random.randint(0, 99))))
print(' * * *')

print('====== Using Python floats')
mpc.run(parsort1(0.5 - random.randint(0, 99)))
mpc.run(parsort2([random.randint(0, 8)/8 + random.randint(0, 99) for _ in range(k)]))
print(' * * *')

print(f'====== Using MPyC floats {secflt}')
mpc.run(parsort1(secflt(random.uniform(0, 1)*10**30)))
print(' * * *')

mpc.run(mpc.shutdown())

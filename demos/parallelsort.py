"""Demo parallel sort, with no secrecy.

First, each party broadcasts a random number to all parties.
Then, the i-th party broadcasts the i-th smallest number to all parties.
This effectively sorts the random numbers.

Next, each party broadcasts a list of m random numbers to all parties,
where m is the number of parties in the protocol. 
The parties thus generate m**2 numbers in total. 
Then, the i-th party broadcasts the sorted segment containing the 
(m*i)-th smallest number up to (but not including) the (m*(i+1))-st
smallest number. This effectively sorts the m**2 random numbers.

The protocols provide no secrecy but show that parallel computation can
be seen as a special case of secure multiparty computation. Accordingly, 
we set the threshold for the number of corrupt parties simply to 0.
"""

import random
import itertools
from mpyc.runtime import mpc

def quickselect(x, k, l=None):
    """Stub for efficient selection of kth smallest element of x 
    up to (k+l-1)st smallest element of x, in arbitrary order.
    
    Efficient selection can be done using O(len(x)) comparisons.
    """
    if l:
        return sorted(x)[k:k+l]
    else:
        return sorted(x)[k]

mpc.threshold = 0 # No secret sharing.
m = len(mpc.parties)

mpc.start()

secint = mpc.SecInt()
print('Using secure integers:', secint)
x = mpc.run(mpc.output(mpc.input(secint(random.randint(0, 99)))))
print('Random inputs, one per party:', x)
x = [a.signed() for a in x]
a = quickselect(x, mpc.id)
x = mpc.run(mpc.output(mpc.input(secint(a))))
print('Sorted outputs, one per party:', x)

x = mpc.input([secint(random.randint(0, 999)) for _ in range(m)])
x = list(itertools.chain(*x))
x = mpc.run(mpc.output(x))
print('Random inputs,', m, 'per party:', x)
x = [a.signed() for a in x]
x = sorted(quickselect(x, m * mpc.id, m))
x = mpc.input([secint(a) for a in x])
x = list(itertools.chain(*x))
x = mpc.run(mpc.output(x))
print('Sorted outputs,', m, 'per party:', x)

secfxp = mpc.SecFxp()
print(f'Using secure fixed-point numbers: {secfxp}')
x = mpc.run(mpc.output(mpc.input(secfxp(0.5 - random.randint(0, 99)))))
print('Random inputs, one per party: ', x)
x = [a.signed() for a in x]
a = quickselect(x, mpc.id)
x = mpc.run(mpc.output(mpc.input(secfxp(a))))
print('Sorted outputs, one per party: ', x)

x = mpc.input([secfxp(random.randint(0,8)/8 + random.randint(0, 99)) for _ in range(m)])
x = list(itertools.chain(*x))
x = mpc.run(mpc.output(x))
print('Random inputs,', m, 'per party:', x)
x = [a.signed() for a in x]
x = sorted(quickselect(x, m * mpc.id, m))
x = mpc.input([secfxp(a) for a in x])
x = list(itertools.chain(*x))
x = mpc.run(mpc.output(x))
print('Sorted outputs,', m, 'per party:', x)

mpc.shutdown()

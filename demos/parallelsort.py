"""Demo parallel sort, with no secrecy.

First, each party broadcasts a random number to all parties.
Then, the i-th party broadcasts the i-th smallest number to all parties.
This effectively sorts the random numbers.

The protocol provides no secrecy but shows that a parallel computation can
be seen as a special case of a secure multiparty computation. Accordingly, 
we set the threshold for the number of corrupt parties simply to 0.
"""

import random
from mpyc.runtime import mpc

mpc.threshold = 0 # no secret sharing

mpc.start()

secint = mpc.SecInt()
print('Using secure integers:', secint)
x = mpc.run(mpc.output(mpc.input(secint(random.randint(0, 99)))))
print('Random inputs, one per party: ', x)
x = [a.signed() for a in x]
x.sort()
x = mpc.run(mpc.output(mpc.input(secint(x[mpc.id]))))
print('Sorted outputs, one per party: ', x)

secfxp = mpc.SecFxp()
print('Using secure fixed-point numbers:', secfxp)
x = mpc.run(mpc.output(mpc.input(secfxp(0.5 - random.randint(0, 99)))))
print('Random inputs, one per party: ', x)
x = [a.signed() for a in x]
x.sort()
x = mpc.run(mpc.output(mpc.input(secfxp(x[mpc.id]))))
print('Sorted outputs, one per party: ', x)

mpc.shutdown()

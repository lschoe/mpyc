"""Demo MPyC oneliners.

Run with m parties to compute:

 - m    =  sum_{i=0}^{m-1} 1    =  sum(1     for i in range(m))
 - m**2 =  sum_{i=0}^{m-1} 2i+1 =  sum(2*i+1 for i in range(m))
 - 2**m = prod_{i=0}^{m-1} 2    = prod(2     for i in range(m))
 - m!   = prod_{i=0}^{m-1} i+1  = prod(i+1   for i in range(m))

Bit lengths of secure integers ensure each result fits for any m >= 1.
"""

from mpyc.runtime import mpc

m = len(mpc.parties)
l = m.bit_length()

mpc.run(mpc.start())
print('m    =', mpc.run(mpc.output(mpc.sum(mpc.input(mpc.SecInt(l+1)(1))))))
print('m**2 =', mpc.run(mpc.output(mpc.sum(mpc.input(mpc.SecInt(2*l+1)(2*mpc.pid + 1))))))
print('2**m =', mpc.run(mpc.output(mpc.prod(mpc.input(mpc.SecInt(m+2)(2))))))
print('m!   =', mpc.run(mpc.output(mpc.prod(mpc.input(mpc.SecInt(int(m*(l-1.4)+3))(mpc.pid + 1))))))
mpc.run(mpc.shutdown())

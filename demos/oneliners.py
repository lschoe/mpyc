"""Couple of MPyC oneliners.

Run with m parties to compute:

 - m    =  sum_{i=0}^{m-1} 1    =  sum(1     for i in range(m))
 - m**2 =  sum_{i=0}^{m-1} 2i+1 =  sum(2*i+1 for i in range(m))
 - 2**m = prod_{i=0}^{m-1} 2    = prod(2     for i in range(m))
 - m!   = prod_{i=0}^{m-1} i+1  = prod(i+1   for i in range(m))

Bit lengths of secure integers ensure each result fits for any m, 1<=m<=256.
"""

from mpyc.runtime import mpc

mpc.run(mpc.start())
print('m    =', mpc.run(mpc.output(mpc.sum(mpc.input(mpc.SecInt(9)(1))))))
print('m**2 =', mpc.run(mpc.output(mpc.sum(mpc.input(mpc.SecInt(17)(2*mpc.pid+1))))))
print('2**m =', mpc.run(mpc.output(mpc.prod(mpc.input(mpc.SecInt(257)(2))))))
print('m!   =', mpc.run(mpc.output(mpc.prod(mpc.input(mpc.SecInt(1685)(mpc.pid+1))))))
mpc.run(mpc.shutdown())

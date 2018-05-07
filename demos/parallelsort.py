import random
from mpyc.runtime import mpc

mpc.start()

secint = mpc.SecInt()

m = mpc.run(mpc.output(mpc.input(secint(random.randint(0, 99)))))
print('Random inputs, one per party: ', m)
m = [x.signed() for x in m]
m.sort()
m = mpc.run(mpc.output(mpc.input(secint(m[mpc.id]))))
print('Sorted outputs, one per party: ', m)

secfxp = mpc.SecFxp()

m = mpc.run(mpc.output(mpc.input(secfxp(0.5 - random.randint(0, 99)))))
print('Random inputs, one per party: ', m)
m = [x.signed() for x in m]
m.sort()
m = mpc.run(mpc.output(mpc.input(secfxp(m[mpc.id]))))
print('Sorted outputs, one per party: ', m)

mpc.shutdown()

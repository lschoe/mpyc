import random
import sys
from mpyc.runtime import mpc

n = len(mpc.parties)

if n % 2 == 0:
    print('OT runs with odd number of parties only')
    sys.exit()

t = n // 2
message = [(None, None)] * t
choice = [None] * t
if mpc.id == 0:
    print('You are the trusted third party.')
elif 1 <= mpc.id <= t:
    message[mpc.id - 1] = (random.randint(0, 99), random.randint(0, 99))
    print('You are sender %d holding messages %d and %d.' % (mpc.id, message[mpc.id - 1][0], message[mpc.id - 1][1]))
else:
    choice[mpc.id - t - 1] = random.randint(0, 1)
    print('You are receiver %d with random choice bit %d.' % (mpc.id - t, choice[mpc.id - t - 1]))

mpc.start()

secnum = mpc.SecInt()
for i in range(1, t + 1):
    m0 = mpc.input(secnum(message[i - 1][0]), i)
    m1 = mpc.input(secnum(message[i - 1][1]), i)
    b = mpc.input(secnum(choice[i - t - 1]), t + i)
    m = mpc.run(mpc.output(m0 +  b * (m1 - m0), t + i))
    if m:
        print('You have received message %s.' % m)

mpc.shutdown()

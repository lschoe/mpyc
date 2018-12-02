"""Demo oblivious transfer (OT)."""
import random
import sys
from mpyc.runtime import mpc

m = len(mpc.parties)

if m % 2 == 0:
    print('OT runs with odd number of parties only.')
    sys.exit()

t = m // 2
message = [(None, None)] * t
choice = [None] * t
if mpc.pid == 0:
    print('You are the trusted third party.')
elif 1 <= mpc.pid <= t:
    message[mpc.pid - 1] = (random.randint(0, 99), random.randint(0, 99))
    print(f'You are sender {mpc.pid} holding messages {message[mpc.pid - 1][0]} and {message[mpc.pid - 1][1]}.')
else:
    choice[mpc.pid - t - 1] = random.randint(0, 1)
    print(f'You are receiver {mpc.pid - t} with random choice bit {choice[mpc.pid - t - 1]}.')

mpc.run(mpc.start())

secnum = mpc.SecInt()
for i in range(1, t + 1):
    x = mpc.input([secnum(message[i - 1][0]), secnum(message[i - 1][1])], i)
    b = mpc.input(secnum(choice[i - t - 1]), t + i)
    a = mpc.run(mpc.output(mpc.if_else(b, x[1], x[0]), t + i))
    if a:
        print(f'You have received message {a}.')

mpc.run(mpc.shutdown())

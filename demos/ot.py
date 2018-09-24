"""Demo oblivious transfer (OT)."""
import random
import sys
from mpyc.runtime import mpc

n = len(mpc.parties)

if n % 2 == 0:
    print('OT runs with odd number of parties only.')
    sys.exit()

t = n // 2
message = [(None, None)] * t
choice = [None] * t
if mpc.id == 0:
    print('You are the trusted third party.')
elif 1 <= mpc.id <= t:
    message[mpc.id - 1] = (random.randint(0, 99), random.randint(0, 99))
    print(f'You are sender {mpc.id} holding messages {message[mpc.id - 1][0]} and {message[mpc.id - 1][1]}.')
else:
    choice[mpc.id - t - 1] = random.randint(0, 1)
    print(f'You are receiver {mpc.id - t} with random choice bit {choice[mpc.id - t - 1]}.')

mpc.start()

secnum = mpc.SecInt()
for i in range(1, t + 1):
    m = mpc.input([secnum(message[i - 1][0]), secnum(message[i - 1][1])], i)
    b = mpc.input(secnum(choice[i - t - 1]), t + i)
    m = mpc.run(mpc.output(m[0] +  b * (m[1] - m[0]), t + i))
    if m:
        print(f'You have received message {m}.')

mpc.shutdown()

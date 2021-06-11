"""Demo oblivious transfer (OT).

See https://en.wikipedia.org/wiki/Oblivious_transfer for an introduction to OT.

The demo shows how to work with private inputs and private outputs in MPyC. When
run with m = 2t+1 parties, the demo performs t oblivious transfers in parallel.
Each oblivious transfer is a 1-out-of-2 OT:

          Sender                                  Receiver
                           |---------------|
    messages x[0],x[1] --->|               |<--- b in {0,1}
                           | 1-out-of-2 OT |
                       <---|               |---> x[b]
                           |---------------|

The sender holds two messages x[0], x[1] as private input to the protocol.
The receiver chooses a bit b as its private input, and obtains the message x[b]
as its private output. The sender receives no output from the protocol.

The security properties of 1-out-of-2 OT are two-fold:
 * The receiver obtains no information on the sender's other message x[1-b].
 * The sender obtains no information on the receiver's choice bit b.

Achieving these security properties is nontrivial, and one needs to make some kind
of assumption. For instance, there are lots of ways to implement 1-out-of-2 OT
using a public key cryptosystem with appropriate algebraic properties. See the
abovementioned Wikipedia page for an example.

In this demo, we will rely on the honest majority assumption underpinning MPyC.
An honest majority is obtained by using a third party P[0] next to a sender P[1]
and a receiver P[2], and assuming P[0] will not collude with either P[1] or P[2].
Party P[0] has no input/output, and will only see uniformly random values (secret
shares) during the entire protocol. Therefore, P[0] can also be viewed as a
trusted helper.

Viewing the messages as numbers of an appropriate type (e.g., secure integers),
1-out-of-2 OT can be implemented as follows:

    OT(x[0],x[1]; b) = (1 - b) * x[0] + b * x[1]

Or, saving one secure multiplication:

    OT(x[0],x[1]; b) = x[0] + b * (x[1] - x[0])

Which is equivalent to using the MPyC if_else() method:

    OT(x[0],x[1]; b) = mpc.if_else(b, x[1], x[0])

In all cases, we get x[b] as output.

In the demo we do this t times in parallel, when run with m = 2t+1 parties. Party P[0]
is the only helper. Parties P[1] and P[t+1] run an OT-instance as sender and receiver,
respectively. Similarly, parties P[2] and P[t+2] run another OT-instance. And so on.

The MPyC input() and output() methods are used to establish private input and output.
For instance, to let party P[1] provide an input value V to the protocol, all parties
issue the following call:

    mpc.input(secnum(v), 1)

where party P[1] sets v=V for this call, and all other parties set v=None. This way all
parties will obtain the (secret-shared) value V of the same secure type secnum.

Similarly, to let party P[2] receive a secret-shared value s as private output, all
parties issue the following call:

    mpc.output(s, 2)

which will give the value of s in the clear for P[2], and value None for all other parties.

The demo can be run using a command like this:

    python ot.py -M17 -T1 --output-file

This runs 8 OTs in parallel with P[0] as trusted helper, P[1] to P[8] as senders and
P[9] to P[16] as the corresponding receivers. The threshold for the maximum number of
colluding parties is set to 1, ensuring that the protocol is secure as long as the
trusted helper does not collude with any of the senders or receivers.

Use of --output-file triggers output of all results to log files in the current directory.
See python ot.py -H for more options.
"""

import random
import sys
from mpyc.runtime import mpc

m = len(mpc.parties)

if m%2 == 0:
    print('OT runs with odd number of parties only.')
    sys.exit()

t = m//2
message = [(None, None)] * t
choice = [None] * t
if mpc.pid == 0:
    print('You are the trusted third party.')
elif 1 <= mpc.pid <= t:
    message[mpc.pid - 1] = (random.randint(0, 99), random.randint(0, 99))
    print(f'You are sender {mpc.pid} holding messages '
          f'{message[mpc.pid - 1][0]} and {message[mpc.pid - 1][1]}.')
else:
    choice[mpc.pid - t - 1] = random.randint(0, 1)
    print(f'You are receiver {mpc.pid - t} with random choice bit {choice[mpc.pid - t - 1]}.')

mpc.run(mpc.start())

secnum = mpc.SecInt()
for i in range(1, t+1):
    x = mpc.input([secnum(message[i-1][0]), secnum(message[i-1][1])], i)
    b = mpc.input(secnum(choice[i-1]), t + i)
    a = mpc.run(mpc.output(mpc.if_else(b, x[1], x[0]), t + i))
    if a is not None:
        print(f'You have received message {a}.')

mpc.run(mpc.shutdown())

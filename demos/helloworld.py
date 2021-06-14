"""Demo Hello World in MPyC.

Example usage from the command line:

    python helloworld.py -M3

to run the demo with m=3 parties on localhost.

When run with m parties, a total of m(m-1)/2 TCP connections will be created between
all parties, noting that TCP connections are full-duplex (bidirectional). So there are
no connections when run with m=1 party only, there is one connection for m=2 parties, and
there are three connections for m=3 parties.

With all m parties running on localhost, your OS may run out of available ports for large m,
and the program will therefore not terminate properly. For example, the default range for
dynamic (private) ports on Windows is 49152-65535, which will take you to around m=180 parties,
before exhausting the available ports. The range for dynamic ports can be increased like this,
requiring administrator privileges:

    netsh int ipv4 set dynamicport tcp start=16000 num=48000

Now run the demo (as a nice stress test) with m=300, for a total of 44850 TCP connections:

    python helloworld.py -M300 -T0

It it essential to use threshold t=0 (or, maybe t=1). Otherwise the time needed to set up the
PRSS (Pseudoranom Secret Sharing) keys, which is proportional to (m choose t) = m!/t!/(m-t)!,
will be prohibitive.

Alternatively, since the PRSS keys are not actually used in this simple demo, the demo can also
be run with PRSS disabled:

    python helloworld.py -M300 --no-prss

This way there is no need to lower the threshold t.
"""

from mpyc.runtime import mpc

mpc.run(mpc.start())     # connect to all other parties
print(''.join(mpc.run(mpc.transfer('Hello world!'))))
mpc.run(mpc.shutdown())  # disconnect, but only once all other parties reached this point

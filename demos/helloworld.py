"""Hello World in MPyC.

Example usage from the command line:

    python helloworld.py -M3

to run the demo with m=3 parties on localhost.

When run with m parties, a total of m(m-1)/2 TCP connections will be created between
all parties, noting that TCP connections are full-duplex (bidirectional). So there are
no connections when run with m=1 party only, there is one connection for m=2 parties, and
there are three connections for m=3 parties.

For the maximum value of m=256 parties, as currently supported by MPyC, there will be a
total of 32640 TCP connections. With all parties running on localhost, your OS may run out
of available ports, and the program will therefore not terminate.

For example, the default range for dynamic (private) ports on Windows is 49152-65535,
which will take you to around m=180 parties, before exhausting the available ports.
To reach m=256 parties the range for dynamic ports can be increased like this, requiring
administrator privileges:

    netsh int ipv4 set dynamicport tcp start=30000 num=35000

Now run the demo (as a nice stress test):

    python helloworld.py -M256 -T0

It it essential to use threshold t=0 (or, maybe t=1). Otherwise the time needed to set up
the PRSS keys, which is proportional to (m choose t) = m!/t!/(m-t)!, will be prohibitive.
"""

from mpyc.runtime import mpc

mpc.run(mpc.start())     # connect to all other parties
print(''.join(mpc.run(mpc.transfer('Hello world!'))))
mpc.run(mpc.shutdown())  # disconnect, but only once all other parties reached this point

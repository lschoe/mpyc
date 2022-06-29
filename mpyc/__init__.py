"""MPyC is a Python package for secure multiparty computation (MPC).

MPyC provides a runtime for performing computations on secret-shared values,
where parties interact by exchanging messages via peer-to-peer connections.
The MPC protocols are based on Shamir's threshold secret sharing scheme
and withstand passive adversaries controlling less than half of the parties.

Secure integer and fixed-point arithmetic is supported for parameterized
number ranges, also including support for comparison and bitwise operations.
Secure finite field arithmetic is supported for fields of arbitrary order.
Basic support for secure floating-point arithmetic is provided as well.
Moreover, support for secure finite group operations is built-in for a range
of groups, particularly for use in threshold cryptography (e.g., Schnorr groups
and elliptic curves). These operations are all available via Python's operator
overloading.

Secure drop-in replacements for lots of Python built-in functions, such as
all(), any(), sum(), min(), max(), sorted() are provided, mimicking the Python
APIs as much as possible. Further operations for container datatypes holding
secret-shared data items are provided as well (e.g., matrix-vector operations
like secure dot products).

And much more functionality still in a couple of extension modules: seclists
(secure lists with oblivious access and updates), mpctools (reduce and accumulate
with log round complexity), random (securely mimicking Python’s random module),
and statistics (securely mimicking Python’s statistics module).
"""

__version__ = '0.8.5'
__license__ = 'MIT License'

import argparse


def get_arg_parser():
    """Return parser for command line arguments passed to the MPyC runtime."""
    parser = argparse.ArgumentParser(add_help=False)

    group = parser.add_argument_group('MPyC help')
    group.add_argument('-H', '--HELP', action='store_true',
                       help='show this help message for MPyC and exit')
    group.add_argument('-h', '--help', action='store_true',
                       help='show help message for this MPyC program (if any)')

    group = parser.add_argument_group('MPyC configuration')
    group.add_argument('-C', '--config', metavar='ini',
                       help='use ini file, defining all m parties')
    group.add_argument('-P', type=str, dest='parties', metavar='addr', action='append',
                       help='use addr=host:port per party (repeat m times)')
    group.add_argument('-M', type=int, metavar='m',
                       help='use m local parties (and run all m, if i is not set)')
    group.add_argument('-I', '--index', type=int, metavar='i',
                       help='set index of this local party to i, 0<=i<m')
    group.add_argument('-T', '--threshold', type=int, metavar='t',
                       help='threshold t, 0<=t<m/2')
    group.add_argument('-B', '--base-port', type=int, metavar='b',
                       help='use port number b+i for party i')
    group.add_argument('--ssl', action='store_true',
                       help='enable SSL connections')

    group = parser.add_argument_group('MPyC parameters')
    group.add_argument('-L', '--bit-length', type=int, metavar='l',
                       help='default bit length l for secure numbers')
    group.add_argument('-K', '--sec-param', type=int, metavar='k',
                       help='security parameter k, leakage probability 2**-k')
    group.add_argument('--no-log', action='store_true',
                       help='disable logging messages')
    group.add_argument('--no-async', action='store_true',
                       help='disable asynchronous evaluation')
    group.add_argument('--no-barrier', action='store_true',
                       help='disable barriers')
    group.add_argument('--no-gmpy2', action='store_true',
                       help='disable use of gmpy2 package')
    group.add_argument('--no-prss', action='store_true',
                       help='disable use of PRSS (pseudorandom secret sharing)')
    group.add_argument('--mix32-64bit', action='store_true',
                       help='enable mix of 32-bit and 64-bit platforms')

    group = parser.add_argument_group('MPyC misc')
    group.add_argument('--output-windows', action='store_true',
                       help='screen output for parties 0<i<m (only on Windows)')
    group.add_argument('--output-file', action='store_true',
                       help='append output of parties 0<i<m to party{m}_{i}.log')
    group.add_argument('-f', type=str, default='',
                       help='consume IPython\'s -f argument F')

    parser.set_defaults(bit_length=32, sec_param=30)
    return parser

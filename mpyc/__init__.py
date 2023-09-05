"""MPyC is a Python package for secure multiparty computation (MPC).

MPyC provides a runtime for performing computations on secret-shared values,
where parties interact by exchanging messages via peer-to-peer connections.
The MPC protocols are based on Shamir's threshold secret sharing scheme
and withstand passive adversaries controlling less than half of the parties.

Secure integer and fixed-point arithmetic are supported for parameterized
number ranges, also including support for comparison and bitwise operations.
Secure finite field arithmetic is supported for fields of arbitrary order.
Secure NumPy arrays over these basic types are available as well.

Basic support for secure floating-point arithmetic is provided. Moreover,
support for secure finite group operations is built-in for a range of groups,
particularly for use in threshold cryptography (e.g., Schnorr groups and
elliptic curves).

The above operations are all available via Python's operator overloading.

Secure drop-in replacements for lots of Python built-in functions, such as
all(), any(), sum(), min(), max(), sorted() are provided, mimicking the Python
APIs as much as possible. Further operations for container datatypes holding
secret-shared data items are provided as well (e.g., matrix-vector operations
like secure dot products), next to the support for NumPy arrays.

And much more functionality still in a couple of extension modules: seclists
(secure lists with oblivious access and updates), mpctools (reduce and accumulate
with log round complexity), random (securely mimicking Python’s random module),
and statistics (securely mimicking Python’s statistics module).
"""

__version__ = '0.9.4'
__license__ = 'MIT License'

import os
import sys
import argparse
import logging
import importlib.util


def get_arg_parser():
    """Return parser for command line arguments passed to the MPyC runtime."""
    parser = argparse.ArgumentParser(add_help=False)

    group = parser.add_argument_group('MPyC help')
    group.add_argument('-V', '--VERSION', action='store_true',
                       help='print MPyC version number and exit')
    group.add_argument('-H', '--HELP', action='store_true',
                       help='print this help message for MPyC and exit')
    group.add_argument('-h', '--help', action='store_true',
                       help='print help message for this MPyC program (if any)')

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
    group.add_argument('-W', '--workers', type=int, metavar='w',
                       help='maximum number of worker threads per party')

    group = parser.add_argument_group('MPyC parameters')
    group.add_argument('-L', '--bit-length', type=int, metavar='l',
                       help='default bit length l for secure numbers')
    group.add_argument('-K', '--sec-param', type=int, metavar='k',
                       help='security parameter k, leakage probability 2**-k')
    group.add_argument('--log-level', type=str, metavar='ll',
                       help='logging level ll=debug/info(default)/warning/error')
    group.add_argument('--no-log', action='store_true',
                       help='disable logging messages')
    group.add_argument('--no-async', action='store_true',
                       help='disable asynchronous evaluation')
    group.add_argument('--no-barrier', action='store_true',
                       help='disable barriers')
    group.add_argument('--no-gmpy2', action='store_true',
                       help='disable use of gmpy2 package')
    group.add_argument('--no-numpy', action='store_true',
                       help='disable use of numpy package')
    group.add_argument('--no-prss', action='store_true',
                       help='disable use of PRSS (pseudorandom secret sharing)')
    group.add_argument('--mix32-64bit', action='store_true',
                       help='enable mix of 32-bit and 64-bit platforms')

    group = parser.add_argument_group('MPyC misc')
    group.add_argument('--output-windows', action='store_true',
                       help='screen output for parties 0<i<m (one window each)')
    group.add_argument('--output-file', action='store_true',
                       help='append output of parties 0<i<m to party{m}_{i}.log')
    group.add_argument('-f', type=str, default='',
                       help='consume IPython\'s -f argument F')

    parser.set_defaults(bit_length=32, sec_param=30, log_level='info')
    return parser


if os.getenv('READTHEDOCS') != 'True':
    options = get_arg_parser().parse_known_args()[0]
    if options.VERSION or options.HELP:
        options.no_log = True

    # Set logging level as early as possible.
    if options.no_log:
        logging.basicConfig(level=logging.WARNING)
    else:
        ch = options.log_level[0].upper()
        ch = {'N': '0', 'D': '1', 'I': '2', 'W': '3', 'E': '4', 'C': '5'}.get(ch, ch)
        ch = ch if '0' <= ch <= '5' else '0'  # default to '0'
        level = int(ch)
        level = (logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR,
                 logging.CRITICAL)[level]
        if sys.flags.dev_mode:
            # Switch to debug mode, just like asyncio does in development mode.
            level = logging.DEBUG
        logging.basicConfig(format='{asctime} {message}', style='{', level=level, stream=sys.stdout)
        logging.debug(f'Set logging level to {level}: {logging.getLevelName(level)}')
        del ch, level
    logging.debug(f'On {sys.platform=}')

    # Experimental speedup for mpyc.finfields.PrimeFieldArray._sqrt() using multithreading.
    env_max_workers = os.getenv('MPYC_MAXWORKERS')  # check if variable MPYC_MAXWORKERS is set
    if not env_max_workers:
        if options.workers is None:
            options.workers = 0
        os.environ['MPYC_MAXWORKERS'] = str(options.workers)
        # NB: MPYC_MAXWORKERS also set for subprocesses
    logging.debug(f'Number of worker threads maximum set to {os.getenv("MPYC_MAXWORKERS")}')

    # Ensure numpy will not be loaded by mpyc.numpy, if demanded (saving resources).
    env_no_numpy = os.getenv('MPYC_NONUMPY') == '1'  # check if variable MPYC_NONUMPY is set
    if not importlib.util.find_spec('numpy'):
        # numpy package not available
        if not (options.no_numpy or env_no_numpy):
            logging.info('Install package numpy for more functionality.')
    else:
        # numpy package available
        if options.no_numpy or env_no_numpy:
            logging.info('Use of package numpy inside MPyC disabled.')
            if not env_no_numpy:
                os.environ['MPYC_NONUMPY'] = '1'  # NB: MPYC_NONUMPY also set for subprocesses

    # Ensure gmpy2 will not be loaded by mpyc.gmpy, if demanded (using stubs instead).
    env_no_gmpy2 = os.getenv('MPYC_NOGMPY') == '1'  # check if variable MPYC_NOGMPY is set
    if not importlib.util.find_spec('gmpy2'):
        # gmpy2 package not available
        if not (options.no_gmpy2 or env_no_gmpy2):
            logging.info('Install package gmpy2 for better performance.')
    else:
        # gmpy2 package available
        if options.no_gmpy2 or env_no_gmpy2:
            logging.info('Use of package gmpy2 inside MPyC disabled.')
            if not env_no_gmpy2:
                os.environ['MPYC_NOGMPY'] = '1'  # NB: MPYC_NOGMPY also set for subprocesses

    del options, env_max_workers, env_no_numpy, env_no_gmpy2

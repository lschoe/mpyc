"""Generate party configuration files.

Example: three parties on hostnames foo, bar, and baz using port 11000.
Configuration files are generated as follows:

    generate_config_files.py foo:11000 bar:11000 baz:11000

If the parties are on the same host (localhost), then use different
port numbers for each party.

Each party has its own configuration file. The parties are numbered
in the order listed, so Party 0 is on host foo and has configuration
file party3_0.ini. Similarly for Party 1 and Party 2.

The party owning a configuration file is the one for which the hostname
is not specified.
"""

import argparse
from mpyc.runtime import generate_configs

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prefix',
                    help='output filename prefix')
parser.add_argument('-m', '--parties', dest='m', type=int,
                    help='number of parties')
parser.add_argument('args', nargs='*')

parser.set_defaults(m=3, prefix='party')

options = parser.parse_args()
args = options.args

if len(args) != options.m:
    parser.error('A hostname:port argument required for each party.')

addresses = [arg.split(':', 1) for arg in args]
configs = generate_configs(options.m, addresses)

for party, config in enumerate(configs):
    filename = f'{options.prefix}{options.m}_{party}.ini'
    config.write(open(filename, 'w'))

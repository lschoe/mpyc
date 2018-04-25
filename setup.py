# This is the MPyC setup script.
#
# Options:                python setup.py --help
# Install by admin/root:  python setup.py install
# Install by user:        python setup.py install --user
# Install options:        python setup.py install --help

import setuptools
import mpyc

long_description=\
"""
MPyC is a framework for secure multi-party computation (MPC). 

Features include:

* secret sharing based on Shamir and pseudo-random secret sharing (PRSS).

* secure arithmetic and comparisons with shares from a prime field.

* computations with any number of parties assuming an honest majority.

All operations are automatically scheduled to run in parallel meaning
that an operation starts as soon as the operands are ready.
"""

setuptools.setup(name='mpyc',
    version=mpyc.__version__,
    author='Berry Schoenmakers',
    author_email='berry@win.tue.nl',
    description='A framework for secure multi-party computation.',
    long_description=long_description,
    keywords=['crypto', 'cryptography', 'multi-party computation', 'MPC',
            'secret sharing', 'Shamir threshold scheme', 
            'pseudorandom secret sharing', 'PRSS'
            ],
    license=mpyc.__license__,
    packages=['mpyc'],
    platforms=['any'],
    python_requires='>=3.6'
    )

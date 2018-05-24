# MPyC setup script.
#
# Options:                python setup.py --help
# Install by admin/root:  python setup.py install
# Install by user:        python setup.py install --user
# Install options:        python setup.py install --help

import setuptools
import mpyc

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(name='mpyc',
    version=mpyc.__version__,
    author='Berry Schoenmakers',
    author_email='berry@win.tue.nl',
    description='A framework for secure multiparty computation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['crypto', 'cryptography', 'multiparty computation', 'MPC',
            'secret sharing', 'Shamir threshold scheme', 
            'pseudorandom secret sharing', 'PRSS'
            ],
    license=mpyc.__license__,
    packages=['mpyc'],
    platforms=['any'],
    python_requires='>=3.6'
    )

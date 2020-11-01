"""MPyC setup script.

Options:                python setup.py --help
Install by admin/root:  python setup.py install
Install by user:        python setup.py install --user
Install options:        python setup.py install --help
"""

from setuptools import setup
import mpyc

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='mpyc',
    version=mpyc.__version__,
    author='Berry Schoenmakers',
    author_email='berry@win.tue.nl',
    url='https://github.com/lschoe/mpyc',
    description='MPyC -- Secure Multiparty Computation in Python',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords=['crypto', 'cryptography', 'multiparty computation', 'MPC',
              'secret sharing', 'Shamir threshold scheme',
              'pseudorandom secret sharing', 'PRSS'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Framework :: AsyncIO',
        'Framework :: Jupyter',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Security :: Cryptography',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Distributed Computing'
    ],
    license=mpyc.__license__,
    packages=['mpyc'],
    platforms=['any'],
    python_requires='>=3.6'
)

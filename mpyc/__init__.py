"""MPyC is a Python package for secure multiparty computation (MPC).

MPyC provides a runtime for performing computations on secret-shared values,
where parties interact by exchanging messages via peer-to-peer connections.
The MPC protocols are based on Shamir's threshold secret sharing scheme
and withstand passive adversaries controlling less than half of the parties.

Secure integer and fixed-point arithmetic is supported for parameterized
number ranges, also including support for comparison and bitwise operations.
Secure finite field arithmetic is supported for fields of arbitrary order, as
long as the order exceeds the number of parties. Basic support for secure
floating-point arithmetic is provided as well. These operations are all
available via Python's operator overloading.

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

__version__ = '0.7.7'
__license__ = 'MIT License'

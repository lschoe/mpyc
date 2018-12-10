""" MPyC is a Python package for secure multi-party computation (MPC).

MPyC provides a runtime for performing computations on secret-shared values,
where parties interact by exchanging messages via peer-to-peer connections.
The MPC protocols are based on Shamir's threshold secret sharing scheme
and withstand passive adversaries controlling less than half of the parties.

Secure finite field arithmetic is supported for fields of prime order
and fields of characteristic two. Secure integer and fixed-point arithmetic
is supported for parameterized number ranges, also including support
for comparison and bitwise operations.
These operations are all available via Python's operator overloading.
Some operations for container datatypes holding secret-shared data items
are provided as well (e.g., some matrix-vector operations).
"""

__version__ = '0.4.7'
__license__ = 'Apache License 2.0'

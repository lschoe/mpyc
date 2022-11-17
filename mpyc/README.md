## Synopsis

[MPyC](https://lschoe.github.io/mpyc) currently consists of 14 modules (all in pure Python):

1. [numpy](https://lschoe.github.io/mpyc/mpyc.numpy.html): stub to avoid dependency on NumPy package (also handling version issues, etc.)
2. [gmpy](https://lschoe.github.io/mpyc/mpyc.gmpy.html): some basic number theoretic algorithms (using GMP via Python package gmpy2, if installed)
3. [gfpx](https://lschoe.github.io/mpyc/mpyc.gfpx.html): polynomial arithmetic over arbitrary prime fields
4. [finfields](https://lschoe.github.io/mpyc/mpyc.finfields.html): arbitrary finite fields, including binary fields and prime fields
5. [fingroups](https://lschoe.github.io/mpyc/mpyc.fingroups.html): finite groups, in particular for use in cryptography (elliptic curves, Schnorr groups, etc.)
6. [thresha](https://lschoe.github.io/mpyc/mpyc.thresha.html): threshold Shamir (and also pseudorandom) secret sharing
7. [sectypes](https://lschoe.github.io/mpyc/mpyc.sectypes.html): SecInt/Fld/Fxp/Flt types for secure (secret-shared) integer/finite-field/fixed-/floating-point values
8. [asyncoro](https://lschoe.github.io/mpyc/mpyc.asyncoro.html): asynchronous communication and computation of secret-shared values
9. [runtime](https://lschoe.github.io/mpyc/mpyc.runtime.html): core MPC protocols (many hidden by Python's operator overloading)
10. [mpctools](https://lschoe.github.io/mpyc/mpyc.mpctools.html): reduce and accumulate with log round complexity
11. [seclists](https://lschoe.github.io/mpyc/mpyc.seclists.html): secure lists with oblivious access and updates
12. [secgroups](https://lschoe.github.io/mpyc/mpyc.secgroups.html): SecGrp types for secure (secret-shared) finite group elements
13. [random](https://lschoe.github.io/mpyc/mpyc.random.html): securely mimicking Python’s [random](https://docs.python.org/3/library/random.html) module
14. [statistics](https://lschoe.github.io/mpyc/mpyc.statistics.html): securely mimicking Python’s [statistics](https://docs.python.org/3/library/statistics.html) module

The modules are listed in topological order w.r.t. internal dependencies:

- Modules 1-5 are basic modules which can also be used outside an MPC context
- Modules 6-9 form the core of MPyC
- Modules 10-12 form the extended core of MPyC
- Modules 13-14 are small libraries on top of the (extended) core
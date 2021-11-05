## Synopsis

[MPyC](https://lschoe.github.io/mpyc) currently consists of 13 modules (all in pure Python):

1. [gmpy](https://lschoe.github.io/mpyc/mpyc.gmpy.html): some basic number theoretic algorithms (using GMP via Python package gmpy2, if installed)
2. [gfpx](https://lschoe.github.io/mpyc/mpyc.gfpx.html): polynomial arithmetic over arbitrary prime fields
3. [finfields](https://lschoe.github.io/mpyc/mpyc.finfields.html): arbitrary finite fields, including binary fields and prime fields
4. [fingroups](https://lschoe.github.io/mpyc/mpyc.fingroups.html): finite groups, in particular for use in cryptography (elliptic curves, etc.)
5. [thresha](https://lschoe.github.io/mpyc/mpyc.thresha.html): threshold Shamir (and also pseudorandom) secret sharing
6. [sectypes](https://lschoe.github.io/mpyc/mpyc.sectypes.html): SecInt/Fxp/Fld/Flt types for secret-shared integer/fixed-point/finite-field/floating-point values
7. [asyncoro](https://lschoe.github.io/mpyc/mpyc.asyncoro.html): asynchronous communication and computation of secret-shared values
8. [runtime](https://lschoe.github.io/mpyc/mpyc.runtime.html): core MPC protocols (mostly hidden by Python's operator overloading)
9. [mpctools](https://lschoe.github.io/mpyc/mpyc.mpctools.html): reduce and accumulate with log round complexity
10. [seclists](https://lschoe.github.io/mpyc/mpyc.seclists.html): secure lists with oblivious access and updates
11. [secgroups](https://lschoe.github.io/mpyc/mpyc.secgroups.html): SecGrp types for secure (secret-shared) finite group elements
12. [random](https://lschoe.github.io/mpyc/mpyc.random.html): securely mimicking Python’s [random](https://docs.python.org/3/library/random.html) module
13. [statistics](https://lschoe.github.io/mpyc/mpyc.statistics.html): securely mimicking Python’s [statistics](https://docs.python.org/3/library/statistics.html) module

The modules are listed in topological order w.r.t. internal dependencies:

- Modules 1-4 are basic modules which can also be used outside an MPC context
- Modules 5-8 form the core of MPyC
- Modules 9-11 form the extended core of MPyC
- Modules 12-13 are small libraries on top of the (extended) core
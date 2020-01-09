## Synopsis

[MPyC](https://lschoe.github.io/mpyc) currently consists of 11 modules (all in pure Python):

1. [gmpy](https://lschoe.github.io/mpyc/mpyc.gmpy.html): some basic number theoretic algorithms (using GMP via Python package gmpy2, if installed)
2. [gfpx](https://lschoe.github.io/mpyc/mpyc.gfpx.html): polynomial arithmetic over arbitrary prime fields
3. [finfields](https://lschoe.github.io/mpyc/mpyc.finfields.html): arbitrary finite fields, including binary fields and prime fields
4. [thresha](https://lschoe.github.io/mpyc/mpyc.thresha.html): threshold Shamir (and also pseudorandom) secret sharing
5. [sectypes](https://lschoe.github.io/mpyc/mpyc.sectypes.html): secure types SecInt/Fxp/Fld for secret-shared integer/fixed-point/finite-field values
6. [asyncoro](https://lschoe.github.io/mpyc/mpyc.asyncoro.html): asynchronous communication and computation of secret-shared values
7. [runtime](https://lschoe.github.io/mpyc/mpyc.runtime.html): core MPC protocols (mostly hidden by Python's operator overloading)
8. [seclists](https://lschoe.github.io/mpyc/mpyc.seclists.html): secure lists with oblivious access and updates
9. [mpctools](https://lschoe.github.io/mpyc/mpyc.mpctools.html): reduce and accumulate with log round complexity
10. [random](https://lschoe.github.io/mpyc/mpyc.random.html): securely mimicking Python’s [random](https://docs.python.org/3/library/random.html) module
11. [statistics](https://lschoe.github.io/mpyc/mpyc.statistics.html): securely mimicking Python’s [statistics](https://docs.python.org/3/library/statistics.html) module

The modules are listed in topological order w.r.t. internal dependencies:

- Modules 1-3 are basic modules which can also be used outside an MPC context
- Modules 1-8 form the core of MPyC
- Modules 9-11 are small libraries on top of the core
# MPyC -- Secure Multiparty Computation in Python.

MPyC supports secure *m*-party computation tolerating a dishonest minority of up to *t* passively corrupt parties,
where *m>=1* and *0 <= t <= (m-1)/2*. The underlying protocols are based on threshold secret sharing over prime
fields (using Shamir's threshold scheme as well as pseudorandom secret sharing).

The details of the secure computation protocols are mostly transparent due to the use of sophisticated operator overloading
combined with asynchronous evaluation of the associated protocols.

See [MPyC homepage](https://www.win.tue.nl/~berry/mpyc/) for more info and background.

## Example installs:

`python setup.py install`

`python setup.py install --user`

See `demos` for usage examples.

## Notes:

1. Python 3.6 or higher required (Python 3.5 or lower is not sufficient).

2. Installing package `gmpy2` is optional, but will considerably benefit the performance of `mpyc`.
On Linux, `pip install gmpy2` should do the job, but on Windows, this may fail with compiler errors.
Fortunately, ready-to-go Python wheels for `gmpy2` can be downloaded from Christoph Gohlke's excellent
[Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/) webpage.
Use, for example, `pip install gmpy2-2.0.8-cp36-cp36m-win_amd64.whl` to finish installation.

3. A few simple Windows batch files are provided in the `demos` directory.

4. Directory `demos\.config` contains configuration info and key material needed to run MPyC with 
multiple parties. Windows batch file 'gen.bat' shows how to generate fresh key material for pseudorandom 
secret sharing and SSL. OpenSSL is required to generate SSL key material of your own, use `pip install pyOpenSSL`.

5. To use the [Jupyter](https://jupyter.org/) notebooks `demos\*.ipynb`, 
you need to have Jupyter installed, e.g., using `pip install jupyter`. 
 
6. Latest versions of Jupyter use Tornado 5.0, which will not work with MPyC, see
[Jupyter notebook issue #3397](https://github.com/jupyter/notebook/issues/3397).
Downgrade Tornado by running `pip install tornado==4.5.3`.

Copyright &copy; 2018, Berry Schoenmakers

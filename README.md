[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lschoe/mpyc/master)
[![Travis CI](https://travis-ci.com/lschoe/mpyc.svg)](https://travis-ci.com/lschoe/mpyc)
[![codecov](https://codecov.io/gh/lschoe/mpyc/branch/master/graph/badge.svg)](https://codecov.io/gh/lschoe/mpyc)
[![Read the Docs](https://readthedocs.org/projects/mpyc/badge/)](https://mpyc.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/mpyc.svg)](https://pypi.org/project/mpyc/)

# MPyC [![MPyC logo](https://raw.githubusercontent.com/lschoe/mpyc/master/images/MPyC_Logo.svg)](https://github.com/lschoe/mpyc) Secure Multiparty Computation in Python

MPyC supports secure *m*-party computation tolerating a dishonest minority of up to *t* passively corrupt parties,
where *m &ge; 1* and *0 &le; t &lt; m/2*. The underlying cryptographic protocols are based on threshold secret sharing over finite
fields (using Shamir's threshold scheme as well as pseudorandom secret sharing).

The details of the secure computation protocols are mostly transparent due to the use of sophisticated operator overloading
combined with asynchronous evaluation of the associated protocols.

See the [MPyC homepage](https://www.win.tue.nl/~berry/mpyc/) for more info and background.

Click the "launch binder" badge above to view the entire repository and try out the Jupyter notebooks from the `demos` directory
in the cloud, without any install.

## Installation:

Just run: `python setup.py install` (pure Python, no dependencies).

See `demos` for Python programs and Jupyter notebooks with lots of example code.

See [Read the Docs](https://mpyc.readthedocs.io/) for `Sphinx`-based documentation, including an overview of the `demos`,
and [GitHub Pages](https://lschoe.github.io/mpyc/) for `pydoc`-based documentation.

## Notes:

1. Python 3.6+ (Python 3.5 or lower is not sufficient).

2. Installing package `gmpy2` is optional, but will considerably enhance the performance of `mpyc`.
If you use the [conda](https://docs.conda.io/) package and environment manager, `conda install gmpy2` should do the job.
Otherwise, `pip install gmpy2` can be used on Linux (first running `apt install libmpc-dev` may be necessary too),
but on Windows, this may fail with compiler errors.
Fortunately, ready-to-go Python wheels for `gmpy2` can be downloaded from Christoph Gohlke's excellent
[Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/) webpage.
Use, for example, `pip install gmpy2-2.0.8-cp39-cp39-win_amd64.whl` to finish installation.

3. Use `run-all.sh` or `run-all.bat` in the `demos` directory to have a quick look at all pure Python demos.
The demos `bnnmnist.py` and `cnnmnist.py` require [NumPy](https://www.numpy.org/), the demo `kmsurvival.py` requires
[pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and [lifelines](https://pypi.org/project/lifelines/),
and the demo `ridgeregression.py` even requires [Scikit-learn](https://scikit-learn.org/). Also note the example Linux shell
scripts and Windows batch files in the `docs` and `tests` directories.

4. Directory `demos\.config` contains configuration info used to run MPyC with multiple parties. Also,
Windows batch file 'gen.bat' shows how to generate fresh key material for SSL. OpenSSL is required to generate
SSL key material of your own, use `pip install pyOpenSSL`.

5. To use the [Jupyter](https://jupyter.org/) notebooks `demos\*.ipynb`, you need to have Jupyter installed,
e.g., using `pip install jupyter`. The latest version of Jupyter will come with IPython 7.x, which supports
top-level `await`. For example, instead of `mpc.run(mpc.start())` one can now simply write `await mpc.start()` anywhere in
a notebook cell, even outside a coroutine.

6. For Python 3.8+, you also get top-level `await` by running `python -m asyncio` to launch a natively async REPL.
By running `python -m mpyc` instead you even get this REPL with the MPyC runtime preloaded!

Copyright &copy; 2018-2021 Berry Schoenmakers
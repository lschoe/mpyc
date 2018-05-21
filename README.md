# mpyc
MPyC for Secure Multiparty Computation in Python

## Example installs:

`python setup.py install`

`python setup.py install --user`

See `demos` for usage examples.

## Notes:

1. Python 3.6 required (Python 3.5 or lower is not sufficient).

2. A few simple Windows batch files are provided in the `demos` directory. Linux equivalents will follow soon.

3. Latest versions of Jupyter use Tornado 5.0, which will not work with MPyC, see [Jupyter notebook issue #3397](https://github.com/jupyter/notebook/issues/3397). Downgrade Tornado by running `pip install tornado==4.5.3`.

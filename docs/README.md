The [MPyC demos](../demos) provide Python programs and Jupyter notebooks with lots of example code.

Technical reference documentation on MPyC is (currently) offered in two similar ways.

## Docs generated with Python's native pydoc module

At [GitHub Pages](https://lschoe.github.io/mpyc/) you can browse documentation generated with [pydoc](https://docs.python.org/3/library/pydoc.html) for the latest MPyC release.

Run `python -m pydoc` with the `-b` option to browse the documentation on your own system, see also [doc.bat](doc.bat) or [doc.sh](doc.sh).

## Docs generated with Sphinx

At [Read the Docs](https://mpyc.readthedocs.io/) you can browse documentation generated with [Sphinx](https://www.sphinx-doc.org/) for the latest MPyC version.

Run `make html` or `make latex` to generate documentation with Sphinx on your own system, controlled by `conf.py` and the `.rst` files.
Set the environment variable `READTHEDOCS=True` to prevent problems when `mpyc/runtime.py` is imported by `autodoc`.
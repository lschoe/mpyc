MPyC installation
=================

MPyC runs on Windows/Mac/Linux platforms supporting Python 3.8+.
There are no dependencies on other Python packages.

If you first want to try MPyC online, click
`launch binder <https://mybinder.org/v2/gh/lschoe/mpyc/master>`_ to run MPyC and the demos
in your browser with `JupyterLab <https://jupyterlab.readthedocs.io>`_  (running multiple
Jupyter notebooks and Linux terminals next to each other).

Quick start
-----------

You can get started right away with::

   pip install mpyc

Then run (downloaded) programs like `helloworld.py <https://github.com/lschoe/mpyc/blob/master/demos/helloworld.py>`_::

   python helloworld.py

Maybe the hardest part is to spell "MPyC", which is pronounced as "em-pie-sea"
and stands for "secure Multiparty Computation in Python."

Latest version
--------------

Using ``pip install mpyc`` will get you the latest `major` release from `PyPI <https://pypi.org>`_.
To get the latest `minor` version from `GitHub <https://github.com>`_ use::

   pip install git+https://github.com/lschoe/mpyc

With ``pip show mpyc`` you will see that both ways the package source files are installed in Python's
``site-packages`` directory.

.. _github clone:

GitHub clone
------------

Use `Git <https://git-scm.com/>`_ (or `GitHub Desktop <https://desktop.github.com/>`_) to clone
the `MPyC repo <https://github.com/lschoe/mpyc>`_ for a local copy including all source files, demos, unit tests, etc.
This way it will also be easy to get updates and switch between versions. The same content
is also available as a ZIP file `mpyc-master.zip <https://github.com/lschoe/mpyc/archive/refs/heads/master.zip>`_.

From the root directory of your local copy (containing ``setup.py``) you may then run::

   pip install .

to (re)install MPyC.

Change to the ``demos`` directory for running a demo with one or more parties, or
change to the ``tests`` directory for checking if the unit tests pass.
In the ``docs`` directory you'll find pregenerated documentation in HTML format, and ways to generate
fresh documentation yourself (with Python's built-in ``pydoc``, or with `Sphinx <https://www.sphinx-doc.org/>`_).
See the ``README.md`` files in these directories for more information.

Deployment
----------

Currently Python 3.8+ is the only requirement.

Written in pure Python, without any dependencies, MPyC should run on common
Windows/Mac/Linux platforms, including the 32-bit Raspberry Pi, if you like.

The parties engaging in a secure multiparty computation are allowed to be on
different 32-bit/64-bit platforms, mixing different implementations of Python.
The parties also need not always run exactly the same version of MPyC.

To enhance the performance of MPyC, it may help to install the
`gmpy2 <https://gmpy2.readthedocs.io>`_ package, e.g., using ``pip install gmpy2``.

PyPy / Nuitka
-------------

Apart from the `CPython <https://www.python.org/>`_ implementation of Python,
also the `PyPy <https://www.pypy.org/>`_ implementation of Python can be used,
sometimes (but certainly not always) yielding a faster result.

Another interesting option is to generate (standalone) executables with the
`Nuitka <https://nuitka.net/>`_ Python compiler, available via ``pip install nuitka``.
For instance, ``nuitka --follow-imports aes.py`` generates an executable
``aes.exe`` for the AES demo with competitive performance. And
``nuitka --onefile aes.py`` even builds a standalone executable, which may be
a simple alternative to `Docker <https://www.docker.com/>`_.

Hacking MPyC
------------

A really nice thing about Python is that programs are run in an interpreter.
This means you can make changes to Python programs and (core) libraries at all
levels, and these changes will take effect immediately upon the next run.

To "hack" MPyC in this way, just do::

   pip install -e .

in the root directory of your MPyC project, assuming you are working with a local
copy as explained above in :ref:`GitHub clone <github clone>`. The ``-e`` option
makes your copy `editable`.

Now you can make changes to the source code of MPyC, and see the effect. Which means
that if you run a program in an arbitrary directory on your computer, your
"hacked" version of MPyC will be used. In combination with ``git`` you can easily
undo your changes, and work on your own ``git`` branch (or fork on GitHub).

If you are often switching between different versions of Python, then append
the root directory of your MPyC project to the environment variable ``PYTHONPATH``.
This way, any Python version you may be using will run with your version of MPyC.

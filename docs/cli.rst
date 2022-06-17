.. _command line:

MPyC command line
=================

An MPyC program is a Python program that uses the MPyC runtime,
typically by including ``from mypc.runtime import mpc``.
A range of options is available to control the
MPyC program and its runtime from the command line.

To get help about the MPyC options, use switch ``-H`` (or, ``--HELP``) like in these examples:

.. code-block::

   $ python -m mpyc -H
   $ python helloworld.py -H
   $ python -c "from mpyc.runtime import mpc" -H

In addition, an MPyC program may have its own command line interface,
on top of the MPyC interface, accessible through switch ``-h`` (or, ``--help``) like in this example:

.. code-block::

   $ python bnnmnist.py -h
   Showing help message for bnnmnist.py, if available:

   usage: bnnmnist.py [-h] [-b B] [-o O] [-d D] [--no-legendre] [--no-vectorization]

   options:
     -h, --help            show this help message and exit
     -b B, --batch-size B  number of images to classify
     -o O, --offset O      offset for batch (otherwise random in [0,10000-B])
     -d D, --d-k-star D    k=D=0,1,2 for Legendre-based comparison using d_k^*
     --no-legendre         disable Legendre-based comparison
     --no-vectorization    disable vectorization of comparisons

It is useful to note that an MPyC program is not required to provide help via ``-h``,
hence ``-h`` may simply be ignored, or the program is not able to handle it.

The command line options for the MPyC runtime generally use upper case for single-letter options,
reserving lower case for the single-letter options of MPyC programs:

.. argparse::
   :module: mpyc
   :func: get_arg_parser
   :prog: mpycprog.py

   MPyC configuration : @before
      To set up the parties and network connections.

   MPyC parameters : @before
      To control the secure computation and underlying protocols.

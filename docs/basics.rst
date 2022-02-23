MPyC basics
===========

The goal of MPyC is that you can render nice and useful programs
with relative ease. The complexities of the secure multiparty computation
protocols running "under the hood" are hidden as much possible.

In this brief tutorial we show how to use MPyC in a Python program, focusing
on the specific steps you need to do (extra) to implement a multiparty computation.

Loading MPyC
------------

To load MPyC we use the following ``import`` statement:

.. code-block:: python

    from mpyc.runtime import mpc

This sets up the MPyC runtime ``mpc``, which means that the identities and addresses of
the MPC parties have been loaded. The details of these parties can be inspected via
``mpc.parties``. If available, the package ``gmpy2`` will have been loaded as well.

Secure types
------------

To perform any meaningful multiparty computation in MPyC we'll first need to create
appropriate secure types. For example, to work with secure integers we may create a secure
type ``secint`` as follows:

.. code-block:: python

    secint = mpc.SecInt(16)

Unlike Python integers (type ``int``), secure integers always have a maximum bit length.
Here, we use 16-bit (signed) integers. The limited range of values ensures that "under the hood"
a secure integer can be represented by an element of a finite field.
More precisely, a secure integer is actually stored in secret-shared form, where each party
will hold exactly one share represented as an element of a field of prime order.

You can create as many secure types as you like, mixing secure integers, secure fixed-point numbers,
secure floating-point numbers, and even secure types for (elements of) finite fields and some classes
of finite groups. If you're curious, run ``python -m mpyc`` from the command line to get a list
of secure types to play with.

Secure input
------------

To let all parties provide their age as a private input, use:

.. code-block:: python

    my_age = int(input('Enter your age: '))  # each party enters a number
    our_ages = mpc.input(secint(my_age))  # list with one secint per party

Each party runs its own copy of this code, so ``my_age`` will only be known to the party entering the number.
The value for ``my_age`` is then converted to a secure integer to tell ``mpc.input()`` the type for secret-sharing
the value of ``my_age``.

The list ``our_ages`` will contain the secret-shares of the ages entered by all parties,
represented by one secure integer of type ``secint`` per party.

Secure computation
------------------

We perform some computations to determine the total age, the maximum age, and the number of ages above average:

.. code-block:: python

    total_age = sum(our_ages)
    max_age = mpc.max(our_ages)
    m = len(mpc.parties)
    above_avg = 0
    for age in our_ages:
        above_avg += mpc.if_else(age * m > total_age, 1, 0)

For the total age we can use the Python function ``sum()``, although ``mpc.sum()`` would be slightly faster.
For the maximum age we cannot use the Python function ``max()``, so we use ``mpc.max()``.
To compute the number of ages above average, we compare each ``age`` with the average age ``total_age / m``,
however, avoiding the use of a division.

As the result of a comparison with secure integers is a secret-shared bit, we can get the result more
directly:

.. code-block:: python

    above_avg = mpc.sum(age * m > total_age for age in our_ages)

This time also using ``mpc.sum()`` instead of ``sum()`` for slightly better performance.

Secure output
-------------

Finally, we reveal the results:

.. code-block:: python

    print('Average age:', await mpc.output(total_age) / m)
    print('Maximum age:', await mpc.output(max_age))
    print('Number of "elderly":', await mpc.output(above_avg))

Note that we need to ``await`` the results of the calls to ``mpc.output()``.

Running MPyC
------------

To run the above code with multiple parties, we put everything together,
inserting calls to ``mpc.start()`` and ``mpc.shutdown()`` to let the
parties actually connect and disconnect:

.. code-block:: python
    :caption: elderly.py

    from mpyc.runtime import mpc

    async def main():
        secint = mpc.SecInt(16)

        await mpc.start()

        my_age = int(input('Enter your age: '))
        our_ages = mpc.input(secint(my_age))

        total_age = sum(our_ages)
        max_age = mpc.max(our_ages)
        m = len(mpc.parties)
        above_avg = mpc.sum(age * m > total_age for age in our_ages)

        print('Average age:', await mpc.output(total_age) / m)
        print('Maximum age:', await mpc.output(max_age))
        print('Number of "elderly":', await mpc.output(above_avg))

        await mpc.shutdown()

    mpc.run(main())

We define function ``main()`` as a coroutine (``async`` function) to enable the
use of ``await`` statements. To call and execute coroutine ``main()``,
we use ``mpc.run(main())`` , much the same as one needs to do with any
`coroutine <https://docs.python.org/3/library/asyncio-task.html#id1>`_ in Python.

An example run between three parties on `localhost` looks as follows:

.. code-block::

    $ python elderly.py -M3 -I0 --no-log
    Enter your age: 21
    Average age: 29.0
    Maximum age: 47
    Number of "elderly": 1

.. code-block::

    $ python elderly.py -M3 -I1 --no-log
    Enter your age: 19
    Average age: 29.0
    Maximum age: 47
    Number of "elderly": 1

.. code-block::

    $ python elderly.py -M3 -I2 --no-log
    Enter your age: 47
    Average age: 29.0
    Maximum age: 47
    Number of "elderly": 1

See :ref:`MPyC demos <mpyc demos>` for lots of other examples, including
some more elaborate explanations in Jupyter notebooks.

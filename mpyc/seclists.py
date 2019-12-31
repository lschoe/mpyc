"""This module provides a secure (oblivious) alternative to Python lists.

A secure list contains secret-shared values. Apart from hiding the contents of the
list, however, it is also possible to hide which items are accessed and which items
are updated. In principle, only the length of a secure list remains public.

A secure list x can be cast to an ordinary list by using list(x), without affecting
the contents of the list. Also, public access to a secure list proceeds the same as
for ordinary Python lists, using an index or a slice.

For secure (oblivious) access to a secure list, however, one uses a secret-shared
index i, which is either a secure number or a secure unit vector. Index i must be
compatible with the secure list x: the type of i must fit with the type of the
elements of x and the value of i must fit with the length of list x, that is,
0 <= i < len(x).

Common usage scenarios of secure lists are supported through functions such as count()
and index(), or can be coded easily. For example, the frequency of values in a list x
of secure integers (which are known to be between 0 and n-1) is computed by:

    s = seclist([0]*n, secint)
    for a in x:
        s[a] += 1

Current implementation is basic, taking advantage of cheap secure dot products, as
provided by runtime.in_prod(). Performance for modestly sized lists of lengths 10 to 1000
should be adequate. Later: With better amortized complexity, e.g., square root ORAM.
"""

from mpyc.sectypes import Share
from mpyc.random import random_unit_vector
from mpyc import asyncoro

runtime = None


class seclist(list):

    def __init__(self, x=(), sectype=None):
        """Build a secure list from the items in iterable x using the given secure type.

        If no secure type is given, it is inferred from the items in x.

        Invariant: all items in x are of the same secure type.
        """
        super().__init__(x)
        t = len(self)
        for a in self:
            if isinstance(a, Share):
                t -= 1
                if sectype is None:
                    sectype = type(a)
                elif sectype != type(a):
                    raise TypeError('inconsistent sectypes')

        if sectype is None:
            raise TypeError('sectype missing')

        i = 0
        while t:
            while isinstance(self[i], Share):
                i += 1
            super().__setitem__(i, sectype(self[i]))
            t -= 1
        self.sectype = sectype

    def __getitem__(self, key):
        """Called to evaluate self[key], where key is either public or secret.

        If key is a public integer (or a slice), the behavior is the same as for ordinary lists.
        If key is a secure number or index, the value at the secret position is returned.
        """
        if isinstance(key, (int, slice)):
            value = super().__getitem__(key)
            if isinstance(key, slice):
                value = seclist(value, self.sectype)
            return value

        if isinstance(key, self.sectype):
            i = runtime.unit_vector(key, len(self))
        elif isinstance(key, secindex):
            i = [self.sectype(0)] * key.offset + list(key)
        else:
            i = key
        if len(self) != len(i):
            raise IndexError('inconsistent index length')

        # unary index i of right length
        return runtime.in_prod(list(self), i)

    def __setitem__(self, key, value):
        """Called to set self[key] = value, where key is either public or secret.
        The type of value should fit with the type of the list.

        If key is a public integer (or a slice), the behavior is the same as for ordinary lists.
        If key is a secure number or index, the list is updated at the secret position.
        """
        if isinstance(key, int):
            if not isinstance(value, self.sectype):
                value = self.sectype(value)
            super().__setitem__(key, value)
            return

        if isinstance(key, slice):
            if not isinstance(value, seclist):
                value = seclist(value, self.sectype)
            if not issubclass(value.sectype, self.sectype):
                raise TypeError('inconsistent sectypes')

            super().__setitem__(key, value)
            return

        if not isinstance(value, self.sectype):
            value = self.sectype(value)

        if isinstance(key, self.sectype):
            i = runtime.unit_vector(key, len(self))
        elif isinstance(key, secindex):
            i = [self.sectype(0)] * key.offset + list(key)
        else:
            i = key
        if len(self) != len(i):
            raise IndexError('inconsistent index length')

        # unary index i of right length
        s_i = runtime.in_prod(list(self), i)
        res = runtime.vector_add(list(self), runtime.scalar_mul(value - s_i, i))
        parent = super()
        for j in range(len(self)):
            parent.__setitem__(j, res[j])

    def append(self, other):
        if not isinstance(other, self.sectype):
            other = self.sectype(other)
        super().append(other)

    def extend(self, other):
        if not isinstance(other, seclist):
            other = seclist(other, self.sectype)
        super().extend(other)

    def __add__(self, other):
        sectype = self.sectype
        if not isinstance(other, seclist):
            other = seclist(other, sectype)
        elif sectype != other.sectype:
            raise TypeError('inconsistent sectypes')

        return seclist(super().__add__(other), sectype)

    def __radd__(self, other):
        return seclist(other + list(self), self.sectype)

    def __iadd__(self, other):
        self.extend(other)
        return self

    def copy(self):
        return seclist(super().copy(), self.sectype)

#    def random_index(self):
#        return secindex(random_unit_vector(self.sectype, len(self)))

    def __contains__(self, item):  # NB: item in self does not work in CPython
        """Check if item occurs in self."""
        return self.count(item) != 0

    def count(self, value):
        """Return the number of occurrences of value."""
        return runtime.sum([a == value for a in list(self)])

    @staticmethod
    def _find_one(x):
        """Return (ix, nz), where ix is the index of first 1 in bit list x (ix=len(x) if no 1 in x)
        and nz indicates if x contains a 1.
        """
        n = len(x)
        if n == 1:
            return 1-x[0], x[0]

        ix0, nz0 = seclist._find_one(x[:n//2])  # low bits
        ix1, nz1 = seclist._find_one(x[n//2:])  # high bits
        return runtime.if_else(nz0, [ix0, nz0], [n//2 + ix1, nz1])

    def find(self, value):  # TODO: add optional i and j to indicate slice
        """Return index of the first occurrence of value.

        If value is not present, then index is equal to len(self).
        """
        if not self:
            ix = self.sectype(0)
        else:
            ix = seclist._find_one([a == value for a in list(self)])[0]
        return ix

    @asyncoro.mpc_coro
    async def index(self, value):  # TODO: add optional i and j to indicate slice
        """Return index of the first occurrence of value.

        Raise ValueError if value is not present.
        """
        await runtime.returnType((self.sectype, True))

        ix = self.find(value)
        if await runtime.eq_public(ix, len(self)):
            raise ValueError(f'value is not in list')

        return ix


class secindex(list):
    """Provisional class to facilitate more efficient manipulation of secure indices."""

    def __init__(self, *args, offset=0, sectype=None):
        if sectype is not None:
            super().__init__(*args)
        else:
            x = seclist(*args)
            sectype = x.sectype  # infer sectype from elements of x
            super().__init__(x)
        self.offset = offset
        self.sectype = sectype

    def __add__(self, other):
        sectype = type(self[0])
        i = runtime.in_prod(list(self), [sectype(_) for _ in range(len(self))])
        j = runtime.in_prod(list(other), [sectype(_) for _ in range(len(other))])
        k = runtime.unit_vector(i + j, len(self) + len(other) - 1)
        offset = self.offset + other.offset
        return secindex(k, offset=offset)

    async def __index__(self):
        sectype = type(self[0])
        f = sectype.field.frac_length
        i = await runtime.output(runtime.in_prod(list(self), list(map(sectype, range(len(self))))))
        return self.offset + i.value >> f

    def __await__(self):
        return self.__index__().__await__()

    @staticmethod
    def random(sectype, length, offset=0):
        n = length
        return secindex(random_unit_vector(sectype, n), offset=offset)

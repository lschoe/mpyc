"""This module provides a secure (oblivious) alternative to Python lists.

A secure list contains secret-shared numbers. Apart from hiding the contents of the
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

Common usage scenarios of secure lists are supported through methods such as sort(),
count(), and index(), or can be coded easily. For example, the frequency of values
in a list x of secure integers (whose values are known to be between 0 and n-1)
is computed by:

    s = seclist([0]*n, secint)

    for a in x: s[a] += 1

Current implementation is basic, taking advantage of cheap secure dot products, as
provided by runtime.in_prod(). Performance for modestly sized lists of lengths 10 to 1000
should be adequate. Later: With better amortized complexity, e.g., square root ORAM.
"""

from asyncio import Future
from mpyc.sectypes import SecureObject, SecureFixedPoint
from mpyc.random import random_unit_vector
from mpyc import asyncoro

runtime = None


class seclist(list):

    sectype = None

    def __init__(self, x=(), sectype=None):
        """Build a secure list from the items in iterable x using the given secure type.

        If no secure type is given, it is inferred from the items in x.

        Invariant: all items in a secure list are of the same secure type.
        """
        parent = super()
        parent.__init__(x)
        t = len(self)
        for a in self:
            if isinstance(a, SecureObject):
                t -= 1
                if sectype is None:
                    sectype = type(a)
                elif sectype != type(a):
                    raise TypeError('inconsistent sectypes')

        if sectype is None:
            raise ValueError('sectype missing')

        i = 0
        while t:
            a = parent.__getitem__(i)
            while isinstance(a, SecureObject):
                i += 1
                a = parent.__getitem__(i)
            parent.__setitem__(i, sectype(a))
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
            i = [self.sectype(0)] * key.offset + key.value
        else:
            i = key
        if len(i) != len(self):
            raise IndexError('inconsistent index length')

        # unary index i of proper length
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

        n = len(self)
        if isinstance(key, self.sectype):
            i = runtime.unit_vector(key, n)
        elif isinstance(key, secindex):
            i = [self.sectype(0)] * key.offset + key.value
        else:
            i = key
        if len(i) != n:
            raise IndexError('inconsistent index length')

        # unary index i of proper length
        x = list(self)
        x_i = runtime.in_prod(x, i)
        x = runtime.vector_add(x, runtime.scalar_mul(value - x_i, i))
        super().__init__(x)

    def __delitem__(self, key):
        """Called to delete self[key], where key is either public or secret.

        If key is a public integer (or a slice), the behavior is the same as for ordinary lists.
        If key is a secure number or index, the list element at the secret position is removed.
        """
        if isinstance(key, (int, slice)):
            super().__delitem__(key)
            return

        n = len(self)
        if isinstance(key, self.sectype):
            i = runtime.unit_vector(key, n)
        elif isinstance(key, secindex):
            i = [self.sectype(0)] * key.offset + key.value
        else:
            i = key[:]
        i.pop()
        if len(i) != n-1:
            raise IndexError('inconsistent index length')

        # unary index i of proper length
        for j in range(1, n-1):
            i[j] += i[j-1]
        # step function i
        x = list(self)
        x1 = x[1:]
        x.pop()
        delta = runtime.schur_prod(i, runtime.vector_sub(x1, x))
        x = runtime.vector_add(x, delta)
        super().__init__(x)

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

    def __mul__(self, other):
        return seclist(super().__mul__(other), self.sectype)

    def __rmul__(self, other):
        return seclist(super().__mul__(other), self.sectype)

    def __imul__(self, other):
        super().__imul__(other)
        return self

    def insert(self, key, value):
        """Insert value before position given by key, where key is either public or secret.
        The key should fit with the length of the list: 0 <= key <= len(self).
        The type of value should fit with the type of the list.

        If key is a public integer, the behavior is the same as for ordinary lists.
        If key is a secure number or index, the value is inserted at the secret position.
        """
        if isinstance(key, int):
            if not isinstance(value, self.sectype):
                value = self.sectype(value)
            super().insert(key, value)
            return

        n = len(self)
        if isinstance(key, self.sectype):
            i = runtime.unit_vector(key, n+1)
        elif isinstance(key, secindex):
            i = [self.sectype(0)] * key.offset + key.value
        else:
            i = key[:]
        if len(i) != n+1:
            raise IndexError('inconsistent index length')

        # unary index i of proper length
        zero = self.sectype(0)
        x = list(self)
        x.insert(0, zero)
        x_i = runtime.in_prod(x, i)
        y = runtime.vector_add(x, runtime.scalar_mul(value - x_i, i))
        x.pop(0)
        x.append(zero)
        for j in range(1, n+1):
            i[j] += i[j-1]
        # step function i
        delta = runtime.schur_prod(i, runtime.vector_sub(y, x))
        x = runtime.vector_add(x, delta)
        super().__init__(x)

    def pop(self, key=-1):
        """Remove and return value at position given by key, where key is either public or secret.

        If key is a public integer, the behavior is the same as for ordinary lists.
        If key is a secure number or index, the item at the secret position is removed,
        where 0 <= key < len(self).
        """
        if isinstance(key, int):
            return super().pop(key)

        n = len(self)
        if isinstance(key, self.sectype):
            i = runtime.unit_vector(key, n)
        elif isinstance(key, secindex):
            i = [self.sectype(0)] * key.offset + key.value
        else:
            i = key
        if len(i) != n:
            raise IndexError('inconsistent index length')

        # unary index i of proper length
        x_i = runtime.in_prod(list(self), i)
        self.__delitem__(i)
        return x_i

    @asyncoro.mpc_coro
    async def remove(self, value) -> Future:
        """Remove first occurrence of value.

        Raise ValueError if value is not present.
        """
        i = self.find(value)
        if await runtime.eq_public(i, -1):
            raise ValueError('value is not in list')

        self.__delitem__(i)

    def copy(self):
        return seclist(super().copy(), self.sectype)

#    def random_index(self):
#        return secindex(random_unit_vector(self.sectype, len(self)))

    def __contains__(self, item):
        """Not implemented for secure lists.

        Corresponds to "item in self", which is defined as a public Boolean value in Python.
        Instead, use seclist.contains(self, item) to get a secure Boolean result in MPyC.
        """
        raise NotImplementedError('use seclist.contains()')

    def contains(self, item):
        """Check if item occurs in self."""
        return self.count(item) != 0

    def count(self, value):
        """Return the number of occurrences of value."""
        return runtime.sum([a == value for a in list(self)])

    def find(self, value):  # TODO: add optional i and j to indicate slice
        """Return index of the first occurrence of value.

        If value is not present, then index is equal to -1.
        """
        if not self:
            ix = self.sectype(-1)
        else:
            ix = runtime.find(list(self), value, bits=False, e=-1)
        return ix

    def index(self, value):  # TODO: add optional i and j to indicate slice
        """Return index of the first occurrence of value.

        Raise ValueError if value is not present.
        """
        return runtime.indexOf(list(self), value, bits=False)

    def sort(self, key=None, reverse=False):
        """Sort the list in-place, similar to Python's list.sort().

        See runtime.sorted() for details on key etc.
        """
        if len(self) < 2:
            return

        if key is None:
            key = lambda a: a
        runtime._sort(self, key)
        if reverse:
            self.reverse()
        return

    @staticmethod
    def _norm(stype, x, x2, EQ=False):
        n = len(x)
        if n == 1:
            if not EQ:
                a = (x2[0] - x[0])/stype.field(2)
            else:
                a = 1 - (x2[0] + x[0])/stype.field(2)
            if issubclass(stype, SecureFixedPoint):
                a.integral = True  # NB: in fact, a is a bit in {0,1}
            return a, x2[0]

        lte0, nz0 = seclist._norm(stype, x[:n//2], x2[:n//2], EQ)  # low positions
        lte1, nz1 = seclist._norm(stype, x[n//2:], x2[n//2:], EQ)  # high positions
        lte, nz = runtime.if_else(nz0, [lte0, nz0], [lte1, nz1])
        return lte, nz

    @staticmethod
    def _less_than(stype, x, y):
        s = [runtime.sgn(a - b) for a, b in zip(x, y)]
        if not s:  # x=[] or y=[]
            return stype(bool(y))  # x < y iff y!=[]

        s2 = runtime.schur_prod(s, s)
        EQ = len(x) < len(y)
        return seclist._norm(stype, s, s2, EQ=EQ)[0]

    def __lt__(self, other):
        return seclist._less_than(self.sectype, self, other)

    def __le__(self, other):
        return 1 - seclist._less_than(self.sectype, other, self)

    def __eq__(self, other):
        if len(self) != len(other):
            return self.sectype(0)

        return runtime.all(a == b for a, b in zip(self, other))

    def __ge__(self, other):
        return 1 - seclist._less_than(self.sectype, self, other)

    def __gt__(self, other):
        return seclist._less_than(self.sectype, other, self)

    def __ne__(self, other):
        return 1 - self.__eq__(other)


class secindex:
    """Provisional class to facilitate more efficient manipulation of secure indices."""

    __slots__ = 'value', 'offset', 'sectype'

    def __init__(self, *args, offset=0, sectype=None):
        if sectype is not None:
            self.value = list(*args)
        else:
            x = seclist(*args)
            sectype = x.sectype  # infer sectype from elements of x
            self.value = list(x)
        self.offset = offset
        self.sectype = sectype

    def __add__(self, other):
        field = self.sectype.field
        m = len(self.value)
        n = len(other.value)
        i = runtime.in_prod(self.value, [field(_) for _ in range(m)])
        j = runtime.in_prod(other.value, [field(_) for _ in range(n)])
        k = runtime.unit_vector(i + j, m + n - 1)
        offset = self.offset + other.offset
        return secindex(k, offset=offset)

    async def __index__(self):
        field = self.sectype.field
        f = self.sectype.frac_length
        i = runtime.in_prod(self.value, [field(_) for _ in range(len(self.value))])
        i = await runtime.output(i)
        return self.offset + i.value >> f

    def __await__(self):
        return self.__index__().__await__()

    @staticmethod
    def random(sectype, length, offset=0):
        n = length
        return secindex(random_unit_vector(sectype, n), offset=offset)

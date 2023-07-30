"""This module supports finite (Galois) fields.

Function GF creates types implementing finite fields.
Instantiate an object from a field and subsequently apply overloaded
operators such as +,-,*,/ etc., to compute with field elements.
In-place versions of the field operators are also provided.
Taking square roots and quadratic residuosity tests supported as well.

Moreover, (multidimensional) arrays over finite fields are available
with operators like +,-,*,/ for convenient and efficient NumPy-based
vectorized processing next to operator @ for matrix multiplication.
Much of the NumPy API can be used to manipulate these arrays as well.
"""

import os
import functools
from mpyc.numpy import np
from mpyc import gmpy as gmpy2
from mpyc import gfpx


def GF(modulus):
    """Create a finite (Galois) field for given modulus (prime number or irreducible polynomial).

    Also creates corresponding array type.
    """
    if isinstance(modulus, gfpx.Polynomial):
        field = xGF(modulus)
    else:
        if isinstance(modulus, tuple):
            p, n, w = modulus
        else:
            p = modulus
            if p == 2:
                n, w = 1, 1
            else:
                n, w = 2, p-1
        field = pGF(p, n, w)

    if not np:
        return field

    if issubclass(field, PrimeFieldElement):
        BaseFFArray = PrimeFieldArray
    elif issubclass(field, BinaryFieldElement):
        BaseFFArray = BinaryFieldArray
    else:  # issubclass(field, ExtensionFieldElement)
        BaseFFArray = ExtensionFieldArray
    name = f'Array{field.__name__}'
    array = type(name, (BaseFFArray,), {'__slots__': ()})
    array.field = field
    field.array = array
    globals()[name] = array  # NB: exploit (almost?) unique name dynamic array type
    return field


class FiniteFieldElement:
    """Abstract base class for finite field elements.

    Invariant: 'value' is reduced w.r.t. modulus.
    """

    __slots__ = 'value'

    modulus: type  # set by subclass
    order = None
    characteristic = None
    ext_deg = None
    byte_length = None
    is_signed = None
    array: type
    _mix_types: type  # or, a tuple of types

    def __init__(self, value):
        self.value = value

    def __array_function__(self, func, types, args, kwargs):
        # Redirect call to corresponding array type.
        return self.array.__array_function__(self, func, types, args, kwargs)

    def __int__(self):
        """Extract field element as an integer value."""
        raise NotImplementedError('abstract method')

    @classmethod
    def to_bytes(cls, x):
        """Return byte string representing the given list/ndarray of integers x."""
        byte_order = 'little'
        r = cls.byte_length
        return r.to_bytes(2, byte_order) + b''.join(v.to_bytes(r, byte_order) for v in x)

    @staticmethod
    def from_bytes(data):
        """Return the list of integers represented by the given byte string."""
        byte_order = 'little'
        from_bytes = int.from_bytes  # cache
        r = from_bytes(data[:2], byte_order)
        return [from_bytes(data[i:i+r], byte_order) for i in range(2, len(data), r)]

    def __add__(self, other):
        """Addition."""
        if isinstance(other, type(self)):
            return type(self)(self.value + other.value)

        if isinstance(other, self._mix_types):
            return type(self)(self.value + other)

        return NotImplemented

    def __radd__(self, other):
        """Addition (with reflected arguments)."""
        if isinstance(other, self._mix_types):
            return type(self)(self.value + other)

        return NotImplemented

    def __iadd__(self, other):
        """In-place addition."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        self.value += other
        self.value %= self.modulus
        return self

    def __sub__(self, other):
        """Subtraction."""
        if isinstance(other, type(self)):
            return type(self)(self.value - other.value)

        if isinstance(other, self._mix_types):
            return type(self)(self.value - other)

        return NotImplemented

    def __rsub__(self, other):
        """Subtraction (with reflected arguments)."""
        if isinstance(other, self._mix_types):
            return type(self)(other - self.value)

        return NotImplemented

    def __isub__(self, other):
        """In-place subtraction."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        self.value -= other
        self.value %= self.modulus
        return self

    def __neg__(self):
        """Negation."""
        return type(self)(-self.value)

    def __pos__(self):
        """Unary +."""
        return type(self)(+self.value)

    def __mul__(self, other):
        """Multiplication."""
        if isinstance(other, type(self)):
            return type(self)(self.value * other.value)

        if isinstance(other, self._mix_types):
            return type(self)(self.value * other)

        return NotImplemented

    def __rmul__(self, other):
        """Multiplication (with reflected arguments)."""
        if isinstance(other, self._mix_types):
            return type(self)(self.value * other)

        return NotImplemented

    def __imul__(self, other):
        """In-place multiplication."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        self.value *= other
        self.value %= self.modulus
        return self

    def __truediv__(self, other):
        """Division."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        return self * type(self)._reciprocal(other)

    def __rtruediv__(self, other):
        """Division (with reflected arguments)."""
        if isinstance(other, self._mix_types):
            return self.reciprocal() * other

        return NotImplemented

    def __itruediv__(self, other):
        """In-place division."""
        if isinstance(other, type(self)):
            other = other.value
        elif not isinstance(other, self._mix_types):
            return NotImplemented

        self.value *= type(self)._reciprocal(other)
        self.value %= self.modulus
        return self

    def __pow__(self, other):
        """Exponentiation."""
        raise NotImplementedError('abstract method')

    def __lshift__(self, other):
        """Left shift."""
        if not isinstance(other, int):
            return NotImplemented

        return type(self)(self.value << other)

    def __rlshift__(self, other):
        """Left shift (with reflected arguments)."""
        return NotImplemented

    def __ilshift__(self, other):
        """In-place left shift."""
        if not isinstance(other, int):
            return NotImplemented

        self.value <<= other
        self.value %= self.modulus
        return self

    def __rshift__(self, other):
        """Right shift."""
        raise NotImplementedError('abstract method')

    def __rrshift__(self, other):
        """Right shift (with reflected arguments)."""
        return NotImplemented

    def __irshift__(self, other):
        """In-place right shift."""
        raise NotImplementedError('abstract method')

    @classmethod
    def _reciprocal(cls, a):
        """Multiplicative inverse."""
        raise NotImplementedError('abstract method')

    def reciprocal(self):
        """Multiplicative inverse."""
        cls = type(self)
        return cls(cls._reciprocal(self.value))

    @classmethod
    def _sqrt(cls, a, INV=False):
        """Modular (inverse) square root."""
        raise NotImplementedError('abstract method')

    def sqrt(self, INV=False):
        """Modular (inverse) square root."""
        cls = type(self)
        return cls(cls._sqrt(self.value, INV=INV))

    @classmethod
    def _is_sqr(cls, a):
        """Test for quadratic residuosity (0 is also square)."""
        raise NotImplementedError('abstract method')

    def is_sqr(self):
        """Test for quadratic residuosity (0 is also square)."""
        return self._is_sqr(self.value)

    def __eq__(self, other):
        """Equality test."""
        if isinstance(other, type(self)):
            return self.value == other.value

        if isinstance(other, self._mix_types):
            return self.value == other % self.modulus

        return NotImplemented

    def __hash__(self):
        """Make finite field elements hashable (e.g., for LRU caching)."""
        return hash((type(self).__name__, self.value))

    def __bool__(self):
        """Truth value testing.

        Return False if this field element is zero, True otherwise.
        Field elements can thus be used directly in Boolean formulas.
        """
        return bool(self.value)


def find_prime_root(l, blum=True, n=1):
    """Find prime of bit length at least l satisfying given constraints.

    Default is to return Blum primes (primes p with p % 4 == 3).
    Also, a primitive root w is returned of prime order at least n (0 < w < p).
    """
    if l <= 2:
        if not blum:
            p = 2
            assert n == 1
            w = 1
        else:
            p = 3
            n, w = 2, p-1
    elif n <= 2:
        p = gmpy2.prev_prime(1 << l)
        if blum:
            while p%4 != 3:
                p = gmpy2.prev_prime(p)
        p = int(p)
        w = p-1 if n == 2 else 1
    else:
        assert blum
        if not gmpy2.is_prime(n):
            n = gmpy2.next_prime(n)
        p = 1 + 2*n * (3 + 2*((1 << l-3) // n))
        while not gmpy2.is_prime(p):
            p += 4*n

        a = 2
        while (w := gmpy2.powmod(a, (p-1) // n, p)) == 1:
            a += 1
        p, n, w = int(p), int(n), int(w)
    return p, n, w


@functools.cache
def pGF(p, n, w):
    """Create a finite field for given prime modulus p."""
    if not gmpy2.is_prime(p):
        raise ValueError('modulus is not a prime')

    GFp = type(f'GF({p})', (PrimeFieldElement,), {'__slots__': ()})
    GFp.__doc__ = 'Class of prime field elements.'
    GFp.modulus = p
    GFp.order = p
    GFp.characteristic = p
    GFp.ext_deg = 1
    GFp.byte_length = (GFp.order.bit_length() + 7) >> 3
    GFp.is_signed = True
    GFp.nth = n
    GFp.root = w % p
    return GFp


class PrimeFieldElement(FiniteFieldElement):
    """Common base class for prime field elements."""

    __slots__ = ()

    modulus: int
    is_signed = None
    nth = None
    root = None
    _mix_types = int

    def __init__(self, value):
        if not isinstance(value, int):
            raise TypeError(f'int required, got {type(value).__name__}')

        # Directly call int.__mod__() for efficiency:
        value = value.__mod__(self.modulus)
        super().__init__(value)

    @staticmethod
    def createGF(p, n, w):
        """Create new object for use by pickle module."""
        obj = pGF(p, n, w)
        return PrimeFieldElement.__new__(obj)

    def __reduce__(self):
        return (PrimeFieldElement.createGF, (self.modulus, self.nth, self.root),
                (None, {'value': self.value}))

    def __int__(self):
        """Extract field element as a (signed) integer value."""
        if self.is_signed:
            v = self.signed_()
        else:
            v = self.unsigned_()
        return v

    def __abs__(self):
        """Absolute value of (signed) value."""
        return abs(self.__int__())

    def __pow__(self, other):
        """Exponentiation."""
        if not isinstance(other, int):
            return NotImplemented

        return type(self)(int(gmpy2.powmod(self.value, other, self.modulus)))

    @classmethod
    def _reciprocal(cls, a):
        return int(gmpy2.invert(a, cls.modulus))

    @classmethod
    @functools.lru_cache(maxsize=1)  # NB: 1-place cache to speed up rshift for secure trunc() etc.
    def _reciprocal2(cls, n):
        """Return multiplicative inverse of 2**n, n>=0."""
        return cls._reciprocal(1 << n)

    def __rshift__(self, other):
        """Right shift."""
        if not isinstance(other, int):
            return NotImplemented

        cls = type(self)
        return cls(self.value * cls._reciprocal2(other))

    def __irshift__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        self.value *= self._reciprocal2(other)
        self.value %= self.modulus
        return self

    @classmethod
    def _sqrt(cls, a, INV=False):
        p = cls.modulus
        if a == 0:
            if INV:
                raise ZeroDivisionError('no inverse sqrt of 0')

            return a

        if p == 2:
            return a

        if p&3 == 3:
            if INV:
                p4 = (p*3 - 5) >> 2  # a**p4 == a**(-1/2) == 1/sqrt(a) mod p
            else:
                p4 = (p+1) >> 2
            return int(gmpy2.powmod(a, p4, p))

        # 1 (mod 4) primes are covered using Cipolla-Lehmer's algorithm.
        # find b s.t. b^2 - 4*a is not a square
        b = 1
        while gmpy2.legendre(b * b - 4*a, p) != -1:
            b += 1

        # compute u*X + v = X^{(p+1)/2} mod f, for f = X^2 - b*X + a
        u, v = 0, 1
        e = (p+1) >> 1
        for i in range(e.bit_length() - 1, -1, -1):
            u2 = (u * u) % p
            u = ((u<<1) * v + b * u2) % p
            v = (v * v - a * u2) % p
            if (e >> i) & 1:
                u, v = (v + b * u) % p, (-a * u) % p
        if INV:
            v = cls._reciprocal(v)

        return v

    @classmethod
    def _is_sqr(cls, a):
        p = cls.modulus
        if p == 2:
            return True

        return gmpy2.legendre(a, p) != -1

    def signed_(self):
        """Return signed integer representation, symmetric around zero."""
        v = self.value
        if v > self.modulus >> 1:
            v -= self.modulus
        return v

    def unsigned_(self):
        """Return unsigned integer representation."""
        return self.value

    def __repr__(self):
        return f'{self.__int__()}'


def find_irreducible(p, d):
    """Find smallest irreducible polynomial of degree d over GF(p)."""
    # TODO: implement constraints, e.g., low weight, primitive
    return gfpx.GFpX(p).next_irreducible(p**d - 1)


@functools.cache
def xGF(modulus):
    """Create a finite field for given irreducible polynomial."""
    p = modulus.p
    poly = gfpx.GFpX(p)
    if not poly.is_irreducible(modulus):
        raise ValueError('modulus is not irreducible')

    d = poly.deg(modulus)
    BaseFieldElement = BinaryFieldElement if p == 2 else ExtensionFieldElement
    GFq = type(f'GF({p}^{d})', (BaseFieldElement,), {'__slots__': ()})
    GFq.__doc__ = f'Class of {"binary" if p == 2 else "extension"} field elements.'
    GFq.modulus = modulus
    GFq.order = p**d
    GFq.characteristic = p
    GFq.ext_deg = d
    GFq.byte_length = (GFq.order.bit_length() + 7) >> 3
    return GFq


class ExtensionFieldElement(FiniteFieldElement):
    """Common base class for extension field elements."""

    __slots__ = ()

    modulus: gfpx.Polynomial
    _least_qnr = None
    _mix_types = (int, gfpx.Polynomial)

    def __init__(self, value):
        cls = type(self.modulus)
        # Directly call gfpx.Polynomial._mod() for efficiency:
        value = cls(cls._mod(cls(value).value, self.modulus.value), check=False)
        super().__init__(value)

    @staticmethod
    def createGF(modulus):
        """Create new object for use by pickle module."""
        obj = xGF(modulus)
        return ExtensionFieldElement.__new__(obj)

    def __reduce__(self):
        return ExtensionFieldElement.createGF, (self.modulus,), (None, {'value': self.value})

    def __int__(self):
        return int(self.value)

    def __pow__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        poly = type(self.value)
        return type(self)(poly.powmod(self.value, other, self.modulus))

    def __rshift__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        return self * self._reciprocal(1 << other)

    def __irshift__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        self.value *= self._reciprocal(1 << other)
        self.value %= self.modulus
        return self

    @classmethod
    def _reciprocal(cls, a):
        return type(cls.modulus).invert(a, cls.modulus)

    @classmethod
    def _sqrt(cls, a, INV=False):
        poly = type(a)
        q = cls.order
        if a == []:
            if INV:
                raise ZeroDivisionError('no inverse sqrt of 0')

            return a

        if q%2 == 0:
            q2 = q >> 1
            if INV:
                q2 -= 1
            return poly.powmod(a, q2, cls.modulus)

        if q&3 == 3:
            if INV:
                q4 = (q*3 - 5) >> 2  # a**q4 == a**(-1/2) == 1/sqrt(a) in GF(q)
            else:
                q4 = (q+1) >> 2
            return poly.powmod(a, q4, cls.modulus)

        # Tonelli-Shanks
        n = q-1
        s = (n & -n).bit_length() - 1  # number of times 2 divides n
        t = n >> s
        # q - 1 = t 2^s, t odd
        z = cls._least_qnr
        if z is None:
            i = 2
            while True:
                z = poly.powmod(i, t, cls.modulus)
                if poly.powmod(z, 1 << s-1, cls.modulus) != 1:
                    break
                i += 1
            cls._least_qnr = z  # cache least QNR raised to power t

        # TODO: improve following code a bit
        w = poly.powmod(a, t>>1, cls.modulus)
        x = a * w % cls.modulus
        b = x * w % cls.modulus
        v = s
        while b != 1:
            b2 = b
            k = 0
            while b2 != 1:
                b2 = b2 * b2 % cls.modulus
                k += 1
            w = poly.powmod(z, 1 << v - k - 1, cls.modulus)
            z = w * w % cls.modulus
            b = b * z % cls.modulus
            x = x * w % cls.modulus
            v = k
        if INV:
            x = cls._reciprocal(x)

        return x

    @classmethod
    def _is_sqr(cls, a):
        poly = type(a)
        q = cls.order
        if q%2 == 0:
            return True

        return poly.powmod(a, (q-1) >> 1, cls.modulus) != [poly.p - 1]

    def __repr__(self):
        return f'{self.value}'


class BinaryFieldElement(ExtensionFieldElement):
    """Common base class for binary field elements."""

    __slots__ = ()

    modulus: gfpx.BinaryPolynomial
    characteristic = 2
    _mix_types = (int, gfpx.BinaryPolynomial)

    @classmethod
    def _is_sqr(cls, a):
        return True

    @classmethod
    def _sqrt(cls, a, INV=False):
        poly = type(a)
        q = cls.order
        if a == 0:
            if INV:
                raise ZeroDivisionError('no inverse sqrt of 0')

            return a

        q2 = q >> 1
        if INV:
            q2 -= 1
        return poly.powmod(a, q2, cls.modulus)


_HANDLED_FUNCTIONS = {}


def _implements(numpy_function_name):
    """Register an __array_function__ implementation."""
    def decorator(func):
        if np:
            numpy_function = eval('np.' + numpy_function_name)
            _HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


class FiniteFieldArray:
    """Common base class for finite field arrays.

    An array over finite field F behaves much like a NumPy array with dtype=F.
    Conceptually, an array a over F can be thought of as, for example

       "a = np.array([[F(1), F(2)], [F(0), F(3)]], dtype=F)"

    Internally, however, arrays over a finite field F are represented by NumPy
    arrays with dtype=object storing just the values of the finite field elements:

       "a = (field=F, value=np.array([[1, 2], [0, 3]], dtype=object))"

    Invariant: elements of array 'value' are reduced w.r.t. modulus.
    """

    __slots__ = 'value'

    field: type  # finite field of the array
    _mix_types: type  # or tuple of types

    def __init__(self, value, check=True, copy=False):
        value = np.array(value, dtype=object, copy=copy)
        if check:
            # TODO: check dtype and type of entries of value (filter float)
            value %= self.field.modulus  # NB: in-place prevents change in shape of value
        self.value = value
        # TODO: optimize using dtype=int8/int16/int32/int64 depending on field.order

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # TODO: make more general
        cls = type(self)
        if any(isinstance(a, np.ndarray) and a.dtype != object and a.dtype not in cls._mix_types
               for a in inputs):
            return NotImplemented

        if ufunc.__name__ == 'equal':
            return inputs[1].__eq__(inputs[0])  # needed to avoid infinite recursion

        if ufunc.__name__ == 'not_equal':
            return inputs[1].__ne__(inputs[0])

        if ufunc.__name__ == 'left_shift':
            return inputs[0] << inputs[1]

        if ufunc.__name__ == 'right_shift':
            return inputs[0] >> inputs[1]

        if ufunc.__name__ == 'power':
            if isinstance(inputs[1], cls):
                return NotImplemented

        if ufunc.__name__ == 'reciprocal':
            return type(self).reciprocal(inputs[0])

        if ufunc.__name__ == 'sqrt':
            return type(self).sqrt(inputs[0])

        inputs = tuple(a.value if isinstance(a, (cls, cls.field)) else a for a in inputs)
        a = getattr(ufunc, method)(*inputs, **kwargs)
        if method == 'at':  # 'at'  is in-place so make sure to enforce invariant
            # TODO: check other methods than '__call__' (at, reduce, reduceat, accumulate, outer)
            self.value = self.value % cls.field.modulus
        else:
            a = cls(a)
        return a

    def __array_function__(self, func, types, args, kwargs):
        # TODO: rework and make more general
        if isinstance(self, FiniteFieldArray):
            cls = type(self)
        elif isinstance(self, FiniteFieldElement):
            cls = type(self).array
        else:
            raise TypeError('wrong type for self in __array_function__(self, ...) call')

        if func in _HANDLED_FUNCTIONS:
            return _HANDLED_FUNCTIONS[func](*args, **kwargs)

        args = list(args)
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, (cls, cls.field)):
                args[i] = arg.value
            elif isinstance(arg, tuple):
                arg = list(arg)
                for j in range(len(arg)):
                    a = arg[j]
                    if isinstance(a, (cls, cls.field)):
                        a = a.value
                    elif not isinstance(a, int) and not isinstance(a, np.ndarray):
                        return NotImplemented

                    arg[j] = a
                args[i] = tuple(arg)
            elif isinstance(arg, list):
                args[i] = [a.value if isinstance(a, (cls, cls.field)) else a for a in arg]
            elif not isinstance(arg, int) and not isinstance(arg, np.ndarray):
                return NotImplemented

        a = func(*args, **kwargs)

        if isinstance(a, np.ndarray):
            if func.__name__ in ('roll',  'diagonal', 'diag_flat'):
                a = cls(a, check=False)
            elif func.__name__ != 'flatnonzero':
                a = cls(a)
        elif isinstance(a, list):
            # for func like vsplit returning list of arrays
            a = list(map(cls, a))
        elif isinstance(a, bool):
            pass
        elif not isinstance(a, tuple):  # shape
            a = cls.field(a)
        return a

    @property
    @_implements('shape')
    def shape(self):
        return self.value.shape

    @property
    @_implements('ndim')
    def ndim(self):
        return self.value.ndim

    @property
    @_implements('size')
    def size(self):
        return self.value.size

    @staticmethod
    @_implements('block')
    def _np_block(arrays):
        def extract_type(s):
            if isinstance(s, list):
                for a in s:
                    if cls := extract_type(a):
                        break
            elif isinstance(s, (FiniteFieldArray, FiniteFieldElement)):
                cls = type(s)
            else:
                cls = None
            return cls

        cls = extract_type(arrays)
        if issubclass(cls, FiniteFieldElement):
            cls = cls.array

        def peel(s):
            if isinstance(s, list):
                s = [peel(_) for _ in s]
            elif isinstance(s, (cls, cls.field)):
                s = s.value
            return s

        a = np.block(peel(arrays))  # NB: a is Numpy array
        return cls(a)

    def __iter__(self):
        for a in self.value:
            if isinstance(a, np.ndarray):
                a = type(self)(a, check=False)
            else:
                a = self.field(a)
            yield a

    @staticmethod
    @_implements('linalg.solve')
    def gauss_solve(A, B):
        """Linear solve by Gaussian elimination on matrix (A | B)."""
        # TODO: extend to more dimensions, solve in last 2 dimensions
        # TODO: remove assumption A and B are finite field arrays (e.g., B = np.eye(n))
        cls = type(A)
        field = cls.field
        modulus = field.modulus  # int or gfpx.Polynomial(0)
        n = A.shape[0]
        if not A.shape == (n, n):
            raise np.linalg.LinAlgError('array must be square')

        A = np.concatenate((A.value, B.value), axis=1)

        # Gaussian elimination
        for k in range(n):
            if A[k, k] == 0:
                for x in range(k+1, n):
                    if A[x, k] != 0:
                        break
                else:
                    raise ZeroDivisionError('no inverse exists')

                A[[k, x]] = A[[x, k]]
            A[k, k] = (1 / field(A[k, k])).value  # store reciprocal of diagonal elts
            A[k+1:, k] = A[k+1:, k] * A[k, k] % modulus
            A[k+1:, k+1:] = (A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:])) % modulus

        # back substitution
        A[n-1, n:] = A[n-1, n:] * A[n-1, n-1] % modulus  # avoid np.dot on empty dtype='O' arrays
        for i in range(n-2, -1, -1):
            A[i, n:] = (A[i, n:] - np.dot(A[i, i+1:n], A[i+1:, n:])) * A[i, i] % modulus

        return cls(A[:, n:], check=False)

    @staticmethod
    @_implements('linalg.inv')
    def gauss_inv(A):
        """Inverse by Gaussian elimination on augmented matrix (A | I)."""
        # TODO: extend to more dimensions, invert last 2 dimensions
        B = type(A)(np.eye(len(A), dtype='O'))
        return FiniteFieldArray.gauss_solve(A, B)

    @staticmethod
    @_implements('linalg.det')
    def gauss_det(a):
        """Determinant by Gaussian elimination on (last 2 dimensions of) array a."""
        cls = type(a)
        field = cls.field
        modulus = field.modulus  # int or gfpx.Polynomial(0)
        if a.ndim < 2 or a.shape[-2] != a.shape[-1]:
            # Let Numpy generate error message by calling det[a] for dummy array a of given shape:
            return np.linalg.det(np.empty(a.shape))

        n = a.shape[-1]
        d = np.empty(a.size // n**2, dtype='O')
        for i, A in enumerate(a.value.reshape((-1, n, n))):
            A = A.copy()

            # Gaussian elimination
            for k in range(n):
                if A[k, k] == 0:
                    for x in range(k+1, n):
                        if A[x, k] != 0:
                            break
                    else:
                        d[i] = 0
                        break

                    A[[k, x]] = A[[x, k]]
                inv_k = (1 / field(A[k, k])).value
                A[k+1:, k] = A[k+1:, k] * inv_k % modulus
                A[k+1:, k+1:] = (A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:])) % modulus
            else:
                d[i] = cls(np.diag(A), check=False).prod().value

        d = d.reshape(a.shape[:-2])
        if d.shape == ():
            d = d[()]
            d = cls.field(d)
        else:
            d = cls(d, check=False)
        return d

    @staticmethod
    @_implements('linalg.matrix_power')
    def matrix_pow(A, n):  # needed for handling negative n
        cls = type(A)
        p = cls.field.modulus

        if n < 0:
            A = np.linalg.inv(A)
            n = -n

        D = A.value
        C = np.eye(len(A), dtype='O')
        for i in range(n.bit_length() - 1):
            # D = A^(2^i) holds
            if (n >> i) & 1:
                C = np.matmul(C, D) % p
            D = np.matmul(D, D) % p
        if n:
            C = np.matmul(C, D) % p
        return cls(C)

    @staticmethod
    @_implements('diag')
    def diag(a, k=0):
        cls = type(a)
        return cls(np.diag(a.value, k), check=False)

    @property
    def flat(self):
        for a in self.value.flat:
            yield self.field(a)

    def __len__(self):
        return len(self.value)

    def __contains__(self, value):
        # TODO: check semantics of __contains__ in NumPy
        cls = type(self)
        if not isinstance(value, (cls, cls.field)):
            value = cls(value)
            if value.ndim == 0:
                # extract single entry
                value = value[()]
        value = value.value

        return self.value.__contains__(value)

    def __getitem__(self, key):
        a = self.value.__getitem__(key)
        if not isinstance(a, np.ndarray):
            return self.field(a)

        return type(self)(a, check=False)  # NB: no copy

    def __setitem__(self, key, value):
        cls = type(self)
        if not isinstance(value, (cls, cls.field)):
            value = cls(value)
            if value.ndim == 0:
                # extract single entry
                value = value[()]
        value = value.value

        # NB: NumPy does not raise exceptions for __setitem__ on dtype='O' arrays
        try:
            # Check if shape of target matches shape of given value.
            shape = np._item_shape(self.value.shape, key)
            np.broadcast_to(value, shape)  # raises if there is a mismatch
        except Exception:
            # Let Numpy generate error message using dummy arrays of given shapes:
            np.empty(self.value.shape)[key] = np.empty(value.shape)

        self.value.__setitem__(key, value)

    def __eq__(self, other):
        """Elementwise equality as for Numpy arrays."""
        cls = type(self)
        if not isinstance(other, (cls, cls.field)):
            other = cls(other)
        return np.equal(self.value, other.value)

    def __ne__(self, other):
        """Elementwise nonequality as for Numpy arrays."""
        cls = type(self)
        if not isinstance(other, (cls, cls.field)):
            other = cls(other)
        return np.not_equal(self.value, other.value)

    @classmethod
    def _coerce(cls, other):
        if isinstance(other, (cls, cls.field)):
            other = other.value
        elif not isinstance(other, cls._mix_types):
            if not isinstance(other, np.ndarray):
                # TODO: check elts and dtype of other?
                return NotImplemented

        return other

    def __add__(self, other):
        """Addition."""
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(self.value + other)

    __radd__ = __add__

    def __iadd__(self, other):
        """In-place addition."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        self.value += other
        self.value %= self.field.modulus
        return self

    def __sub__(self, other):
        """Subtraction."""
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(self.value - other)

    def __rsub__(self, other):
        """Subtraction (with reflected arguments)."""
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(other - self.value)

    def __isub__(self, other):
        """In-place subtraction."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        self.value -= other
        self.value %= self.field.modulus
        return self

    def __mul__(self, other):
        """Multiplication."""
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(self.value * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        """In-place multiplication."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        self.value *= other
        self.value %= self.field.modulus
        return self

    def __matmul__(self, other):
        """Matrix multiplication."""
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        a = self.value @ other
        a = cls(a) if isinstance(a, np.ndarray) else cls.field(a)
        return a

    def __rmatmul__(self, other):
        """Matrix multiplication (with reflected arguments)."""
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        a = other @ self.value
        a = cls(a) if isinstance(a, np.ndarray) else cls.field(a)
        return a

    def __imatmul__(self, other):
        """In-place matrix multiplication."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        self.value @= other  # NB: raises TypeError until NumPy aupports in-place matmul
        self.value %= self.field.modulus
        # TODO: case self.value is not an array, if allowed at all for in-place matmul
        return self

    def __pow__(self, other):
        """Exponentiation."""
        if not isinstance(other, (int, np.int8)):
            if not isinstance(other, np.ndarray):
                # TODO: check dtype?
                return NotImplemented

        cls = type(self)
        return cls(cls._pow(self.value, other), check=False)

    def __rpow__(self, other):
        """Exponentiation (with reflected arguments)."""
        # TODO: keep or omit like in FiniteFieldElement?
        return NotImplemented

    def __ipow__(self, other):
        """In-place exponentiation."""
        if not isinstance(other, (int, np.int8)):
            if not isinstance(other, np.ndarray):
                # TODO: check dtype?
                return NotImplemented

        self.value = self._pow(self.value, other)
        return self

    @classmethod
    def _pow(cls, a, b):
        """Exponentiation."""
        raise NotImplementedError('abstract method')

    def __neg__(self):
        """Negation."""
        a = self.value.__neg__()
        return type(self)(a)

    def __pos__(self):
        """Unary +."""
        a = self.value.__pos__()  # NB: copy
        return type(self)(a, check=False)

    def __truediv__(self, other):
        """Division."""
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(self.value * cls._reciprocal(other))

    def __rtruediv__(self, other):
        """Division (with reflected arguments)."""
        cls = type(self)
        other = cls._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        return cls(other * cls._reciprocal(self.value))

    def __itruediv__(self, other):
        """"In-place division."""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        self.value *= self._reciprocal(other)
        self.value %= self.field.modulus
        return self

    def __lshift__(self, other):
        """Left shift."""
        if not isinstance(other, (int, np.integer)):
            if not isinstance(other, np.ndarray):
                return NotImplemented
        # TODO: check other.dtype (if array), also for rshift etc.

        return type(self)(self.value << other)

    def __rlshift__(self, other):
        """Left shift (with reflected arguments)."""
        return NotImplemented

    def __ilshift__(self, other):
        """In-place left shift."""
        if not isinstance(other, (int, np.integer)):
            if not isinstance(other, np.ndarray):
                return NotImplemented

        self.value <<= other
        self.value %= self.field.modulus
        return self

    def __rshift__(self, other):
        """Right shift."""
        if not isinstance(other, (int, np.integer)):
            if not isinstance(other, np.ndarray):
                return NotImplemented

        cls = type(self)
        return cls(self.value * cls._reciprocal(1 << other))

    def __rrshift__(self, other):
        """Right shift (with reflected arguments)."""
        return NotImplemented

    def __irshift__(self, other):
        """In-place right shift."""
        if not isinstance(other, (int, np.integer)):
            if not isinstance(other, np.ndarray):
                return NotImplemented

        self.value *= self._reciprocal(1 << other)
        self.value %= self.field.modulus
        return self

    @classmethod
    def _reciprocal(cls, a):
        """Multiplicative inverse."""
        raise NotImplementedError('abstract method')

    def reciprocal(self):
        """Multiplicative inverse."""
        cls = type(self)
        return cls(cls._reciprocal(self.value), check=False)

    @classmethod
    def _sqrt(cls, a, INV=False):
        """Modular (inverse) square root."""
        raise NotImplementedError('abstract method')

    def sqrt(self, INV=False):
        """Modular (inverse) square root."""
        cls = type(self)
        return cls(cls._sqrt(self.value, INV=INV), check=False)

    @classmethod
    def _is_sqr(cls, a):
        """Test for quadratic residuosity (0 is also square)."""
        raise NotImplementedError('abstract method')

    def is_sqr(self):
        """Test for quadratic residuosity (0 is also square)."""
        return self._is_sqr(self.value)

    def reshape(self, *args, **kwargs):
        return type(self)(self.value.reshape(*args, **kwargs), check=False)

    def copy(self, *args, **kwargs):
        return type(self)(self.value.copy(*args, **kwargs), check=False)

    def compress(self, *args, **kwargs):
        return type(self)(self.value.compress(*args, **kwargs), check=False)

    def nonzero(self, *args, **kwargs):
        return self.value.nonzero(*args, **kwargs)

    def flatten(self, *args, **kwargs):
        return type(self)(self.value.flatten(*args, **kwargs), check=False)

    def take(self, *args, **kwargs):
        return type(self)(self.value.take(*args, **kwargs), check=False)

    def tolist(self):
        return np.vectorize(self.field, otypes='O')(self.value).tolist()

    def ravel(self, *args, **kwargs):  # view -- no copy
        return type(self)(self.value.ravel(*args, **kwargs), check=False)

    def repeat(self, *args, **kwargs):
        return type(self)(self.value.repeat(*args, **kwargs), check=False)

    def diagonal(self, *args, **kwargs):  # view -- no copy
        return type(self)(self.value.diagonal(*args, **kwargs), check=False)

    def sum(self, *args, **kwargs):
        a = self.value.sum(*args, **kwargs)
        if not isinstance(a, np.ndarray):
            return self.field(a)

        return type(self)(a, check=True)

    def prod(self, *args, **kwargs):
        v = kwargs.get('initial', 1)
        if not isinstance(v, self.field):
            v = self.field(v)  # prevent growth using field multiplication
        kwargs['initial'] = v
        a = self.value.prod(*args, **kwargs)
        if not isinstance(a, np.ndarray):
            return a

        a = np.vectorize(lambda v: v.value, otypes='O')(a)
        return type(self)(a, check=False)

    def trace(self, *args, **kwargs):
        a = self.value.trace(*args, **kwargs)
        if not isinstance(a, np.ndarray):
            return self.field(a)

        return type(self)(a, check=True)

    # TODO: add atleast1d(a), atleast2d(a), atleast3d(a)

    def transpose(self, *axes):
        a = self.value.transpose(*axes)
        return type(self)(a, check=False)

    def swapaxes(self, axis1, axis2):    # view -- no copy
        a = self.value.swapaxes(axis1, axis2)
        return type(self)(a, check=False)

    @property
    def T(self):
        return self.transpose()


class PrimeFieldArray(FiniteFieldArray):

    _mix_types = int
    if np:
        _mix_types = (int, np.int64, np.int32, np.int16, np.int8,
                      np.uint64, np.uint32, np.uint16, np.uint8)
        # TODO: consider use of np.integer

    @classmethod
    def intarray(cls, a):
        """Extract finite field array as a (signed) integer array."""
        return a.signed_() if cls.field.is_signed else a.unsigned_()

    def __int__(self):
        """Extract (signed) integer value for size-1 arrays only."""
        # TODO: reconsider use of this function
        if self.field.is_signed:
            a = self.signed_()
        else:
            a = self.unsigned_()
        return a.__int__()

    def __abs__(self):
        """Absolute value of (signed) value."""
        return abs(self.signed_())  # TODO: reconsider need for this function

    def signed_(self):
        """Return signed integer representation, symmetric around zero."""
        p = self.field.modulus
        f = np.vectorize(lambda a: a - p if a > p>>1 else a, otypes='O')
        return f(self.value)
        # TODO: check following alternative, issue with np.where result containing floats
        # return self.value - np.where(self.value > p>>1, p, 0)

    def unsigned_(self):
        """Return unsigned integer representation."""
        return self.value.copy()  # NB: copy for consistency with self.signed_()

    @classmethod
    def _pow(cls, a, b):  # NB: exponents assumed to be integer(s)
        """Exponentiation."""
        p = cls.field.modulus
        powmod = gmpy2.powmod
        f = np.vectorize(lambda a, b: int(powmod(a, b, p)), otypes='O')
        return f(a, b)

    @classmethod
    def _reciprocal(cls, a):
        """Multiplicative inverse."""
        p = cls.field.modulus
        invert = gmpy2.invert
        f = np.vectorize(lambda a: int(invert(a, p)), otypes='O')
        return f(a)  # NB: otypes='O' ensures entries of a are Python int (before f is applied)

    @classmethod
    def _sqrt(cls, a, INV=False):
        """Modular (inverse) square roots."""
        p = cls.field.modulus
        if INV and (a == 0).any():
            raise ZeroDivisionError('no inverse sqrt of 0')

        if p == 2:
            return a.copy()  # TODO: check use of copy (here: copy returned in all cases for p)

        if p&3 == 3:
            if INV:
                p4 = (p*3 - 5) >> 2  # a**p4 == a**(-1/2) == 1/sqrt(a) mod p
            else:
                p4 = (p+1) >> 2
            p = gmpy2.mpz(p)
            p4 = gmpy2.mpz(p4)

            W = int(os.getenv('MPYC_MAXWORKERS'))
            if W == 0:
                powmod = gmpy2.powmod
                return np.vectorize(lambda a: int(powmod(a, p4, p)), otypes='O')(a)

            # Experimental use of W worker threads.
            # Using gmpy2's new function powmod_base_list(), which releases the GIL.
            # Example: "python np_lpsolver.py -i6 -M3 -W2" about 1.4x faster than for W=0.
            from gmpy2 import powmod_base_list  # NB: requires gmpy2 >= 2.1.3
            import concurrent.futures
            n = a.size
            s = a.flat
            with concurrent.futures.ThreadPoolExecutor(max_workers=W) as executor:
                tasks = {executor.submit(powmod_base_list, s[i*n//W:(i+1)*n//W], p4, p): i
                         for i in range(W)}
            s = np.empty(n, dtype='O')
            for task in concurrent.futures.as_completed(tasks):
                i = tasks[task]
                s[i*n//W:(i+1)*n//W] = task.result()
            return np.vectorize(int, otypes='O')(s).reshape(a.shape)

        _sqrt = cls.field._sqrt
        return np.vectorize(lambda a: _sqrt(a, INV=INV), otypes='O')(a)

    @classmethod
    def _is_sqr(cls, a):
        p = cls.field.modulus
        if p == 2:
            return np.full(a.shape, True, dtype=bool)

        legendre = gmpy2.legendre
        return np.vectorize(lambda a: legendre(a, p) != -1, otypes=[bool])(a)

    def __repr__(self):
        return f'{self.intarray(self)}'


class ExtensionFieldArray(FiniteFieldArray):

    _mix_types = (int, gfpx.Polynomial)
    if np:
        _mix_types += (np.int64, np.int32, np.int16, np.int8,
                       np.uint64, np.uint32, np.uint16, np.uint8)

    @classmethod
    def _pow(cls, a, b):  # NB: exponents assumed to be integer(s)
        """Exponentiation."""
        modulus = cls.field.modulus
        powmod = type(modulus).powmod
        f = np.vectorize(lambda a, n: powmod(a, n, modulus), otypes='O')
        return f(a, b)

    @classmethod
    def _reciprocal(cls, a):
        """Multiplicative inverse."""
        modulus = cls.field.modulus
        invert = type(modulus).invert
        f = np.vectorize(lambda a: invert(a, modulus), otypes='O')
        return f(a)

    @classmethod
    def _is_sqr(cls, a):
        q = cls.field.order
        if q%2 == 0:
            return np.full(a.shape, True, dtype=bool)

        modulus = cls.field.modulus
        powmod = type(modulus).powmod
        q2 = (q-1) >> 1
        minus_one = [type(modulus).p - 1]
        f = np.vectorize(lambda a: powmod(a, q2, modulus) != minus_one, otypes=[bool])
        return f(a)

    @classmethod
    def _sqrt(cls, a, INV=False):
        """Modular (inverse) square root."""
        q = cls.field.order
        modulus = cls.field.modulus
        poly = type(modulus)
        if INV and (a == poly(0)).any():
            raise ZeroDivisionError('no inverse sqrt of 0')

        if q%2 == 0:
            q2 = q >> 1
            if INV:
                q2 -= 1
            powmod = poly.powmod
            return np.vectorize(lambda a: powmod(a, q2, modulus), otypes='O')(a)

        if q&3 == 3:
            if INV:
                q4 = (q*3 - 5) >> 2  # a**q4 == a**(-1/2) == 1/sqrt(a) in GF(q)
            else:
                q4 = (q+1) >> 2
            powmod = poly.powmod
            return np.vectorize(lambda a: powmod(a, q4, modulus), otypes='O')(a)

        _sqrt = cls.field._sqrt
        return np.vectorize(lambda a: _sqrt(a, INV=INV), otypes='O')(a)

    def __repr__(self):
        return f'{self.value}'


class BinaryFieldArray(ExtensionFieldArray):

    _mix_types = (int, gfpx.BinaryPolynomial)
    if np:
        _mix_types += (np.int64, np.int32, np.int16, np.int8,
                       np.uint64, np.uint32, np.uint16, np.uint8)

    @classmethod
    def _is_sqr(cls, a):
        return np.full(a.shape, True, dtype=bool)

    @classmethod
    def _sqrt(cls, a, INV=False):
        modulus = cls.field.modulus
        poly = type(modulus)
        if INV and (a == poly(0)).any():
            raise ZeroDivisionError('no inverse sqrt of 0')

        powmod = poly.powmod
        q2 = cls.field.order >> 1
        if INV:
            q2 -= 1
        return np.vectorize(lambda a: powmod(a, q2, modulus), otypes='O')(a)

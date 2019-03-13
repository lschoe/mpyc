"""The MPyC runtime module is used to execute secure multiparty computations.

Parties perform computations on secret-shared values by exchanging messages.
Shamir's threshold secret sharing scheme is used for fields of prime order and
fields of characteristic two. MPyC provides secure number types and operations,
many of which are available through Python's mechanism for operator overloading.
"""

import os
import sys
import time
import logging
import math
import secrets
import itertools
import functools
import configparser
import argparse
import asyncio
import ssl
from mpyc import thresha
from mpyc import sectypes
from mpyc import asyncoro
import mpyc.random

Future = asyncio.Future
Share = sectypes.Share
SecureFiniteField = sectypes.SecureFiniteField
gather_shares = asyncoro.gather_shares
mpc_coro = asyncoro.mpc_coro
mpc_coro_no_pc = asyncoro._mpc_coro_no_pc
returnType = asyncoro.returnType


class Runtime:
    """MPyC runtime secure against passive attacks.

    The runtime maintains basic information such as a program counter,
    the list of parties, etc., and handles secret-shared values of type Share.

    1-party case is supported (with option to disable asynchronous evaluation).
    Threshold 0 (no corrupted parties) is supported for m-party case as well
    to enable distributed computation (without secret sharing).
    """

    version = mpyc.__version__
    random = mpyc.random

    def __init__(self, pid, parties, options):
        """Initialize runtime."""
        self.pid = pid
        self.parties = parties
        self.options = options
        self.threshold = options.threshold
        self._logging_enabled = not options.no_log
        self._program_counter = (0,)
        self._loop = asyncio.get_event_loop()  # cache running loop
        self.start_time = None

    @property
    def threshold(self):
        """Threshold for MPC."""
        return self._threshold

    @threshold.setter
    def threshold(self, t):
        self._threshold = t
        # generate new PRSS keys
        self.prfs.cache_clear()
        keys = {}
        m = len(self.parties)
        for t in range((m + 1) // 2):  # all possible thresholds
            for subset in itertools.combinations(range(m), m - t):
                if self.pid == min(subset):
                    keys[subset] = secrets.token_bytes(16)  # 128-bit key
        self._prss_keys = keys
        # caching (m choose t):
        self._bincoef = math.factorial(m) // math.factorial(t) // math.factorial(m - t)

    @functools.lru_cache(maxsize=None)
    def prfs(self, bound):
        """PRFs with codomain range(bound) for pseudorandom secret sharing.

        Return a mapping from sets of parties to PRFs.
        """
        f = {}
        for subset, key in self._prss_keys.items():
            if len(subset) == len(self.parties) - self.threshold:
                f[subset] = thresha.PRF(key, bound)
        return f

    def _increment_pc(self):
        """Increment the program counter."""
        pc = self._program_counter
        self._program_counter = (pc[0] + 1,) + pc[1:]

    def _send_shares(self, peer_pid, data):
        self.parties[peer_pid].protocol.send_data(self._program_counter, data)

    def _receive_shares(self, peer_pid):
        pc = self._program_counter
        if pc in self.parties[peer_pid].protocol.buffers:
            # Data already received from peer.
            data = self.parties[peer_pid].protocol.buffers.pop(pc)
        else:
            # Data not yet received from peer.
            data = self.parties[peer_pid].protocol.buffers[pc] = Future(loop=self._loop)
        return data

    def _exchange_shares(self, in_shares):
        out_shares = [None] * len(in_shares)
        for peer_pid, data in enumerate(in_shares):
            if peer_pid == self.pid:
                d = data
            else:
                self._send_shares(peer_pid, data)
                d = self._receive_shares(peer_pid)
            out_shares[peer_pid] = d
        return out_shares

    async def barrier(self):
        """Barrier for runtime."""
        if self.options.no_barrier:
            return

        logging.info(f'Barrier {asyncoro.pc_level} '
                     f'{len(self._program_counter)} '
                     f'{list(reversed(self._program_counter))}'
                     )
        if not self.options.no_async:
            while asyncoro.pc_level >= len(self._program_counter):
                await asyncio.sleep(0)

    def run(self, f):
        """Run the given coroutine or future until it is done."""
        if self._loop.is_running():
            if not asyncio.iscoroutine(f):
                async def _wrap(fut):
                    return await fut
                f = _wrap(f)
            while True:
                try:
                    f.send(None)
                except StopIteration as exc:
                    return exc.value

        return self._loop.run_until_complete(f)

    def logging(self, enable=None):
        """Toggle/enable/disable logging."""
        if enable is None:
            self._logging_enabled = not self._logging_enabled
        else:
            self._logging_enabled = enable
        if self._logging_enabled:
            logging.disable(logging.NOTSET)
        else:
            logging.disable(logging.INFO)

    async def start(self):
        """Start the MPyC runtime.

        Open connections with other parties, if any.
        """
        logging.info(f'Start MPyC runtime v{self.version}')
        self.start_time = time.time()
        m = len(self.parties)
        if m == 1:
            return

        # m > 1
        for peer in self.parties:
            peer.protocol = Future(loop=self._loop) if peer.pid == self.pid else None
        if self.options.ssl:
            crtfile = os.path.join('.config', f'party_{self.pid}.crt')
            keyfile = os.path.join('.config', f'party_{self.pid}.key')
            cafile = os.path.join('.config', 'mpyc_ca.crt')
        loop = asyncio.get_event_loop()

        # Listen for all parties < self.pid.
        if self.pid:
            listen_port = self.parties[self.pid].port
            if self.options.ssl:
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                context.load_cert_chain(crtfile, keyfile=keyfile)
                context.load_verify_locations(cafile=cafile)
                context.verify_mode = ssl.CERT_REQUIRED
            else:
                context = None
            factory = asyncoro.SharesExchanger
            server = await loop.create_server(factory, port=listen_port, ssl=context)
            logging.debug(f'Listening on port {listen_port}')

        # Connect to all parties > self.pid.
        for peer in self.parties[self.pid + 1:]:
            logging.debug(f'Connecting to {peer}')
            while True:
                try:
                    if self.options.ssl:
                        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                        context.load_cert_chain(crtfile, keyfile=keyfile)
                        context.load_verify_locations(cafile=cafile)
                        server_hostname = f'MPyC party {peer.pid}'
                    else:
                        context = None
                        server_hostname = None
                    factory = lambda: asyncoro.SharesExchanger(peer.pid)
                    await loop.create_connection(factory, peer.host, peer.port, ssl=context,
                                                 server_hostname=server_hostname)
                    break
                except Exception as exc:
                    logging.debug(exc)
                time.sleep(0.1)

        await self.parties[self.pid].protocol
        if self.options.ssl:
            logging.info(f'All {m} parties connected via SSL.')
        else:
            logging.info(f'All {m} parties connected.')
        if self.pid:
            server.close()

    async def shutdown(self):
        """Shutdown the MPC runtime.

        Close all connections, if any.
        """
        m = len(self.parties)
        if m > 1:
            # Wait for all parties after a barrier.
            while asyncoro.pc_level >= len(self._program_counter):
                await asyncio.sleep(0)
            await self.output(self.input(sectypes.SecFld(101)(self.pid)), threshold=m-1)
            # Close connections to all parties.
            for peer in self.parties:
                if peer.pid != self.pid:
                    peer.protocol.close_connection()

        elapsed = time.time() - self.start_time
        from datetime import timedelta
        logging.info(f'Stop MPyC runtime -- elapsed time: {timedelta(seconds=elapsed)}')

    SecFld = staticmethod(sectypes.SecFld)
    SecInt = staticmethod(sectypes.SecInt)
    SecFxp = staticmethod(sectypes.SecFxp)
    gather = staticmethod(gather_shares)
    coroutine = staticmethod(mpc_coro)
    returnType = staticmethod(returnType)

    def input(self, x, senders=None):
        """Input x to the computation.

        Value x is a secure number, or a list of secure numbers.
        The senders are the parties that provide an input.
        The default is to let every party be a sender.
        """
        x_is_list = isinstance(x, list)
        if x_is_list:
            x = x[:]
        else:
            x = [x]
        if senders is None:
            m = len(self.parties)
            senders = list(range(m))
        senders_is_list = isinstance(senders, list)
        if not senders_is_list:
            senders = [senders]
        y = self._distribute(x, senders)
        if not senders_is_list:
            y = y[0]
            if not x_is_list:
                y = y[0]
        else:
            if not x_is_list:
                y = [a[0] for a in y]
        return y

    @mpc_coro
    async def _distribute(self, x, senders):
        """Distribute shares for each x provided by a sender."""
        stype = type(x[0])  # all elts assumed of same type
        field = stype.field
        await returnType(stype, len(senders), len(x))
        value = x[0].df if not isinstance(x[0].df, Future) else None
        assert value is None or self.pid in senders
        m = len(self.parties)
        t = self.threshold
        x = [a.df for a in x]  # Extract values from all elements of x.
        shares = [None] * len(senders)
        for i, peer_pid in enumerate(senders):
            if peer_pid == self.pid:
                in_shares = thresha.random_split(x, t, m)
                for other_pid, data in enumerate(in_shares):
                    data = field.to_bytes(data)
                    if other_pid == self.pid:
                        shares[i] = data
                    else:
                        self._send_shares(other_pid, data)
            else:
                shares[i] = self._receive_shares(peer_pid)
        shares = await gather_shares(shares)
        return [[field(a) for a in field.from_bytes(r)] for r in shares]

    async def output(self, x, receivers=None, threshold=None):
        """Output the value of x to the receivers specified.

        Value x is a secure number, or a list of secure numbers.
        The receivers are the parties that will obtain the output.
        The default is to let every party be a receiver.
        """
        x_is_list = isinstance(x, list)
        if x_is_list:
            x = x[:]
        else:
            x = [x]
        if receivers is None:
            m = len(self.parties)
            receivers = list(range(m))
        elif isinstance(receivers, int):
            receivers = [receivers]
        if threshold is None:
            threshold = self.threshold
        y = self._recombine(x, receivers, threshold)
        if not x_is_list:
            y = y[0]
        return await gather_shares(y)

    @mpc_coro
    async def _recombine(self, x, receivers, t):
        """Recombine shares of elements of x."""
        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, Share):
            field = sftype.field
            if not field.frac_length:
                await returnType(sftype, len(x))
            else:
                await returnType((sftype, x[0].integral), len(x))
            x = await gather_shares(x)
        else:
            field = sftype
            await returnType(Future, len(x))

        m = len(self.parties)
        x = [a.value for a in x]
        # Send share to all successors in receivers.
        for peer_pid in receivers:
            if 0 < (peer_pid - self.pid) % m <= t:
                self._send_shares(peer_pid, field.to_bytes(x))
        # Receive and recombine shares if this party is a receiver.
        if self.pid in receivers:
            shares = [None] * t
            for i in range(t):
                shares[i] = self._receive_shares((self.pid - t + i) % m)
            shares = await gather_shares(shares)
            points = [((self.pid - t + j) % m + 1, field.from_bytes(shares[j])) for j in range(t)]
            points.append((self.pid + 1, x))
            return thresha.recombine(field, points)

        return [None] * len(x)

    @mpc_coro
    async def _reshare(self, x):
        x_is_list = isinstance(x, list)
        if not x_is_list:
            x = [x]
        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, Share):
            field = sftype.field
            if not field.frac_length:
                await returnType(sftype, len(x))
            else:
                await returnType((sftype, x[0].integral), len(x))
            x = await gather_shares(x)
        else:
            field = sftype
            await returnType(Future)

        m = len(self.parties)
        t = self.threshold
        in_shares = thresha.random_split(x, t, m)
        in_shares = [field.to_bytes(elts) for elts in in_shares]
        # Recombine the first 2t+1 output_shares.
        out_shares = await gather_shares(self._exchange_shares(in_shares)[:2 * t + 1])
        points = [(j + 1, field.from_bytes(out_shares[j])) for j in range(len(out_shares))]
        y = thresha.recombine(field, points)

        if issubclass(sftype, Share):
            y = [sftype(s) for s in y]
        if not x_is_list:
            y = y[0]
        return y

    @mpc_coro
    async def trunc(self, x, f=None, l=None):
        """Secure truncation of f least significant bits of (elements of) x.

        Probabilistic rounding of a / 2**f for a in x.
        """
        x_is_list = isinstance(x, list)
        if not x_is_list:
            x = [x]
        n = len(x)
        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, Share):
            if x_is_list:
                await returnType(sftype, n)
            else:
                await returnType(sftype)
            Zp = sftype.field
            l = sftype.bit_length
        else:
            await returnType(Future)
            Zp = sftype
        if f is None:
            f = Zp.frac_length
        k = self.options.sec_param
        r_bits = await self.random_bits(Zp, f * n)
        r_modf = [None] * n
        for j in range(n):
            s = 0
            for i in range(f - 1, -1, -1):
                s <<= 1
                s += r_bits[f*j + i].value
            r_modf[j] = Zp(s)
        r_divf = self._randoms(Zp, n, 1<<(k + l - f))
        if issubclass(sftype, Share):
            x = await gather_shares(x)
        c = await self.output([a + ((1<<f) + (q.value << f) + r.value)
                              for a, q, r in zip(x, r_divf, r_modf)])
        c = [c.value % (1<<f) for c in c]
        y = [(a - c + r.value) >> f for a, c, r in zip(x, c, r_modf)]
        if not x_is_list:
            y = y[0]
        return y

    def eq_public(self, a, b):
        """Secure public equality test of a and b."""
        return self.is_zero_public(a - b)

    @mpc_coro
    async def is_zero_public(self, a) -> Future:
        """Secure public zero test of a."""
        stype = type(a)
        field = stype.field
        m = len(self.parties)
        t = self.threshold
        if issubclass(stype, SecureFiniteField):
            prfs = self.prfs(field.order)
            while True:
                r, s = self._randoms(field, 2)
                z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), 1)
                if await self.output(r * s + z[0], threshold=2 * t):
                    break
        else:
            r = self._random(field)  # NB: failure r=0 with probability 1/p
        a = await gather_shares(a)
        if issubclass(stype, SecureFiniteField):
            z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), 1)
            b = a * r + z[0]
        else:
            b = a * r
        c = await self.output(b, threshold=2 * t)
        return c == 0

    @mpc_coro_no_pc
    async def neg(self, a):
        """Secure negation (additive inverse) of a."""
        stype = type(a)
        if not stype.field.frac_length:
            await returnType(stype)
        else:
            await returnType((stype, a.integral))
        a = await gather_shares(a)
        return -a

    @mpc_coro_no_pc
    async def add(self, a, b):
        """Secure addition of a and b."""
        stype = type(a)
        if not stype.field.frac_length:
            await returnType(stype)
        else:
            await returnType((stype, a.integral and b.integral))
        a, b = await gather_shares(a, b)
        return a + b

    @mpc_coro_no_pc
    async def sub(self, a, b):
        """Secure subtraction of a and b."""
        stype = type(a)
        if not stype.field.frac_length:
            await returnType(stype)
        else:
            await returnType((stype, a.integral and b.integral))
        a, b = await gather_shares(a, b)
        return a - b

    @mpc_coro
    async def mul(self, a, b):
        """Secure multiplication of a and b."""
        stype = type(a)
        field = stype.field
        f = field.frac_length
        if not f:
            await returnType(stype)
        else:
            a_integral = a.integral
            b_integral = isinstance(b, int) or isinstance(b, Share) and b.integral
            if isinstance(b, float):
                b = round(b * 2**f)
            await returnType((stype, a_integral and b_integral))

        shb = isinstance(b, Share)
        if not shb:
            a = await gather_shares(a)
        elif a is b:
            a = b = await gather_shares(a)
        else:
            a, b = await gather_shares(a, b)
        if f and b_integral:
            a, b = b, a
        if f and (a_integral or b_integral) and not isinstance(a, int):
            a = a >> f  # NB: no inplace a >>=
        c = a * b
        if shb:
            c = self._reshare(c)
        if f and not (a_integral or b_integral):
            c = self.trunc(stype(c))
        return c

    def div(self, a, b):
        """Secure division of a by b, for nonzero b."""
        if isinstance(b, Share):
            if type(b).field.frac_length:
                c = self._rec(b)
            else:
                c = self.reciprocal(b)
            return self.mul(c, a)

        # isinstance(a, Share) ensured
        if type(a).field.frac_length:
            if isinstance(b, (int, float)):
                c = 1 / b
                if c.is_integer():
                    c = round(c)
            else:
                c = b.reciprocal() << type(a).field.frac_length
        else:
            if not isinstance(b, a.field):
                b = a.field(b)
            c = b.reciprocal()
        return self.mul(a, c)

    @mpc_coro
    async def reciprocal(self, a):
        """Secure reciprocal (multiplicative inverse) of a, for nonzero a."""
        stype = type(a)
        field = stype.field
        await returnType(stype)
        a = await gather_shares(a)
        while True:
            r = self._random(field)
            ar = await self.output(a * r, threshold=2*self.threshold)
            if ar:
                break
        r <<= field.frac_length
        return r / ar

    def pow(self, a, b):
        """Secure exponentation a raised to the power of b, for public integer b."""
        if b == 254:  # addition chain for AES S-Box (11 multiplications in 9 rounds)
            d = a
            c = self.mul(d, d)
            c = self.mul(c, c)
            c = self.mul(c, c)
            c = self.mul(c, d)
            c = self.mul(c, c)
            c, d = self.scalar_mul(c, [c, d])
            c, d = self.scalar_mul(c, [c, d])
            c = self.mul(c, d)
            c = self.mul(c, c)
            return c

        if b == 0:
            return type(a)(1)

        if b < 0:
            a = self.reciprocal(a)
            b = -b
        d = a
        c = 1
        for i in range(b.bit_length() - 1):
            # d = a ** (1 << i) holds
            if (b >> i) & 1:
                c = c * d
            d = d * d
        c = c * d
        return c

    def and_(self, a, b):
        """Secure bitwise and of a and b."""
        return self.from_bits(self.schur_prod(self.to_bits(a), self.to_bits(b)))

    def xor(self, a, b):
        """Secure bitwise xor of a and b."""
        return a + b

    def invert(self, a):
        """Secure bitwise inverse (not) of a."""
        return a + type(a)(a.field.order - 1)

    def or_(self, a, b):
        """Secure bitwise or of a and b."""
        return a + b + self.and_(a, b)

    def eq(self, a, b):
        """Secure comparison a == b."""
        return self.is_zero(a - b)

    def ge(self, a, b):
        """Secure comparison a >= b."""
        return self.sgn(a - b, GE=True)

    def is_zero(self, a):
        """Secure zero test a == 0."""
        if isinstance(a, SecureFiniteField):
            return 1 - self.pow(a, a.field.order - 1)

        if (a.bit_length / 2 > self.options.sec_param >= 8 and a.field.order % 4 == 3):
            return self._is_zero(a)

        return self.sgn(a, EQ=True)

    @mpc_coro
    async def _is_zero(self, a):
        """Probabilistic zero test."""
        stype = type(a)
        await returnType((stype, True))
        Zp = stype.field

        k = self.options.sec_param
        z = self.random_bits(Zp, k)
        u = self._randoms(Zp, k)
        u2 = self.schur_prod(u, u)
        a, u2, z = await gather_shares(a, u2, z)
        r = self._randoms(Zp, k)
        c = [Zp(a.value * r[i].value + (1 - (z[i].value << 1)) * u2[i].value) for i in range(k)]
        # -1 is nonsquare for Blum p, u_i !=0 w.v.h.p.
        # If a == 0, c_i is square mod p iff z[i] == 0.
        # If a != 0, c_i is square mod p independent of z[i].
        c = await self.output(c, threshold=2*self.threshold)
        for i in range(k):
            if c[i] == 0:
                c[i] = Zp(1)
            else:
                c[i] = 1 - z[i] if c[i].is_sqr() else z[i]
        e = await self.prod(c)
        e <<= Zp.frac_length
        return e

    @mpc_coro
    async def sgn(self, a, EQ=False, GE=False):
        """Secure sign(um) of a, -1 if a < 0 else 0 if a == 0 else 1."""
        stype = type(a)
        await returnType((stype, True))
        Zp = stype.field

        l = stype.bit_length
        r_bits = await self.random_bits(Zp, l)
        r_modl = 0
        for r_i in reversed(r_bits):
            r_modl <<= 1
            r_modl += r_i.value
        a = await gather_shares(a)
        a_rmodl = a + ((1<<l) + r_modl)
        k = self.options.sec_param
        r_divl = self._random(Zp, 1<<k)
        c = await self.output(a_rmodl + (r_divl.value << l))
        c = c.value % (1<<l)

        if not EQ:  # a la Toft
            s_sign = (await self.random_bits(Zp, 1, signed=True))[0].value
            e = [None] * (l + 1)
            sumXors = 0
            for i in range(l - 1, -1, -1):
                c_i = (c >> i) & 1
                r_i = r_bits[i].value
                e[i] = Zp(s_sign + r_i - c_i + 3 * sumXors)
                sumXors += 1 - r_i if c_i else r_i
            e[l] = Zp(s_sign - 1 + 3 * sumXors)
            e = await self.prod(e)
            g = await self.is_zero_public(stype(e))
            UF = 1 - s_sign if g else s_sign + 1
            z = (a_rmodl - (c + (UF << l - 1))) / (1<<l)

        if not GE:
            h = self.prod([r_bits[i] if (c >> i) & 1 else 1 - r_bits[i] for i in range(l)])
            h = await h
            if EQ:
                z = h
            else:
                z = (1 - h) * (2 * z - 1)
                z = await self._reshare(z)

        z <<= Zp.frac_length
        return z

    def min(self, *x):
        """Secure minimum of all given elements in x."""
        if len(x) == 1:
            x = x[0]
        n = len(x)
        if n == 1:
            return x[0]

        m0 = self.min(x[:n // 2])
        m1 = self.min(x[n // 2:])
        d = m0 - m1
        return m1 + (d <= 0) * d

    def max(self, *x):
        """Secure maximum of all given elements in x."""
        if len(x) == 1:
            x = x[0]
        n = len(x)
        if n == 1:
            return x[0]

        m0 = self.max(x[:n // 2])
        m1 = self.max(x[n // 2:])
        d = m1 - m0
        return m0 + (d >= 0) * d

    @mpc_coro
    async def lsb(self, a):
        """Secure least significant bit of a."""  # a la [ST06]
        stype = type(a)
        await returnType((stype, True))
        Zp = stype.field
        l = stype.bit_length
        k = self.options.sec_param
        b = self.random_bit(stype)
        a, b = await gather_shares(a, b)
        b >>= Zp.frac_length
        r = self._random(Zp, 1 << (l + k - 1))
        c = await self.output(a + ((1<<l) + (r.value << 1) + b.value))
        x = 1 - b if c.value & 1 else b  # xor
        x <<= Zp.frac_length
        return x

    def mod(self, a, b):
        """Secure modulo reduction."""
        if b == 2:
            r = self.lsb(a)
        elif not b & (b - 1):
            r = self.from_bits(self.to_bits(a, b.bit_length() - 1))
        else:
            r = self._mod(a, b)
        return r

    @mpc_coro
    async def _mod(self, a, b):
        """Secure modulo reduction, for public b."""  # a la [GMS10]
        stype = type(a)
        await returnType(stype)
        Zp = stype.field
        l = stype.bit_length
        k = self.options.sec_param
        f = Zp.frac_length
        r_bits = self.random._randbelow(stype, b, bits=True)
        a, r_bits = await gather_shares(a, r_bits)
        r_bits = [(r >> f).value for r in r_bits]
        r_modb = 0
        for r_i in reversed(r_bits):
            r_modb <<= 1
            r_modb += r_i
        r_modb = Zp(r_modb)
        r_divb = self._random(Zp, 1 << k)
        c = await self.output(a + ((1<<l) - ((1<<l) % b) + b * r_divb.value - r_modb.value))
        c = c.value % b
        c_bits = [(c >> i) & 1 for i in range(len(r_bits))]
        c_bits.append(0)
        r_bits = [stype(r) for r in r_bits]
        r_bits.append(stype(0))
        z = stype(r_modb - (b - c)) >= 0  # TODO: avoid full comparison (use r_bits)
        return (self.from_bits(self.add_bits(r_bits, c_bits)) - z * b) * 2**-f

    @mpc_coro_no_pc
    async def sum(self, x):
        """Secure sum of all elements in x."""
        x = x[:]
        field = x[0].field
        await returnType(type(x[0]))
        x = await gather_shares(x)
        s = 0
        for i in range(len(x)):
            s += x[i].value
        return field(s)

    @mpc_coro_no_pc
    async def lin_comb(self, a, x):
        """Secure linear combination: dot product of public a and secret x."""
        # TODO: merge with in_prod()
        x = x[:]
        field = x[0].field
        await returnType(type(x[0]))
        x = await gather_shares(x)
        s = 0
        for i in range(len(x)):
            s += a[i].value * x[i].value
        return field(s)

    @mpc_coro
    async def in_prod(self, x, y):
        """Secure dot product of x and y (one resharing)."""
        if x is y:
            x = x[:]
            y = x
        else:
            x, y = x[:], y[:]
        stype = type(x[0])
        field = stype.field
        f = field.frac_length
        if not f:
            await returnType(stype)
        else:
            x_integral = x[0].integral
            await returnType((stype, x_integral and y[0].integral))
        if x is y:
            x = y = await gather_shares(x)
        else:
            x, y = await gather_shares(x, y)
        s = 0
        for i in range(len(x)):
            s += x[i].value * y[i].value
        s = field(s)
        if f and x_integral:
            s >>= f
        s = self._reshare(s)
        if f and not x_integral:
            s = self.trunc(stype(s))
        return s

    @mpc_coro
    async def prod(self, x):
        """Secure product of all elements in x (in log_2 len(x) rounds)."""
        x = x[:]
        if isinstance(x[0], Share):
            await returnType(type(x[0]))
            x = await gather_shares(x)
        else:
            await returnType(Future)

        while len(x) > 1:
            h = [None] * (len(x)//2)
            for i in range(len(h)):
                h[i] = x[2*i] * x[2*i+1]
            h = await self._reshare(h)  # TODO: handle trunc
            x = x[2*len(h):] + h
        return x[0]

    @mpc_coro_no_pc
    async def vector_add(self, x, y):
        """Secure addition of vectors x and y."""
        x, y = x[:], y[:]
        stype = type(x[0])
        if not stype.field.frac_length:
            await returnType(stype, len(x))
        else:
            await returnType((stype, x[0].integral and y[0].integral), len(x))
        x, y = await gather_shares(x, y)
        for i in range(len(x)):
            x[i] = x[i] + y[i]
        return x

    @mpc_coro_no_pc
    async def vector_sub(self, x, y):
        """Secure subtraction of vectors x and y."""
        x, y = x[:], y[:]
        stype = type(x[0])
        if not stype.field.frac_length:
            await returnType(stype, len(x))
        else:
            await returnType((stype, x[0].integral and y[0].integral), len(x))
        x, y = await gather_shares(x, y)
        for i in range(len(x)):
            x[i] = x[i] - y[i]
        return x

    @mpc_coro_no_pc
    async def matrix_add(self, A, B, tr=False):
        """Secure addition of matrices A and (transposed) B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        await returnType(type(A[0][0]), len(A), len(A[0]))
        A, B = await gather_shares(A, B)
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = A[i][j] + (B[j][i] if tr else B[i][j])
        return A

    @mpc_coro_no_pc
    async def matrix_sub(self, A, B, tr=False):
        """Secure subtraction of matrices A and (transposed) B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        await returnType(type(A[0][0]), len(A), len(A[0]))
        A, B = await gather_shares(A, B)
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = A[i][j] - (B[j][i] if tr else B[i][j])
        return A

    @mpc_coro
    async def scalar_mul(self, a, x):
        """Secure scalar multiplication of scalar a with vector x."""
        x = x[:]
        stype = type(a)
        field = stype.field
        f = field.frac_length
        if not f:
            await returnType(stype, len(x))
        else:
            a_integral = a.integral
            await returnType((stype, a_integral and x[0].integral), len(x))

        a, x = await gather_shares(a, x)
        if f and a_integral:
            a = a >> f  # NB: no inplace a >>=
        for i in range(len(x)):
            x[i] = x[i] * a
        x = await self._reshare(x)
        if f and not a_integral:
            x = self.trunc(x, l=stype.bit_length)
            x = await gather_shares(x)
        return x

    @mpc_coro
    async def _if_else_list(self, a, x, y):
        x, y = x[:], y[:]
        stype = type(a)
        field = stype.field
        f = field.frac_length
        if not f:
            await returnType(stype, len(x))
        else:
            a_integral = a.integral
            if not a_integral:
                raise ValueError('condition must be integral')
            await returnType((stype, a_integral and x[0].integral and y[0].integral), len(x))

        a, x, y = await gather_shares(a, x, y)
        if f:
            a = a >> f  # NB: no inplace a >>=
        for i in range(len(x)):
            x[i] = field(a.value * (x[i].value - y[i].value) + y[i].value)
        x = await self._reshare(x)
        return x

    def if_else(self, c, x, y):
        '''Secure selection based on condition c between x and y.'''
        if isinstance(x, list):
            z = self._if_else_list(c, x, y)
        else:
            z = c * (x - y) + y
        return z

    @mpc_coro
    async def schur_prod(self, x, y):
        """Secure entrywise multiplication of vectors x and y."""
        if x is y:
            x = x[:]
            y = x
        else:
            x, y = x[:], y[:]
        if isinstance(x[0], Share):
            stype = type(x[0])
            await returnType(stype, len(x))
            if x is y:
                x = y = await gather_shares(x)
            else:
                x, y = await gather_shares(x, y)
            truncy = stype.field.frac_length
        else:
            await returnType(Future)
            truncy = False
        for i in range(len(x)):
            x[i] = x[i] * y[i]
        x = await self._reshare(x)
        if truncy:
            x = self.trunc(x, l=stype.bit_length)
            x = await gather_shares(x)
        return x

    @mpc_coro
    async def matrix_prod(self, A, B, tr=False):
        """Secure matrix product of A with (transposed) B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        shA = isinstance(A[0][0], Share)
        shB = isinstance(B[0][0], Share)
        stype = type(A[0][0]) if shA else type(B[0][0])
        field = stype.field
        n = len(B) if tr else len(B[0])
        await returnType(stype, len(A), n)
        if shA and shB:
            A, B = await gather_shares(A, B)
        elif shA:
            A = await gather_shares(A)
        else:
            B = await gather_shares(B)

        C = [None] * len(A)
        for ia in range(len(A)):
            C[ia] = [None] * n
            for ib in range(n):
                s = 0
                for i in range(len(A[0])):
                    s += A[ia][i].value * (B[ib][i] if tr else B[i][ib]).value
                C[ia][ib] = field(s)
            if shA and shB:
                C[ia] = self._reshare(C[ia])
        if shA and shB:
            C = await gather_shares(C)
        if field.frac_length:
            l = stype.bit_length
            C = [self.trunc(c, l=l) for c in C]
            C = await gather_shares(C)
        return C

    @mpc_coro
    async def gauss(self, A, d, b, c):
        """Secure Gaussian elimination A d - b c."""
        A, b, c = [r[:] for r in A], b[:], c[:]
        stype = type(A[0][0])
        field = stype.field
        n = len(A[0])
        await returnType(stype, len(A), n)
        A, d, b, c = await gather_shares(A, d, b, c)
        d = d.value
        for i in range(len(A)):
            b[i] = b[i].value
            for j in range(n):
                A[i][j] = field(A[i][j].value * d - b[i] * c[j].value)
            A[i] = self._reshare(A[i])
        A = await gather_shares(A)
        if field.frac_length:
            l = stype.bit_length
            A = [self.trunc(a, l=l) for a in A]
            A = await gather_shares(A)
        return A

    def _prss_uci(self):
        """Create unique common input for PRSS.

        Increments the program counter to ensure that consecutive calls
        to PRSS-related methods will use unique program counters.
        """
        self._increment_pc()
        return self._program_counter

    def _random(self, sftype, bound=None):
        """Secure random value of the given type in the given range."""
        return self._randoms(sftype, 1, bound)[0]

    def _randoms(self, sftype, n, bound=None):
        """n secure random values of the given type in the given range."""
        if issubclass(sftype, Share):
            field = sftype.field
        else:
            field = sftype
        if bound is None:
            bound = field.order
        else:
            bound = (bound - 1) // self._bincoef + 1  # TODO: round to power of 2
        m = len(self.parties)
        prfs = self.prfs(bound)
        shares = thresha.pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
        if issubclass(sftype, Share):
            shares = [sftype(s) for s in shares]
        return shares

    def random_bit(self, stype, signed=False):
        """Secure uniformly random bit of the given type."""
        return self.random_bits(stype, 1, signed)[0]

    @mpc_coro
    async def random_bits(self, sftype, n, signed=False):
        """n secure uniformly random bits of the given type."""
        prss0 = False
        f0 = 0
        if issubclass(sftype, Share):
            await returnType((sftype, True), n)
            field = sftype.field
            if issubclass(sftype, SecureFiniteField):
                prss0 = True
            f0 = field.frac_length
        else:
            await returnType(Future)
            field = sftype

        m = len(self.parties)

        if not isinstance(field.modulus, int):
            prfs = self.prfs(2)
            bits = thresha.pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
            return bits

        bits = [None] * n
        p = field.modulus
        if not signed:
            q = (p + 1) >> 1  # q = 1/2 mod p
        prfs = self.prfs(p)
        t = self.threshold
        h = n
        while h > 0:
            rs = thresha.pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), h)
            # Compute and open the squares and compute square roots.
            r2s = [r * r for r in rs]
            if prss0:
                z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), h)
                for i in range(h):
                    r2s[i] += z[i]
            r2s = await self.output(r2s, threshold=2 * t)
            for r, r2 in zip(rs, r2s):
                if r2.value != 0:
                    h -= 1
                    s = r.value * r2.sqrt(INV=True).value
                    if not signed:
                        s += 1
                        s %= p
                        s *= q
                    bits[h] = field(s << f0)
        return bits

    def add_bits(self, x, y):
        """Secure binary addition of bit vectors x and y."""
        x, y = x[:], y[:]

        def f(i, j, high=False):
            n = j - i
            if n == 1:
                c[i] = x[i] * y[i]
                if high:
                    d[i] = x[i] + y[i] - c[i] * 2
            else:
                h = i + (n // 2)
                f(i, h, high=high)
                f(h, j, high=True)
                c[h:j] = self.vector_add(c[h:j], self.scalar_mul(c[h - 1], d[h:j]))
                if high:
                    d[h:j] = self.scalar_mul(d[h - 1], d[h:j])
        n = len(x)
        c = [None] * n
        if n >= 1:
            d = [None] * n
            f(0, n)
        # c = prefix carries for addition of x and y
        for i in range(n - 1, -1, -1):
            c[i] = x[i] + y[i] - c[i] * 2 + (c[i - 1] if i > 0 else 0)
        return c

    @mpc_coro
    async def to_bits(self, a, l=None):
        """Secure extraction of l (or all) least significant bits of a."""  # a la [ST06].
        stype = type(a)
        if l is None:
            l = stype.bit_length
        assert l <= stype.bit_length, 'extracting too many bits'
        await returnType((stype, True), l)
        field = stype.field

        r_bits = await self.random_bits(field, l)
        r_modl = 0
        for r_i in reversed(r_bits):
            r_modl <<= 1
            r_modl += r_i.value
        if isinstance(field.modulus, int):
            k = self.options.sec_param
            r_divl = self._random(field, 1<<k)
            a = await gather_shares(a)
            c = await self.output(a + ((1<<l) + (r_divl.value << l) - r_modl))
            c = c.value % (1<<l)
            c_bits = [(c >> i) & 1 for i in range(l)]
            r_bits = [stype(r.value) for r in r_bits]
            return self.add_bits(r_bits, c_bits)

        a = await gather_shares(a)
        c = await self.output(a + r_modl)
        c = int(c.value)
        return [r_bits[i] + ((c >> i) & 1) for i in range(l)]

    @mpc_coro_no_pc
    async def from_bits(self, x):
        """Recover secure number from its binary representation x."""
        # TODO: also handle negative numbers with sign bit (NB: from_bits() in random.py)
        x = x[:]
        stype = type(x[0])
        await returnType((stype, True))
        x = await gather_shares(x)
        s = 0
        for a in reversed(x):
            s <<= 1
            s += a.value
        return stype.field(s)

    def _norm(self, a):  # signed normalization factor
        x = self.to_bits(a)  # low to high bits
        b = x[-1]  # sign bit
        s = 1 - b * 2  # sign s = (-1)^b
        x = x[:-1]
        _1 = type(a)(1)

        def __norm(x):
            n = len(x)
            if n == 1:
                t = s * x[0] + b  # self.xor(b, x[0])
                return 2 - t, t

            i0, nz0 = __norm(x[:n//2])  # low bits
            i1, nz1 = __norm(x[n//2:])  # high bits
            i0 *= (1 << ((n + 1) // 2))
            return self.if_else(nz1, [i1, _1], [i0, nz0])  # self.or_(nz0, nz1)

        l = type(a).bit_length
        f = type(a).field.frac_length
        return s * __norm(x)[0] * (2 ** (f - (l - 1)))  # NB: f <= l

    def _rec(self, a):  # enhance performance by reducing no. of truncs
        f = type(a).field.frac_length
        v = self._norm(a)
        b = a * v  # 1/2 <= b <= 1
        theta = int(math.ceil(math.log((2 * f + 1) / 3.5, 2)))
        c = 2.9142 - b * 2
        for _ in range(theta):
            c *= 2 - c * b
        return c * v


class Party:
    """Information about a party in the MPC protocol."""

    def __init__(self, pid, host=None, port=None):
        """Initialize a party with given party identity pid."""
        self.pid = pid
        self.host = host
        self.port = port

    def __repr__(self):
        """String representation of the party."""
        if self.host is None:
            return f'<Party {self.pid}>'

        return f'<Party {self.pid}: {self.host}:{self.port}>'


def generate_configs(m, addresses):
    """Generate party configurations.

    Generates m-party configurations from the addresses given as
    a list of '(host, port)' pairs, specifying the hostnames and
    port numbers for each party.

    Returns a list of ConfigParser instances, which can be saved
    in m separate INI-files. The party owning an INI-file is
    indicated by not specifying its hostname (host='').
    """
    configs = [configparser.ConfigParser() for _ in range(m)]
    for i in range(m):
        host, port = addresses[i]
        if host == '':
            host = 'localhost'
        for config in configs:
            config.add_section(f'Party {i}')
            config.set(f'Party {i}', 'host', host)
            config.set(f'Party {i}', 'port', port)
        configs[i].set(f'Party {i}', 'host', '')  # empty host string for owner
    return configs


def setup():
    """Setup a runtime."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-H', '--HELP', action='store_true', default=False,
                        help=f'show this help message for MPyC and exit')
    parser.add_argument('-h', '--help', action='store_true', default=False,
                        help=f'show {sys.argv[0]} help message (if any)')

    group = parser.add_argument_group('MPyC configuration')
    group.add_argument('-C', '--config', metavar='ini',
                       help='use ini file, defining all m parties')
    group.add_argument('-P', type=str, dest='parties', metavar='addr', action='append',
                       help='use addr=host:port per party (repeat m times)')
    group.add_argument('-M', type=int, metavar='m',
                       help='use m local parties (and run all m, if i is not set)')
    group.add_argument('-I', '--index', type=int, metavar='i',
                       help='set index of this local party to i, 0<=i<m')
    group.add_argument('-T', '--threshold', type=int, metavar='t',
                       help='threshold t, 0<=t<m/2')
    group.add_argument('-B', '--base-port', type=int, metavar='b',
                       help='use port number b+i for party i')
    group.add_argument('--ssl', action='store_true',
                       default=False, help='enable SSL connections')

    group = parser.add_argument_group('MPyC parameters')
    group.add_argument('-L', '--bit-length', type=int, metavar='l',
                       help='default bit length l for secure numbers')
    group.add_argument('-K', '--sec-param', type=int, metavar='k',
                       help='security parameter k, leakage probability 2**-k')
    group.add_argument('--no-log', action='store_true',
                       default=False, help='disable logging messages')
    group.add_argument('--no-async', action='store_true',
                       default=False, help='disable asynchronous evaluation')
    group.add_argument('--no-barrier', action='store_true',
                       default=False, help='disable barriers')

    group = parser.add_argument_group('MPyC misc')
    group.add_argument('--output-windows', action='store_true',
                       default=False, help='screen output for parties i>0 (only on Windows)')
    group.add_argument('--output-file', action='store_true',
                       default=False, help='append output for parties i>0 to party{m}_{i}.log')
    group.add_argument('-f', type=str,
                       default='', help='consume IPython\'s -f argument F')
    parser.set_defaults(bit_length=32, sec_param=30)

    argv = sys.argv  # keep raw args
    options, args = parser.parse_known_args()
    if options.HELP:
        parser.print_help()
        sys.exit()

    if options.help:
        args += ['-h']
        print(f'Showing help message for {sys.argv[0]}, if available:')
        print()
    sys.argv = [sys.argv[0]] + args
    if options.no_log:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(format='{asctime} {message}', style='{',
                            level=logging.INFO, stream=sys.stdout)
    if 'gmpy2' not in sys.modules:
        logging.info('Install package gmpy2 for better performance.')

    if options.config or options.parties:
        # use host:port for each local or remote party
        addresses = []
        if options.config:
            # from ini configuration file
            config = configparser.ConfigParser()
            config.read_file(open(os.path.join('.config', options.config), 'r'))
            for party in config.sections():
                host = config.get(party, 'host')
                port = config.get(party, 'port')
                addresses.append((host, port))
        else:
            # from command-line -P args
            for party in options.parties:
                host, *port_suffix = party.rsplit(':', maxsplit=1)
                port = ' '.join(port_suffix)
                addresses.append((host, port))
        parties = []
        pid = None
        for i, (host, port) in enumerate(addresses):
            if not host:
                pid = i  # empty host string for owner
                host = 'localhost'
            if options.base_port:
                port = options.base_port + i
            elif not port:
                port = 11365 + i
            else:
                port = int(port)
            parties.append(Party(i, host, port))
        if pid is None:
            pid = options.index
    else:
        # use default port for each local party
        if options.M is None:
            options.no_async = True
            pid = 0
            parties = [Party(pid)]
        elif options.M == 1 or options.index is not None:
            pid = options.index or 0
            base_port = options.base_port if options.base_port else 11365
            parties = [Party(i, 'localhost', base_port + i) for i in range(options.M)]
        else:
            import platform
            import subprocess
            prog, args = argv[0], argv[1:]
            for i in range(options.M - 1, 0, -1):
                if options.output_windows and platform.platform().startswith('Windows'):
                    os.system(f'start python {prog} -I{i} {" ".join(args)}')
                elif options.output_file:
                    with open(f'party{options.M}_{i}.log', 'a') as f:
                        cmd_line = ['python', prog, '-I', str(i)] + args
                        f.write('\n')
                        f.write(f'$> {" ".join(cmd_line)}\n')
                        subprocess.Popen(cmd_line, stdout=f, stderr=subprocess.STDOUT)
                else:
                    cmd_line = ['python', prog, '-I', str(i)] + args
                    subprocess.Popen(cmd_line, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            cmd_line = ['python', prog, '-I', str(0)] + args
            subprocess.run(cmd_line)
            sys.exit()

    m = len(parties)
    if options.threshold is None:
        options.threshold = (m - 1) // 2
    assert 2 * options.threshold < m, f'threshold {options.threshold} too large for {m} parties'

    rt = Runtime(pid, parties, options)
    sectypes.runtime = rt
    asyncoro.runtime = rt
    mpyc.random.runtime = rt
    return rt


try:  # suppress exceptions for pydoc etc.
    mpc = setup()
except Exception as exc:
    print('MPyC runtime.setup() exception:', exc)

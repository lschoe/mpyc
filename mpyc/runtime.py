"""The MPyC runtime module is used to execute secure multiparty computations.

Parties perform computations on secret-shared values by exchanging messages.
Shamir's threshold secret sharing scheme is used for fields of prime order.
MPyC provides secure number types and operations, many of which are
available through Python's mechanism for operator overloading.
"""

import os
import sys
import time
import logging
import math
import secrets
import itertools
import configparser
import argparse
import asyncio
import ssl
from mpyc import thresha
from mpyc import sectypes
from mpyc import asyncoro

Future = asyncio.Future
Share = sectypes.Share
gather_shares = asyncoro.gather_shares
mpc_coro = asyncoro.mpc_coro
returnType = asyncoro.returnType

class Runtime:
    """MPyC runtime secure against passive attacks.

    The runtime maintains basic information such as a program counter,
    the list of parties, etc., and handles secret-shared values of type Share.

    1-party case is supported (with option to disable asynchronous evaluation).
    Threshold 0 (no corrupted parties) is supported for m-party case as well
    to enable distributed computation (without secret sharing).
    """

    version = None

    def __init__(self, pid, parties, options):
        """Initialize runtime."""
        self.pid = pid
        self.parties = parties
        self.options = options
        self.threshold = options.threshold
        self._logging_enabled = not options.no_log
        self._program_counter = [0]
        m = len(self.parties)
        t = self.threshold
        #caching (m choose t):
        self._bincoef = math.factorial(m) // math.factorial(t) // math.factorial(m - t)
        self._loop = asyncio.get_event_loop() # cache running loop
        self.start_time = None

    def _increment_pc(self):
        """Increment the program counter."""
        self._program_counter[-1] += 1

    def _fork_pc(self):
        """Fork the program counter."""
        self._program_counter.append(0)

    def _unfork_pc(self):
        """Leave a fork of the program counter."""
        self._program_counter.pop()

    def _send_share(self, peer_pid, data):
        pc = tuple(self._program_counter)
        self.parties[peer_pid].protocol.send_data(pc, data)

    def _expect_share(self, peer_pid):
        pc = tuple(self._program_counter)
        if pc in self.parties[peer_pid].protocol.buffers:
            # Data already received from peer.
            data = self.parties[peer_pid].protocol.buffers.pop(pc)
        else:
            # Data not yet received from peer.
            data = self.parties[peer_pid].protocol.buffers[pc] = Future()
        return data

    def _exchange_shares(self, in_shares):
        out_shares = [None] * len(in_shares)
        for peer_pid, data in enumerate(in_shares):
            if peer_pid == self.pid:
                d = data
            else:
                self._send_share(peer_pid, data)
                d = self._expect_share(peer_pid)
            out_shares[peer_pid] = d
        return out_shares

    async def barrier(self):
        """Barrier for runtime."""
        logging.info(f'Barrier {asyncoro.pc_level} '
                     f'{len(self._program_counter)} '
                     f'{self._program_counter}'
                    )
        if not self.options.no_async:
            while asyncoro.pc_level >= len(self._program_counter):
                await asyncio.sleep(0)

    def run(self, f):
        """Run the given (MPC) coroutine or future until it is done."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            val = None
            try:
                while True:
                    val = f.send(val)
            except StopIteration as exc:
                d = exc.value
            return d

        return loop.run_until_complete(f)

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
        """Start the MPC runtime.

        Open connections with other parties, if any.
        """
        logging.info(f'Start MPyC runtime v{self.version}')
        self.start_time = time.time()
        m = len(self.parties)
        if m == 1:
            return

        # m > 1
        for peer in self.parties:
            peer.protocol = Future() if peer.pid == self.pid else None
        factory = asyncoro.SharesExchanger
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
                    await loop.create_connection(factory, peer.host, peer.port, ssl=context,
                                                 server_hostname=server_hostname)
                    break
                except Exception as exc:
                    logging.debug(exc)
                time.sleep(1)

        await self.parties[self.pid].protocol
        if self.options.ssl:
            logging.info('SSL connections to all parties.')
        else:
            logging.info('Connected to all parties.')
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
        x = [a.df for a in x] # Extract values from all elements of x.
        shares = [None] * len(senders)
        for i, peer_pid in enumerate(senders):
            if peer_pid == self.pid:
                in_shares = thresha.random_split(x, t, m)
                for other_pid, data in enumerate(in_shares):
                    data = field.to_bytes(data)
                    if other_pid == self.pid:
                        shares[i] = data
                    else:
                        self._send_share(other_pid, data)
            else:
                shares[i] = self._expect_share(peer_pid)
        shares = await gather_shares(shares)
        return [[field(a) for a in field.from_bytes(r)] for r in shares]

    def output(self, x, receivers=None, threshold=None):
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
        return gather_shares(y)

    @mpc_coro
    async def _recombine(self, x, receivers, t):
        """Recombine shares of elements of x."""
        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, Share):
            field = sftype.field
            if field.frac_length == 0:
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
                self._send_share(peer_pid, field.to_bytes(x))
        # Receive and recombine shares if this party is a receiver.
        if self.pid in receivers:
            shares = [None] * t
            for i in range(t):
                shares[i] = self._expect_share((self.pid - t + i) % m)
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
        sftype = type(x[0]) # all elts assumed of same type
        if issubclass(sftype, Share):
            field = sftype.field
            if field.frac_length == 0:
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
    async def trunc(self, a, f=None):
        """Secure truncation of f least significant bits of a.

        Probabilistic rounding of a / 2**f.
        """
        stype = type(a)
        await returnType(stype)
        Zp = stype.field
        l = stype.bit_length
        if f is None:
            f = Zp.frac_length
            rsf = Zp.rshift_factor
        else:
            rsf = 1 / Zp(1 << f)
        k = self.options.security_parameter
        r_bits = await self.random_bits(Zp, f)
        r_modf = 0
        for i in range(f - 1, -1, -1):
            r_modf <<= 1
            r_modf += r_bits[i].value
        r_divf = self._random(Zp, 1<<(k + l - f))
        a = await gather_shares(a)
        c = await self.output(a + ((1<<l) + (r_divf.value << f) + r_modf)) # pylint: disable=E1101
        c = c.value % (1<<f)
        return (r_modf - c + a) * rsf

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
        if stype.__name__.startswith('SecFld'):
            prfs = self.parties[self.pid].prfs(field.order)
            while True:
                r, s = self._randoms(field, 2)
                z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), 1)
                if await self.output(r * s + z[0], threshold=2 * t):
                    break
        else:
            r = self._random(field) #failure shared r is 0 with prob. 1/p
        a = await gather_shares(a)
        if stype.__name__.startswith('SecFld'):
            z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), 1)
            b = a * r + z[0]
        else:
            b = a * r
        c = await self.output(b, threshold=2 * t)
        return c == 0

    @mpc_coro
    async def neg(self, a):
        """Secure negation (additive inverse) of a."""
        stype = type(a)
        if stype.field.frac_length == 0:
            await returnType(stype)
        else:
            await returnType((stype, a.integral))
        a = await gather_shares(a)
        return -a

    @mpc_coro
    async def add(self, a, b):
        """Secure addition of a and b."""
        stype = type(a)
        if stype.field.frac_length == 0:
            await returnType(stype)
        else:
            await returnType((stype, a.integral and b.integral))
        a, b = await gather_shares(a, b)
        return a + b

    @mpc_coro
    async def sub(self, a, b):
        """Secure subtraction of a and b."""
        stype = type(a)
        if stype.field.frac_length == 0:
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
        if field.frac_length == 0:
            await returnType(stype)
        else:
            a_integral = a.integral
            b_integral = isinstance(b, int) or (isinstance(b, Share) and b.integral)
            await returnType((stype, a_integral and b_integral))
        if not isinstance(b, Share):
            a = await gather_shares(a)
            return a * b

        if a is b:
            a = b = await gather_shares(a)
        else:
            a, b = await gather_shares(a, b)
        if field.frac_length > 0 and a_integral:
            a = a * field.rshift_factor # NB: no inplace a *=
        c = self._reshare(a * b)
        if field.frac_length > 0 and not a_integral:
            c = self.trunc(stype(c))
        return c

    def div(self, a, b):
        """Secure division of a by b, for nonzero b."""
        if isinstance(b, Share):
            if type(b).field.frac_length == 0:
                c = self.reciprocal(b)
            else:
                c = self._rec(b)
        else: # isinstance(a, Share) ensured
            if not isinstance(b, a.field):
                b = a.field(b)
            c = b._reciprocal()
        return a * c

    @mpc_coro
    async def reciprocal(self, a):
        """Secure reciprocal (multiplicative inverse) of a, for nonzero a."""
        stype = type(a)
        field = stype.field
        await returnType(stype)
        a = await gather_shares(a)
        while 1:
            r = self._random(field)
            ar = await self.output(a * r, threshold=2*self.threshold)
            if ar:
                break
        r *= field.lshift_factor
        return r / ar

    def pow(self, a, b):
        """Secure exponentation a raised to the power of b, for public integer b."""
        if b == 0:
            return type(a)(1)

        if b < 0:
            a = self.reciprocal(a)
            b = -b
        d = a
        c = 1
        for i in range(b.bit_length() - 1):
            # d = a ** (1 << i) holds
            if b & (1 << i):
                c = c * d
            d = d * d
        c = c * d
        return c

    def and_(self, a, b):
        """Secure bitwise and of a and b."""
        field = type(a).field
        a2 = self.to_bits(a)
        b2 = self.to_bits(b)
        c2 = [a2[i] * b2[i] for i in range(len(a2))]
        return sum(c2[i] * field(1 << i) for i in range(len(c2)))

    def xor(self, a, b):
        """Secure bitwise xor of a and b."""
        return a + b

    def invert(self, a):
        """Secure bitwise inverse (not) of a."""
        return a + type(a)(a.field.order - 1)

    def or_(self, a, b):
        """Secure bitwise or of a and b."""
        field = type(a).field
        a2 = self.to_bits(a)
        b2 = self.to_bits(b)
        c2 = [a2[i] + b2[i] + a2[i] * b2[i] for i in range(len(a2))]
        return sum(c2[i] * field(1 << i) for i in range(len(c2)))
#        return a + b - a * b # wrong, go via bits

    def eq(self, a, b):
        """Secure comparison a == b."""
        return self.is_zero(a - b)

    def ge(self, a, b):
        """Secure comparison a >= b."""
        return self.sgn(a - b, GE=True)

    def is_zero(self, a):
        """Secure zero test a == 0."""
        if type(a).__name__.startswith('SecFld'):
            return 1 - self.pow(a, a.field.order - 1)

        if (a.bit_length/2 > self.options.security_parameter >= 8 and a.field.order%4 == 3):
            return self._is_zero(a)

        return self.sgn(a, EQ=True)

    @mpc_coro
    async def _is_zero(self, a):
        """Probabilistic zero test."""
        stype = type(a)
        await returnType((stype, True))
        Zp = stype.field

        k = self.options.security_parameter
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
        e *= Zp.lshift_factor
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
        for i in range(l - 1, -1, -1):
            r_modl <<= 1
            r_modl += r_bits[i].value
        a = await gather_shares(a)
        a_rmodl = a + ((1<<l) + r_modl)
        k = self.options.security_parameter
        r_divl = self._random(Zp, 1<<k)
        c = await self.output(a_rmodl + (r_divl.value << l)) # pylint: disable=E1101
        c = c.value % (1<<l)

        if not EQ: # a la Toft
            s_sign = (await self.random_bits(Zp, 1, signed=True))[0].value
            e = [None] * (l + 1)
            sumXors = 0
            for i in range(l - 1, -1, -1):
                c_i = (c >> i) & 1
                e[i] = Zp(s_sign + r_bits[i].value - c_i + 3 * sumXors)
                sumXors += 1 - r_bits[i].value if c_i else r_bits[i].value
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

        z *= Zp.lshift_factor
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
        """Secure least significant bit of a.""" # a la [ST06]
        stype = type(a)
        await returnType((stype, True))
        Zp = stype.field
        l = stype.bit_length
        k = self.options.security_parameter
        b = self.random_bit(stype)
        a, b = await gather_shares(a, b)
        b *= Zp.rshift_factor
        r = self._random(Zp, 1 << (l + k - 1))
        c = await self.output(a + ((1<<l) + (r.value << 1) + b.value)) # pylint: disable=E1101
        x = 1 - b if c.value & 1 else b # xor
        x *= Zp.lshift_factor
        return x

    @mpc_coro
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

    @mpc_coro
    async def lin_comb(self, a, x):
        """Secure linear combination: dot product of public a and secret x."""
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
        if field.frac_length == 0:
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
        if field.frac_length > 0 and x_integral:
            s *= field.rshift_factor
        s = self._reshare(field(s))
        if field.frac_length > 0 and not x_integral:
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
            h = await self._reshare(h) # TODO: handle trunc
            x = x[2*len(h):] + h
        return x[0]

    @mpc_coro
    async def vector_add(self, x, y):
        """Secure addition of vectors x and y."""
        x, y = x[:], y[:]
        stype = type(x[0])
        if stype.field.frac_length == 0:
            await returnType(stype, len(x))
        else:
            await returnType((stype, x[0].integral and y[0].integral), len(x))
        x, y = await gather_shares(x, y)
        for i in range(len(x)):
            x[i] = x[i] + y[i]
        return x

    @mpc_coro
    async def vector_sub(self, x, y):
        """Secure subtraction of vectors x and y."""
        x, y = x[:], y[:]
        stype = type(x[0])
        if stype.field.frac_length == 0:
            await returnType(stype, len(x))
        else:
            await returnType((stype, x[0].integral and y[0].integral), len(x))
        x, y = await gather_shares(x, y)
        for i in range(len(x)):
            x[i] = x[i] - y[i]
        return x

    @mpc_coro
    async def matrix_add(self, A, B, tr=False):
        """Secure addition of matrices A and (transposed) B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        await returnType(type(A[0][0]), len(A), len(A[0]))
        A, B = await gather_shares(A, B)
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = A[i][j] + (B[j][i] if tr else B[i][j])
        return A

    @mpc_coro
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
        if field.frac_length == 0:
            await returnType(stype, len(x))
        else:
            a_integral = a.integral
            await returnType((stype, a_integral and x[0].integral), len(x))

        a, x = await gather_shares(a, x)
        if field.frac_length > 0 and a_integral:
            a = a * field.rshift_factor # NB: no inplace a *=
        for i in range(len(x)):
            x[i] = x[i] * a
        x = await self._reshare(x)
        if field.frac_length > 0 and not a_integral:
            x = [self.trunc(stype(xi)) for xi in x]
        return x

    @mpc_coro
    async def _if_else_list(self, a, x, y):
        x, y = x[:], y[:]
        stype = type(a)
        field = stype.field
        if field.frac_length == 0:
            await returnType(stype, len(x))
        else:
            a_integral = a.integral
            if not a_integral:
                raise ValueError('condition must be integral')
            await returnType((stype, a_integral and x[0].integral and y[0].integral), len(x))

        a, x, y = await gather_shares(a, x, y)
        if field.frac_length > 0:
            a = a * field.rshift_factor # NB: no inplace a *=
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
            truncy = stype.field.frac_length > 0
        else:
            await returnType(Future)
            truncy = False
        for i in range(len(x)):
            x[i] = x[i] * y[i]
        x = await self._reshare(x)
        if truncy:
            x = [self.trunc(stype(xi)) for xi in x]
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
        if field.frac_length > 0:
            C = [[self.trunc(stype(Cij)) for Cij in Ci] for Ci in C]
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
        if field.frac_length > 0:
            A = [[self.trunc(stype(Aij)) for Aij in Ai] for Ai in A]
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
            bound = (bound - 1) // self._bincoef + 1
        m = len(self.parties)
        prfs = self.parties[self.pid].prfs(bound)
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
        f1 = 1
        if issubclass(sftype, Share):
            await returnType((sftype, True), n)
            field = sftype.field
            if sftype.__name__.startswith('SecFld'):
                prss0 = True
            f1 = field.lshift_factor
        else:
            await returnType(Future)
            field = sftype

        m = len(self.parties)

        if not isinstance(field.modulus, int):
            prfs = self.parties[self.pid].prfs(2)
            bits = thresha.pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
            return bits

        bits = [None] * n
        p = field.modulus
        if not signed:
            q = (p + 1) >> 1 # q = 1/2 mod p
        prfs = self.parties[self.pid].prfs(p)
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
                    bits[h] = field(f1 * s)
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
    async def to_bits(self, a):
        """Secure extraction of bits of a.""" # a la [ST06].
        stype = type(a)
        l = stype.bit_length
        await returnType((stype, True), l)
        field = stype.field

        r_bits = await self.random_bits(field, l)
        r_modl = 0
        for i in range(l - 1, -1, -1):
            r_modl *= 2
            r_modl += r_bits[i].value
        if isinstance(field.modulus, int):
            k = self.options.security_parameter
            r_divl = self._random(field, 1<<k)
            a = await gather_shares(a)
            c = await self.output(a + ((1<<l) + (r_divl.value << l) - r_modl)) # pylint: disable=E1101
            c = c.value % (1<<l)
            c_bits = [(c >> i) & 1 for i in range(l)]
            r_bits = [stype(r.value) for r in r_bits]
            return self.add_bits(r_bits, c_bits)

        a = await gather_shares(a)          # use a.value?
        c = await self.output(a + field(r_modl))  # fix this in +
        c = int(c.value)
        return [r_bits[i] + ((c >> i) & 1) for i in range(l)]

    def from_bits(self, x):
        """Recover secure number from its binary representation x."""
        # TODO: also handle negative numbers with sign bit
        return sum(x[i] * (1<<i) for i in range(len(x)))

    def _norm(self, a): # signed normalization factor
        x = self.to_bits(a) # low to high bits
        b = x[-1] # sign bit
        s = 1 - b * 2 # sign s = (-1)^b
        x = x[:-1]
        def __norm(x):
            n = len(x)
            if n == 1:
                t = s * x[0] + b # self.xor(b, x[0])
                return 2 - t, t

            i0, nz0 = __norm(x[:n//2]) # low bits
            i1, nz1 = __norm(x[n//2:]) # high bits
            i0 *= (1 << ((n + 1) // 2))
            return nz1 * (i1 - i0) + i0, nz0 + nz1 - nz0 * nz1 # self.or_(nz0, nz1)
        l = type(a).bit_length
        f = type(a).field.frac_length
        return s * __norm(x)[0] * (2 ** (f - (l - 1))) # NB: f <= l

    def _rec(self, a): # enhance performance by reducing no. of truncs
        f = type(a).field.frac_length
        v = self._norm(a)
        b = a * v # 1/2 <= b <= 1
        theta = int(math.ceil(math.log((2 * f + 1) / 3.5, 2)))
        c = 2.9142 - b * 2
        for _ in range(theta):
            c *= 2 - c * b
        return c * v

class _Party:
    """Information about a party in the MPC protocol."""

    def __init__(self, pid, host=None, port=None, keys=None):
        """Initialize a party with given party identity pid."""
        self.pid = pid
        self.host = host
        self.port = port
        self.keys = keys
        self._prfs = {}

    def prfs(self, bound):
        """PRFs with codomain range(bound) for pseudorandom secret sharing.

        Return a mapping from sets of parties to PRFs.
        """
        try:
            return self._prfs[bound]
        except KeyError:
            self._prfs[bound] = {}
            for subset, key in self.keys.items():
                self._prfs[bound][subset] = thresha.PRF(key, bound)
            return self._prfs[bound]

    def __repr__(self):
        """String representation of the party."""
        if self.host is None:
            return f'<_Party {self.pid}>'

        return f'<_Party {self.pid}: {self.host}:{self.port}>'

def generate_configs(m, addresses):
    """Generate party configurations.

    Generates m party configurations with thresholds 0 up to (m-1)//2.
    addresses is a list of '(host, port)' pairs, specifying the
    hostnames and port numbers for each party. Moreover, the keys
    used in pseudorandom secret sharing (PRSS) are generated.

    The m party configurations are returned as a list of ConfigParser
    instances, which be saved in m separate INI-files.
    """
    parties = range(m)
    configs = [configparser.ConfigParser() for _ in parties]
    for p in parties:
        host, port = addresses[p]
        if host == '':
            host = 'localhost'
        for config in configs:
            config.add_section(f'Party {p}')
            config.set(f'Party {p}', 'host', host)
            config.set(f'Party {p}', 'port', port)

    for t in range((m + 1) // 2):
        for subset in itertools.combinations(parties, m - t):
            key = hex(secrets.randbits(128)) # 128-bit key
            subset_str = ' '.join(map(str, subset))
            for p in subset:
                configs[p].set(f'Party {p}', subset_str, key)
    return configs

def _load_config(filename, t=None):
    """Load m-party configuration file using threshold t (default (m-1) // 2).

    Configuration files are simple INI-files containing information
    (hostname and port number) about the other parties in the protocol.

    One of the parties owns the configuration file and for this party
    additional information on PRSS keys is available.

    Returns the pid of the owning party and a list of _Party objects.
    """
    config = configparser.ConfigParser()
    config.read_file(open(filename, 'r'))
    m = len(config.sections())
    if t is None:
        t = (m - 1) // 2
    parties = [None] * m
    for party in config.sections():
        pid = int(party[6:]) # strip 'Party ' prefix
        host = config.get(party, 'host')
        port = config.getint(party, 'port')
        if len(config.options(party)) > 2:
            # read PRSS keys
            my_pid = pid
            keys = {}
            for option in config.options(party):
                if not option in ['host', 'port']:
                    subset = frozenset(map(int, option.split()))
                    if len(subset) == m - t:
                        keys[subset] = config.get(party, option)
            parties[my_pid] = _Party(my_pid, host, port, keys)
        else:
            parties[pid] = _Party(pid, host, port)
    return my_pid, parties

def setup():
    """Setup a runtime."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--HELP', action='store_true', default=False,
                        help=f'show -h help message for {sys.argv[0]}, if any')
    group = parser.add_argument_group('MPyC')
    group.add_argument('-c', '--config', metavar='C',
                       help='party configuration file C, which defines M')
    group.add_argument('-t', '--threshold', type=int, metavar='T',
                       help='threshold T, 2T+1<=M')
    group.add_argument('-l', '--bit-length', type=int, metavar='L',
                       help='maximum bit length L (for comparisons etc.)')
    group.add_argument('-k', '--security-parameter', type=int, metavar='K',
                       help='security parameter K for leakage probability 1/2**K')
    group.add_argument('--no-log', action='store_true',
                       default=False, help='disable logging')
    group.add_argument('--ssl', action='store_true',
                       default=False, help='enable SSL connections')
    group.add_argument('--no-async', action='store_true',
                       default=False, help='disable asynchronous evaluation')
    group.add_argument('-f', type=str,
                       default='', help='consume IPython string')
    parser.set_defaults(bit_length=32, security_parameter=30)
    options, args = parser.parse_known_args()
    if options.HELP:
        args += ['-h']
        print(f'Showing -h help message for {sys.argv[0]}, if available:')
        print()
    sys.argv = [sys.argv[0]] + args
    if options.no_log:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(format='{asctime} {message}', style='{',
                            level=logging.INFO, stream=sys.stdout)
    if 'gmpy2' not in sys.modules:
        logging.info('Install package gmpy2 for better performance.')
    if not options.config:
        options.no_async = True
        pid = 0
        parties = [_Party(pid, keys={frozenset([pid]): hex(secrets.randbits(128))})]
    else:
        options.config = os.path.join('.config', options.config)
        pid, parties = _load_config(options.config, options.threshold)
    m = len(parties)
    if options.threshold is None:
        options.threshold = (m - 1) // 2
    assert 2 * options.threshold < m

    rt = Runtime(pid, parties, options)
    sectypes.runtime = rt
    asyncoro.runtime = rt
    import mpyc.random
    mpyc.random.runtime = rt
    rt.version = mpyc.__version__
    global mpc
    mpc = rt

mpc = None
try: # suppress exceptions for pydoc etc.
    setup()
except Exception as exc:
    print('MPyC runtime.setup() exception:', exc)

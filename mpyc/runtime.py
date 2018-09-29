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
import operator
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

    def __init__(self, id, parties, options):
        """Initialize runtime."""
        self.id = id
        self.parties = parties
        self.options = options
        self.threshold = options.threshold
        self._program_counter = [0]
        m = len(self.parties)
        t = self.threshold
        #caching (m choose t):
        self._bincoef = math.factorial(m) // math.factorial(t) // math.factorial(m - t)

    def _increment_pc(self):
        """Increment the program counter."""
        self._program_counter[-1] += 1

    def _fork_pc(self):
        """Fork the program counter."""
        self._program_counter.append(0)

    def _unfork_pc(self):
        """Leave a fork of the program counter."""
        self._program_counter.pop()

    def _send_share(self, peer_id, data):
        pc = tuple(self._program_counter)
        self.parties[peer_id].protocol.send_data(pc, data)

    def _expect_share(self, peer_id):
        pc = tuple(self._program_counter)
        if pc in self.parties[peer_id].protocol.buffers:
            # Data already received from peer.
            data = self.parties[peer_id].protocol.buffers.pop(pc)
        else:
            # Data not yet received from peer.
            data = self.parties[peer_id].protocol.buffers[pc] = Future()
        return data

    def _exchange_shares(self, in_shares):
        out_shares = [None] * len(in_shares)
        for peer_id, data in enumerate(in_shares):
            if peer_id == self.id:
                d = data
            else:
                self._send_share(peer_id, data)
                d = self._expect_share(peer_id)
            out_shares[peer_id] = d
        return out_shares

    async def barrier(self):
        """Barrier for runtime."""
        logging.info(f'Barrier {asyncoro.pc_level} {len(self._program_counter)} {self._program_counter}')
        if not self.options.no_async:
            while asyncoro.pc_level >= len(self._program_counter):
                await asyncio.sleep(0)

    def run(self, f):
        """Run the given (MPC) coroutine or future until it is done."""
        return self._loop.run_until_complete(f)

    def log(self, enable=None):
        """Toggle/enable/disable logging."""
        if enable is None:
            self.logging_enabled = not self.logging_enabled
        else:
            self.logging_enabled = enable
        if self.logging_enabled:
            logging.disable(logging.NOTSET)
        else:
            logging.disable(logging.INFO)

    def start(self):
        """Start the MPC runtime.

        Open connections with other parties, if any.
        """
        logging.info(f'Start MPyC runtime v{self.version}')
        self.start_time = time.time()
        m = len(self.parties)
        if  m == 1:
            return

        # m > 1
        for peer in self.parties:
            peer.protocol = Future() if peer.id == self.id else None
        factory = lambda: asyncoro.SharesExchanger(self)
        if self.options.ssl:
            crtfile = os.path.join('.config', f'party_{self.id}.crt')
            keyfile = os.path.join('.config', f'party_{self.id}.key')
            cafile = os.path.join('.config', 'mpyc_ca.crt')

        # Listen for all parties < self.id.
        if self.id:
            listen_port = self.parties[self.id].port
            if self.options.ssl:
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                context.load_cert_chain(crtfile, keyfile=keyfile)
                context.load_verify_locations(cafile=cafile)
                context.verify_mode = ssl.CERT_REQUIRED
            else:
                context = None
            listen = self._loop.create_server(factory, port=listen_port, ssl=context)
            server = self.run(listen)
            logging.debug(f'Listening on port {listen_port}')

        # Connect to all parties > self.id.
        for peer in self.parties[self.id + 1:]:
            logging.debug(f'Connecting to {peer}')
            while True:
                try:
                    if self.options.ssl:
                        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                        context.load_cert_chain(crtfile, keyfile=keyfile)
                        context.load_verify_locations(cafile=cafile)
                        server_hostname = f'MPyC party {peer.id}'
                    else:
                        context = None
                        server_hostname = None
                    connect = self._loop.create_connection(factory, peer.host, peer.port,
                                           ssl=context, server_hostname=server_hostname)
                    self.run(connect)
                    break
                except Exception as e:
                    logging.debug(e)
                time.sleep(1)

        self.run(self.parties[self.id].protocol)
        if self.options.ssl:
            logging.info('SSL connections to all parties.')
        else:
            logging.info('Connected to all parties.')
        if self.id:
            server.close()

    def shutdown(self):
        """Shutdown the MPC runtime.

        Close all connections, if any.
        """
        m = len(self.parties)
        if m > 1:
            # Wait for all parties.
            self.run(self.output(self.input(sectypes.SecFld(101)(self.id))))
            # Close connections to all parties.
            for peer in self.parties:
                if peer.id != self.id:
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

        Value x is a secure number, or x list of secure numbers.
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
        assert value is None or self.id in senders

        m = len(self.parties)
        t = self.threshold
        x = [a.df for a in x] # Extract values from all elements of x.
        shares = [None] * len(senders)
        for i, peer_id in enumerate(senders):
            if peer_id == self.id:
                in_shares = thresha.random_split(x, t, m)
                for other_id, data in enumerate(in_shares):
                    data = field.to_bytes(data)
                    if other_id == self.id:
                        shares[i] = data
                    else:
                        self._send_share(other_id, data)
            else:
                shares[i] = self._expect_share(peer_id)
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
            if sftype.field.frac_length == 0:
                await returnType(sftype, len(x))
            else:
                await returnType((sftype, x[0].integral), len(x))
            x = await gather_shares(x)
            field = type(x[0])
        else:
            await returnType(Share, len(x))
            field = sftype

        m = len(self.parties)
        # Send share to all successors in receivers.
        for peer_id in receivers:
            if 0 < (peer_id - self.id) % m <= t:
                self._send_share(peer_id, field.to_bytes(x))
        # Receive and recombine shares if this party is a receiver.
        if self.id in receivers:
            shares = [None] * t
            for i in range(t):
                shares[i] = self._expect_share((self.id - t + i) % m)
            shares = await gather_shares(shares)
            shares = [((self.id - t + j) % m + 1, field.from_bytes(shares[j])) for j in range(t)]
            shares.append((self.id + 1, x))
            return thresha.recombine(field, shares)
        else:
            return [None] * len(x)

    @mpc_coro
    async def _reshare(self, x):
        x_is_list = isinstance(x, list)
        if not x_is_list:
            x = [x]
        sftype = type(x[0]) # all elts assumed of same type
        if issubclass(sftype, Share):
            if  sftype.field.frac_length == 0:
                await returnType(sftype, len(x))
            else:
                await returnType((sftype, x[0].integral), len(x))
            x = await mpc.gather(x)
            field = sftype.field
        else:
            await returnType(Share)
            field = sftype

        m = len(self.parties)
        t = self.threshold
        in_shares = thresha.random_split(x, t, m)
        in_shares = [field.to_bytes(elts) for elts in in_shares]
        # Recombine the first 2t+1 output_shares.
        out_shares = await gather_shares(self._exchange_shares(in_shares)[:2 * t + 1])
        y = thresha.recombine(field, [(j + 1, field.from_bytes(out_shares[j])) for j in range(len(out_shares))])

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
        k = self.options.security_parameter
        r_bits = await gather_shares(self.random_bits(Zp, f))
        r_modf = 0
        for i in range(f - 1, -1, -1):
            r_modf <<= 1
            r_modf += r_bits[i].value
        r_divf = self.random(Zp, 1<<(k + l - f))
        a = await gather_shares(a)
        c = await self.output(a + (1<<l) + r_modf + (1<<f) * r_divf)
        c = c.value % (1<<f)
        return (a - c + r_modf) / (1<<f)

    def eq_public(self, a, b):
        """Secure public equality test of a and b."""
        return self.is_zero_public(a - b)

    @mpc_coro
    async def is_zero_public(self, a) -> Future:
        """Secure public zero test of a."""
        stype = type(a)
        m = len(self.parties)
        t = self.threshold
        if stype.__name__.startswith('SecFld'):
            prfs = self.parties[self.id].prfs(stype.field.modulus)
            while True:
                r, s = self.randoms(stype.field, 2)
                z = thresha.pseudorandom_share_zero(stype.field, m, self.id, prfs, self._prss_uci(), 1)
                if await self.output(r * s + z[0], threshold=2 * t):
                    break
        else:
            r = self.random(stype.field) #failure shared r is 0 with prob. 1/p
        a = await gather_shares(a)
        if stype.__name__.startswith('SecFld'):
            z = thresha.pseudorandom_share_zero(stype.field, m, self.id, prfs, self._prss_uci(), 1)
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
        if stype.field.frac_length == 0:
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
        if stype.field.frac_length > 0 and a_integral:
            a = a / (1 << stype.field.frac_length) # expensive # a /= inplace issue
        c = self._reshare(a * b)
        if stype.field.frac_length > 0 and not a_integral:
            c = self.trunc(stype(c.df))  # c.df
        return c

    def div(self, a, b):
        """Secure division of a by b, for non-zero b."""
        if isinstance(b, Share):
            if type(b).field.frac_length == 0:
                c = self.reciprocal(b)
            else:
                c = self._rec(b)
        elif isinstance(b, a.field):
            c = 1 / b
        else:
            c = 1 / a.field(b)
        return a * c

    @mpc_coro
    async def reciprocal(self, a):
        """Secure reciprocal (multiplicative inverse) of a, for non-zero a."""
        stype = type(a)
        await returnType(stype)
        a = await gather_shares(a)
        r = self.random(stype.field)
        ar = await self.output(a * r, threshold=2*self.threshold)
        if ar == 0:
            return self.reciprocal(a)
        return r * (1 << stype.field.frac_length) / ar

    def pow(self, a, b):
        """Secure exponentation of a to public power b."""
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
        """Secure logical and of bits a and b."""
        return a * b

    def xor(self, a, b):
        """Secure logical xor of bits a and b."""
        return a + b - 2 * a * b

    def invert(self, a):
        """Secure logical inverse (not) of bit a."""
        return 1 - a

    def or_(self, a, b):
        """Secure logical or of bits a and b."""
        return a + b - a * b

    def eq(self, a, b):
        """Secure comparison a == b."""
        return self.is_zero(a - b)

    def ge(self, a, b):
        """Secure comparison a >= b."""
        return self.sgn(a - b, GE=True)

    def is_zero(self, a):
        """Secure zero test a == 0."""
        if type(a).__name__.startswith('SecFld'):
            return 1 - self.pow(a, a.field.modulus - 1)
        elif (a.bit_length > 2 * self.options.security_parameter
              and a.field.modulus % 4 == 3):
            return self._is_zero(a)
        else:
            return self.sgn(a, EQ=True)

    @mpc_coro
    async def _is_zero(self, a):
        """Probabilistic zero test."""
        stype = type(a)
        await returnType((stype, True))
        Zp = a.field

        k = self.options.security_parameter
        z = self.random_bits(Zp, k)
        u = self.randoms(Zp, k)
        u2 = self.schur_prod(u, u)
        a, u2, z = await gather_shares(a, u2, z)
        a = a.value
        r = self.randoms(Zp, k)
        c = [Zp(a * r[i].value + (1 - (z[i].value << 1)) * u2[i].value) for i in range(k)]
        # -1 is non-square for Blum p, u_i !=0 w.v.h.p.
        # If a == 0, c_i is square mod p iff z[i] == 0.
        # If a != 0, c_i is square mod p independent of z[i].
        c = await self.output(c, threshold=2*self.threshold)
        for i in range(k):
            if c[i] == 0:
                c[i] = Zp(1)
            else:
                c[i] = 1 - z[i] if c[i].is_sqr() else z[i]
        e = await gather_shares(self.prod(c))
        return e * (1 << stype.field.frac_length)

    @mpc_coro
    async def sgn(self, a, EQ=False, GE=False):
        """Secure sign(um) of a, -1 if a < 0 else 0 if a == 0 else 1."""
        stype = type(a)
        await returnType((stype, True))
        Zp = a.field

        l = stype.bit_length
        r_bits = await gather_shares(self.random_bits(Zp, l))
        r_modl = 0
        for i in range(l - 1, -1, -1):
            r_modl <<= 1
            r_modl += r_bits[i].value
        a = await gather_shares(a)
        a_rmodl = a + (1<<l) + r_modl
        k = self.options.security_parameter
        r_divl = self.random(Zp, 1<<k)
        c = await self.output(a_rmodl + (1<<l) * r_divl)
        c = c.value % (1<<l)

        if not EQ: # a la Toft
            s_bit = (await gather_shares(self.random_bits(Zp, 1)))[0]
            s_sign = (1 - 2 * s_bit).value
            e = [None] * (l + 1)
            sumXors = 0
            for i in range(l - 1, -1, -1):
                c_i = (c >> i) & 1
                e[i] = Zp(s_sign + r_bits[i].value - c_i + 3 * sumXors)
                sumXors += 1 - r_bits[i].value if c_i else r_bits[i].value
            e[l] = Zp(s_sign - 1 + 3 * sumXors)
            f = await self.is_zero_public(self.prod(e))
            UF = s_bit if f == 1 else 1 - s_bit
            z = (a_rmodl - (c + UF * (1<<l))) / (1<<l)

        if not GE:
            h = self.prod([r_bits[i] if (c >> i) & 1 else 1 - r_bits[i] for i in range(l)])
            h = await gather_shares(h)
            if EQ:
                z = h
            else:
                z = (1 - h) * (2 * z - 1)
                z = self._reshare(z)
                z = await gather_shares(z)

        return z * (1 << stype.field.frac_length)

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
        l = stype.bit_length
        k = self.options.security_parameter
        b = self.random_bit(stype)
        a, b = await gather_shares(a, b)
        b = b / (1 << stype.field.frac_length)
        r = self.random(stype.field, 1 << (l + k))
        c = await self.output(a + (1<<l) + b + (r.value << 1))
        x = 1 - b if c.value & 1 else b #xor
        return x * (1 << stype.field.frac_length)

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
        if stype.field.frac_length == 0:
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
        if stype.field.frac_length > 0 and x_integral:
            f1 = 1 / stype.field(1 << stype.field.frac_length) # expensive
            s = s * f1.value
        s = self._reshare(stype.field(s))
        if stype.field.frac_length > 0 and not x_integral:
            s = self.trunc(stype(s.df))
        return s

    @mpc_coro
    async def prod(self, x):
        """Secure product of all elements in x (in log_2 len(x) rounds)."""
        x = x[:]
        if isinstance(x[0], Share):
            await returnType(type(x[0]))
            x = await gather_shares(x)
        else:
            Share.field = type(x[0])  # avoid hack? see prod in sgn, _is_zero
            await returnType(Share)

        while len(x) > 1:
            h = [None] * (len(x)//2)
            for i in range(len(h)):
                h[i] = x[2*i] * x[2*i+1]
            h = await gather_shares(self._reshare(h))
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
    async def matrix_add(self, A, B):
        """Secure addition of matrices A and B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        await returnType(type(A[0][0]), len(A), len(A[0]))
        A, B = await gather_shares(A, B)
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = A[i][j] + B[i][j]
        return A

    @mpc_coro
    async def matrix_sub(self, A, B):
        """Secure subtraction of matrices A and B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        await returnType(type(A[0][0]), len(A), len(A[0]))
        A, B = await gather_shares(A, B)
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = A[i][j] - B[i][j]
        return A

    @mpc_coro
    async def scalar_mul(self, a, x):
        """Secure scalar multiplication of scalar a with vector x."""
        x = x[:]
        stype = type(a)
        if stype.field.frac_length == 0:
            await returnType(stype, len(x))
        else:
            a_integral = a.integral
            await returnType((stype, a_integral and x[0].integral), len(x))
        a, x = await gather_shares(a, x)
        if stype.field.frac_length > 0 and a_integral:
            a = a / (1 << stype.field.frac_length) # expensive # a /= inplace issue
        for i in range(len(x)):
            x[i] = x[i] * a
        x = self._reshare(x)
        x = await gather_shares(x)
        if stype.field.frac_length > 0 and not a_integral:
            x = [self.trunc(stype(xi)) for xi in x]
        return x

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
            await returnType(Share)
            truncy = False
        for i in range(len(x)):
            x[i] = x[i] * y[i]
        x = self._reshare(x)
        x = await gather_shares(x)
        if truncy:
            x = [self.trunc(stype(xi)) for xi in x]
        return x

    @mpc_coro
    async def matrix_prod(self, A, B, tr=False):
        """Secure matrix product of A with (transposed) B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        stype = type(A[0][0])
        n = len(B) if tr else len(B[0])
        await returnType(stype, len(A), n)
        A, B = await gather_shares(A, B)
        C = [None] * len(A)
        for ia in range(len(A)):
            C[ia] = [None] * n
            for ib in range(n):
                s = 0
                for i in range(len(A[0])):
                    s += A[ia][i].value * (B[ib][i] if tr else B[i][ib]).value
                C[ia][ib] = stype.field(s)
            C[ia] = self._reshare(C[ia])
        C = await gather_shares(C)
        if stype.field.frac_length > 0:
            C = [[self.trunc(stype(Cij)) for Cij in Ci] for Ci in C]
        return C

    @mpc_coro
    async def gauss(self, A, d, b, c):
        """Secure Gaussian elimination A d - b c."""
        A, b, c = [r[:] for r in A], b[:], c[:]
        stype = type(A[0][0])
        n = len(A[0])
        await returnType(stype, len(A), n)
        A, d, b, c = await gather_shares(A, d, b, c)
        d = d.value
        for i in range(len(A)):
            b[i] = b[i].value
            for j in range(n):
                A[i][j] = stype.field(A[i][j].value * d - b[i] * c[j].value)
            A[i] = self._reshare(A[i])
        A = await gather_shares(A)
        if stype.field.frac_length > 0:
            A = [[self.trunc(stype(Aij)) for Aij in Ai] for Ai in A]
        return A

    def _prss_uci(self):
        """Create unique common input for PRSS.

        Increments the program counter to ensure that consecutive calls
        to PRSS-related methods will use unique program counters.
        """
        self._increment_pc()
        return self._program_counter

    def random(self, sftype, max=None):
        """Secure random value of the given type in the given range."""
        return self.randoms(sftype, 1, max)[0]

    def randoms(self, sftype, n, max=None):
        """n secure random values of the given type in the given range."""
        if issubclass(sftype, Share):
            field = sftype.field
        else:
            field = sftype
        if max is None:
            max = field.modulus
        else:
            max = (max - 1) // self._bincoef + 1
        m = len(self.parties)
        prfs = self.parties[self.id].prfs(max)
        shares = thresha.pseudorandom_share(field, m, self.id, prfs, self._prss_uci(), n)
        if issubclass(sftype, Share):
            return [sftype(s) for s in shares]
        else:
            return shares

    def random_bit(self, sftype, signed=False):
        """Secure random bit of the given type."""
        return self.random_bits(sftype, 1, signed)[0]

    @mpc_coro
    async def random_bits(self, sftype, n, signed=False):
        """n secure random bits of the given type."""
        prss0 = False
        f1 = 1
        if issubclass(sftype, Share):
            await returnType((sftype, True), n)
            field = sftype.field
            if sftype.__name__.startswith('SecFld'):
                prss0 = True
            f1 = 1<<sftype.field.frac_length
        else:
            await returnType(Share)
            field = sftype

        bits = [None] * n
        p = field.modulus
        if not signed:
            q = (p + 1) >> 1 # q = 1/2 mod p
        m = len(self.parties)
        t = self.threshold
        prfs = self.parties[self.id].prfs(p)
        h = n
        while h > 0:
            rs = thresha.pseudorandom_share(field, m, self.id, prfs, self._prss_uci(), h)
            # Compute and open the squares and compute square roots.
            r2s = [r * r for r in rs]
            if prss0:
                z = thresha.pseudorandom_share_zero(field, m, self.id, prfs, self._prss_uci(), h)
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
            c[i] = x[i] + y[i] - 2 * c[i] + (c[i - 1] if i >= 1 else 0)
        return c

    @mpc_coro
    async def to_bits(self, a):
        """Secure extraction of bits of a.""" # a la [ST06].
        stype = type(a)
        l = stype.bit_length
        await returnType((stype, True), l)
        Zp = stype.field
        k = self.options.security_parameter

        r_bits = await gather_shares(self.random_bits(Zp, l))
        r_modl = 0
        for i in range(l - 1, -1, -1):
            r_modl <<= 1
            r_modl += r_bits[i].value
        r_divl = self.random(Zp, 1<<k)
        a = await gather_shares(a)
        c = await self.output(a + (1<<l) - r_modl + (1<<l) * r_divl)
        c = c.value % (1<<l)
        r_bits = [stype(r.value) for r in r_bits]
        c_bits = [stype((c >> i) & 1) for i in range(l)]
        return self.add_bits(r_bits, c_bits)

    def _norm(self, a): # signed normalization factor
        stype = type(a)
        l = stype.bit_length
        f = stype.field.frac_length
        x = self.to_bits(a) # low to high bits
        s = x[-1] # sign bit
        def __norm(x):
            n = len(x)
            if n == 1:
                t = self.xor(s, x[0])
                return 2 - t, t
            i0, nz0 = __norm(x[:n//2]) # low bits
            i1, nz1 = __norm(x[n//2:]) # high bits
            i0 *= (1 << ((n + 1) // 2))
            return nz1 * (i1 - i0) + i0, self.or_(nz0, nz1)
        return (1 - 2 * s) * __norm(x[:-1])[0] * (2 ** (f - (l - 1))) # note f <= l

    def _rec(self, a): # enhance performance by reducing no. of truncs
        f = type(a).field.frac_length
        v = self._norm(a)
        b = a * v # 1/2 <= b <= 1
        theta = int(math.ceil(math.log((2 * f + 1) / 3.5, 2)))
        c = 2.9142 - 2 * b
        for _ in range(theta):
            c *= 2 - c * b
        return c * v

class _Party:
    """Information about a party in the MPC protocol."""

    def __init__(self, id, host=None, port=None, keys=None):
        """Initialize a party with given id."""
        self.id = id
        self.host = host
        self.port = port
        self.keys = keys
        self._prfs = {}

    def prfs(self, max):
        """PRFs with codomain range(max) for pseudorandom secret sharing.

        Return a mapping from sets of parties to PRFs.
        """
        try:
            return self._prfs[max]
        except KeyError:
            self._prfs[max] = {}
            for subset, key in self.keys.items():
                self._prfs[max][subset] = thresha.PRF(key, max)
            return self._prfs[max]

    def __repr__(self):
        """String representation of the party."""
        if self.host is None:
            return f'<_Party {self.id}>'
        else:
            return f'<_Party {self.id}: {self.host}:{self.port}>'

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

    Returns the id of the owner and a list of _Party objects.
    """
    config = configparser.ConfigParser()
    config.read_file(open(filename, 'r'))
    m = len(config.sections())
    if t is None:
        t = (m - 1) // 2
    parties = [None] * m
    for party in config.sections():
        id = int(party[6:]) # strip 'Party ' prefix
        host = config.get(party, 'host')
        port = config.getint(party, 'port')
        if len(config.options(party)) > 2:
            # read PRSS keys
            my_id = id
            keys = {}
            for option in config.options(party):
                if not option in ['host', 'port']:
                    subset = frozenset(map(int, option.split()))
                    if len(subset) == m - t:
                        keys[subset] = config.get(party, option)
            parties[my_id] = _Party(my_id, host, port, keys)
        else:
            parties[id] = _Party(id, host, port)
    return my_id, parties

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
    logging_enabled = not options.no_log
    if not logging_enabled:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(stream=sys.stdout,
                            level=logging.INFO, format='%(asctime)s %(message)s')
    if 'gmpy2' not in sys.modules:
        logging.info('Install package gmpy2 for better performance.')
    if not options.config:
        options.no_async = True
        id = 0
        parties = [_Party(id, keys={frozenset([id]): hex(secrets.randbits(128))})]
    else:
        options.config = os.path.join('.config', options.config)
        id, parties = _load_config(options.config, options.threshold)
    m = len(parties)
    if options.threshold is None:
        options.threshold = (m - 1) // 2
    assert 2 * options.threshold < m

    runtime = Runtime(id, parties, options)
    runtime.parser = parser
    runtime.options = options
    runtime.logging_enabled = logging_enabled
    runtime._loop = asyncio.get_event_loop()
    Share.runtime = runtime
    import mpyc.asyncoro
    mpyc.asyncoro.runtime = runtime
    runtime.version = mpyc.__version__
    global mpc
    mpc = runtime

try: # ignore exceptions for pydoc etc.
    setup()
except Exception:
    pass

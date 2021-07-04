"""The MPyC runtime module is used to execute secure multiparty computations.

Parties perform computations on secret-shared values by exchanging messages.
Shamir's threshold secret sharing scheme is used for finite fields of any order
exceeding the number of parties. MPyC provides secure number types and operations,
many of which are available through Python's mechanism for operator overloading.
"""

import os
import sys
import time
import datetime
import importlib.util
import logging
import math
import secrets
import itertools
import functools
import configparser
import argparse
import pickle
import asyncio
import ssl
from mpyc import thresha
from mpyc import sectypes
from mpyc import asyncoro
import mpyc.random
import mpyc.statistics
import mpyc.seclists

Future = asyncio.Future
SecureObject = sectypes.SecureObject
SecureFiniteField = sectypes.SecureFiniteField
mpc_coro = asyncoro.mpc_coro
mpc_coro_no_pc = asyncoro._mpc_coro_no_pc
returnType = asyncoro.returnType


class Runtime:
    """MPyC runtime secure against passive attacks.

    The runtime maintains basic information such as a program counter, the list
    of parties, etc., and handles secret-shared objects of type SecureObject.

    1-party case is supported (with option to disable asynchronous evaluation).
    Threshold 0 (no corrupted parties) is supported for m-party case as well
    to enable distributed computation (without secret sharing).
    """

    version = mpyc.__version__
    random = mpyc.random
    statistics = mpyc.statistics

#    __slots__ = ('pid', 'parties', 'options', '_threshold', '_logging_enabled', '_program_counter',
#                 '_pc_level', '_loop', 'start_time', 'aggregate_load', '_prss_keys', '_bincoef')

    def __init__(self, pid, parties, options):
        """Initialize runtime."""
        self.pid = pid
        self.parties = tuple(parties)
        self.options = options
        self.threshold = options.threshold
        self._logging_enabled = not options.no_log
        self._program_counter = [0, 0]  # [hopping-counter, program-depth]
        self._pc_level = 0  # used for implementation of barriers
        self._loop = asyncio.get_event_loop()  # cache running loop
        self._loop.set_exception_handler(asyncoro.exception_handler)  # exceptions re MPyC coroutine
        self.start_time = None
        self.aggregate_load = 0.0 * 10000  # unit: basis point 0.0001 = 0.01%

    @property
    def threshold(self):
        """Threshold for MPC."""
        return self._threshold

    @threshold.setter
    def threshold(self, t):
        m = len(self.parties)
        self._threshold = t
        # caching (m choose t):
        self._bincoef = math.factorial(m) // math.factorial(t) // math.factorial(m - t)
        if self.options.no_prss:
            return

        # generate new PRSS keys
        self.prfs.cache_clear()
        keys = {}
        for subset in itertools.combinations(range(m), m - t):
            if subset[0] == self.pid:
                keys[subset] = secrets.token_bytes(16)  # 128-bit key
        self._prss_keys = keys

    @functools.lru_cache(maxsize=None)
    def prfs(self, bound):
        """PRFs with codomain range(bound) for pseudorandom secret sharing.

        Return a mapping from sets of parties to PRFs.
        """
        if self.options.no_prss:
            raise NotImplementedError('Functionality not (yet) supported when PRSS is disabled.')

        f = {}
        for subset, key in self._prss_keys.items():
            f[subset] = thresha.PRF(key, bound)
        return f

    def _send_message(self, peer_pid, data):
        """Send data to given peer, labeled by current program counter."""
        self.parties[peer_pid].protocol.send(self._program_counter[0], data)

    def _receive_message(self, peer_pid):
        """Receive data from given peer, labeled by current program counter."""
        return self.parties[peer_pid].protocol.receive(self._program_counter[0])

    def _exchange_shares(self, in_shares):
        pc = self._program_counter[0]
        out_shares = [None] * len(in_shares)
        for peer_pid, data in enumerate(in_shares):
            if peer_pid != self.pid:
                protocol = self.parties[peer_pid].protocol
                protocol.send(pc, data)
                data = protocol.receive(pc)
            out_shares[peer_pid] = data
        return out_shares

    def gather(self, *obj):
        return asyncoro.gather_shares(self, *obj)

    async def barrier(self, name=None):
        """Barrier for runtime, using optional string name for easy identification."""
        if self.options.no_barrier:
            return

        name = f'-{name}' if name else ''
        logging.info(f'Barrier{name} {self._pc_level} {self._program_counter[1]}')
        if not self.options.no_async:
            while self._pc_level > self._program_counter[1]:
                await asyncio.sleep(0)

    async def throttler(self, load_percentage=1.0, name=None):
        """Throttle runtime by given percentage (default 1.0), using optional name for barrier."""
        assert 0.0 <= load_percentage <= 1.0, 'percentage as decimal fraction between 0.0 and 1.0'
        self.aggregate_load += load_percentage * 10000
        if self.aggregate_load < 10000:
            return

        self.aggregate_load -= 10000
        await mpc.barrier(name=name)

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
        loop = self._loop
        for peer in self.parties:
            peer.protocol = Future(loop=loop) if peer.pid == self.pid else None
        if self.options.ssl:
            crtfile = os.path.join('.config', f'party_{self.pid}.crt')
            keyfile = os.path.join('.config', f'party_{self.pid}.key')
            cafile = os.path.join('.config', 'mpyc_ca.crt')

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
            factory = lambda: asyncoro.MessageExchanger(self)
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
                    factory = lambda: asyncoro.MessageExchanger(self, peer.pid)
                    await loop.create_connection(factory, peer.host, peer.port, ssl=context,
                                                 server_hostname=server_hostname)
                    break
                except asyncio.CancelledError:
                    raise

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
        """Shutdown the MPyC runtime.

        Close all connections, if any.
        """
        # Wait for all parties behind a barrier.
        while self._pc_level > self._program_counter[1]:
            await asyncio.sleep(0)
        m = len(self.parties)
        if m > 1:
            await self.gather(self.transfer(self.pid))
            # Close connections to all parties.
            for peer in self.parties:
                if peer.pid != self.pid:
                    peer.protocol.close_connection()

        elapsed = time.time() - self.start_time
        logging.info(f'Stop MPyC runtime -- elapsed time: {datetime.timedelta(seconds=elapsed)}')

    async def __aenter__(self):
        """Start MPyC runtime when entering async with context."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Shutdown MPyC runtime when exiting async with context."""
        if exc:
            # Limited shutdown: only close connections to all parties.
            for peer in self.parties:
                if peer.pid != self.pid:
                    peer.protocol.close_connection()
            return

        await self.shutdown()

    Object = sectypes.SecureObject
    Number = sectypes.SecureNumber
    FiniteField = sectypes.SecureFiniteField
    Integer = sectypes.SecureInteger
    FixedPoint = sectypes.SecureFixedPoint
    Float = sectypes.SecureFloat
    SecFld = staticmethod(sectypes.SecFld)
    SecInt = staticmethod(sectypes.SecInt)
    SecFxp = staticmethod(sectypes.SecFxp)
    SecFlt = staticmethod(sectypes.SecFlt)
    coroutine = staticmethod(mpc_coro)
    returnType = staticmethod(returnType)

    @mpc_coro
    async def transfer(self, obj, senders=None, receivers=None, sender_receivers=None) -> Future:
        """Transfer pickable Python objects between specified parties.

        The senders are the parties that provide input.
        The receivers are the parties that will obtain output.
        The default is to let every party be a sender as well as a receiver.

        The (directed) communication graph specifying which parties sends their message
        given as obj to which receivers is represented by:

         - either the senders/receivers arguments for a complete bipartite graph,
         - or the sender_receivers argument for an arbitrary graph.

        Each party i corresponds to a node in the communication graph.
        The senders/receivers arguments represent subsets of nodes, in the form
        of a list, a Python range object, or a Python int.
        The sender_receivers argument represents a set of arcs, in the form of a
        list of node pairs or as a Python dict mapping nodes to subsets of nodes.
        """
        assert (senders is None and receivers is None) or sender_receivers is None
        senders_is_int = isinstance(senders, int)
        if sender_receivers is None:
            m = len(self.parties)
            if senders is None:
                senders = range(m)  # default
            senders = [senders] if senders_is_int else list(senders)
            if receivers is None:
                receivers = range(m)  # default
            receivers = [receivers] if isinstance(receivers, int) else list(receivers)
            my_senders = senders if self.pid in receivers else []
            my_receivers = receivers if self.pid in senders else []
        else:
            if isinstance(sender_receivers, dict):
                my_senders = [a for a, b in sender_receivers.items() if self.pid in b]
                my_receivers = list(sender_receivers[self.pid])
            else:
                my_senders = [a for a, b in sender_receivers if b == self.pid]
                my_receivers = [b for a, b in sender_receivers if a == self.pid]

        indata = None
        for peer_pid in my_receivers:
            if indata is None:
                indata = pickle.dumps(obj)
            if peer_pid != self.pid:
                self._send_message(peer_pid, indata)

        outdata = [None] * len(my_senders)
        for i, peer_pid in enumerate(my_senders):
            if peer_pid == self.pid:
                outdata[i] = indata
            else:
                outdata[i] = self._receive_message(peer_pid)
        outdata = await self.gather(outdata)
        outdata = list(map(pickle.loads, outdata))
        if senders_is_int:
            outdata = outdata[0]
        return outdata

    def input(self, x, senders=None):
        """Input x to the computation.

        Value x is a secure object, or a list of secure objects.
        The senders are the parties that provide an input.
        The default is to let every party be a sender.

        Except for secure integers, secure fixed-point numbers, and
        secure finite field elements, which are handled directly, the
        input of secure objects is controlled by their _input() method.
        For instance, mpyc.sectypes.SecureFloat._input() does this for
        secure floating-point numbers.
        """
        x_is_list = isinstance(x, list)
        if x_is_list:
            x = x[:]
        else:
            x = [x]
        senders_is_int = isinstance(senders, int)
        if senders is None:
            m = len(self.parties)
            senders = range(m)  # default
        senders = [senders] if senders_is_int else list(senders)
        y = self._distribute(x, senders)
        if senders_is_int:
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
        field = getattr(stype, 'field', None)  # TODO: avoid this use of 'field' attr
        if not field:
            return stype._input(x, senders)

        if not stype.frac_length:
            await returnType(stype, len(senders), len(x))
        else:
            await returnType((stype, x[0].integral), len(senders), len(x))

        shares = [None] * len(senders)
        for i, peer_pid in enumerate(senders):
            if peer_pid == self.pid:
                x = await self.gather(x)
                t = self.threshold
                m = len(self.parties)
                in_shares = thresha.random_split(x, t, m)
                for other_pid, data in enumerate(in_shares):
                    data = field.to_bytes(data)
                    if other_pid == self.pid:
                        shares[i] = data
                    else:
                        self._send_message(other_pid, data)
            else:
                shares[i] = self._receive_message(peer_pid)
        shares = await self.gather(shares)
        return [[field(a) for a in field.from_bytes(r)] for r in shares]

    @mpc_coro
    async def output(self, x, receivers=None, threshold=None, raw=False) -> Future:
        """Output the value of x to the receivers specified.

        Value x is a secure object, or a list of secure objects.
        The receivers are the parties that will obtain the output.
        The default is to let every party be a receiver.

        A secure integer is output as a Python int, a secure
        fixed-point number is output as a Python float, and a secure
        finite field element is output as an MPyC finite field element.
        Set flag raw=True to suppress output conversion.

        For all other types of secure objects their _output() method controls
        what is output. For instance, mpyc.sectypes.SecureFloat._output()
        outputs secure floating-point numbers as Python floats.
        The flag raw is ignored for these types.
        """
        x_is_list = isinstance(x, list)
        if x_is_list:
            x = x[:]
        else:
            x = [x]
        if x == []:
            return []

        t = self.threshold if threshold is None else threshold
        m = len(self.parties)
        if receivers is None:
            receivers = range(m)  # default
        receivers = [receivers] if isinstance(receivers, int) else list(receivers)
        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, SecureObject):
            field = getattr(sftype, 'field', None)  # TODO: avoid this use of 'field' attr
            if not field:
                y = await sftype._output(x, receivers, threshold)
                if not x_is_list:
                    y = y[0]
                return y

            x = await self.gather(x)
        else:
            field = sftype
        x = [a.value for a in x]

        # Send share to all successors in receivers.
        for peer_pid in receivers:
            if 0 < (peer_pid - self.pid) % m <= t:
                self._send_message(peer_pid, field.to_bytes(x))
        # Receive and recombine shares if this party is a receiver.
        if self.pid in receivers:
            shares = [self._receive_message((self.pid - t + j) % m) for j in range(t)]
            shares = await self.gather(shares)
            points = [((self.pid - t + j) % m + 1, field.from_bytes(shares[j])) for j in range(t)]
            points.append((self.pid + 1, x))
            y = thresha.recombine(field, points)
            if issubclass(sftype, SecureObject):
                f = sftype._output_conversion
                if not raw and f is not None:
                    y = [f(a) for a in y]
        else:
            y = [None] * len(x)
        if not x_is_list:
            y = y[0]
        return y

    @mpc_coro
    async def _reshare(self, x):
        x_is_list = isinstance(x, list)
        if not x_is_list:
            x = [x]
        if x == []:
            return []

        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, SecureObject):
            if not sftype.frac_length:
                await returnType(sftype, len(x))
            else:
                if x_is_list:
                    await returnType((sftype, x[0].integral), len(x))
                else:
                    await returnType((sftype, x[0].integral))
            x = await self.gather(x)
            field = sftype.field
        else:
            await returnType(Future)
            field = sftype

        t = self.threshold
        m = len(self.parties)
        in_shares = thresha.random_split(x, t, m)
        in_shares = [field.to_bytes(elts) for elts in in_shares]
        # Recombine the first 2t+1 output_shares.
        out_shares = await self.gather(self._exchange_shares(in_shares)[:2*t+1])
        points = [(j+1, field.from_bytes(s)) for j, s in enumerate(out_shares)]
        y = thresha.recombine(field, points)

        if issubclass(sftype, SecureObject):
            y = [sftype(s) for s in y]
        if not x_is_list:
            y = y[0]
        return y

    def convert(self, x, ttype):
        """Secure conversion of (elements of) x to given ttype.

        Value x is a secure number, or a list of secure numbers.
        Converted values assumed to fit in target type.
        """
        x_is_list = isinstance(x, list)
        if x_is_list:
            x = x[:]
        else:
            x = [x]
        if x == []:
            return []

        if (isinstance(x[0], sectypes.SecureFiniteField)
                and issubclass(ttype, sectypes.SecureFiniteField)):
            # conversion via secure integers
            stype = type(x[0])
            size = max(stype.field.order, ttype.field.order)
            l = max(32, size.bit_length())
            secint = self.SecInt(l=l)
            y = self._convert(self._convert(x, secint), ttype)
        else:
            y = self._convert(x, ttype)

        if not x_is_list:
            y = y[0]
        return y

    @mpc_coro
    async def _convert(self, x, ttype):
        stype = type(x[0])  # source type
        n = len(x)
        await returnType((ttype, not stype.frac_length), n)  # target type
        m = len(self.parties)
        k = self.options.sec_param
        l = min(stype.bit_length, ttype.bit_length)
        if issubclass(stype, sectypes.SecureFiniteField):
            bound = stype.field.order
        else:
            bound = (1<<(k + l)) // self._bincoef + 1
        prfs = self.prfs(bound)
        uci = self._prss_uci()  # NB: same uci in both calls for r below

        x = await self.gather(x)
        d = ttype.frac_length - stype.frac_length  # TODO: use integral attribute fxp
        if d < 0:
            x = await self.trunc(x, f=-d, l=stype.bit_length)  # TODO: take minimum with ttype or so
        offset = 1 << l-1
        r = thresha.pseudorandom_share(stype.field, m, self.pid, prfs, uci, n)
        for i in range(n):
            x[i] = x[i].value + offset + r[i]

        x = await self.output(x)
        r = thresha.pseudorandom_share(ttype.field, m, self.pid, prfs, uci, n)
        for i in range(n):
            x[i] = x[i].value - r[i]
            if issubclass(stype, sectypes.SecureFiniteField):
                x[i] = self._mod(ttype(x[i]), stype.field.modulus)
            x[i] = x[i] - offset
        if d > 0 and not issubclass(stype, sectypes.SecureFiniteField):
            for i in range(n):
                x[i] <<= d
        return x

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
        if issubclass(sftype, SecureObject):
            if x_is_list:
                await returnType(sftype, n)
            else:
                await returnType(sftype)
            Zp = sftype.field
            l = sftype.bit_length
            if f is None:
                f = sftype.frac_length
        else:
            await returnType(Future)
            Zp = sftype

        k = self.options.sec_param
        r_bits = await self.random_bits(Zp, f * n)
        r_modf = [None] * n
        for j in range(n):
            s = 0
            for i in range(f-1, -1, -1):
                s <<= 1
                s += r_bits[f * j + i].value
            r_modf[j] = Zp(s)
        r_divf = self._randoms(Zp, n, 1 << k + l)
        if issubclass(sftype, SecureObject):
            x = await self.gather(x)
        c = await self.output([a + ((1 << l - 1 + f) + (q.value << f) + r.value)
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
        if issubclass(stype, sectypes.SecureFloat):
            return await stype.is_zero_public(a)

        field = stype.field
        m = len(self.parties)
        t = self.threshold
        if field.order.bit_length() <= 60:  # TODO: introduce MPyC parameter for failure probability
            prfs = self.prfs(field.order)
            while True:
                r, s = self._randoms(field, 2)
                z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), 1)
                if await self.output(r * s + z[0], threshold=2*t):
                    break
        else:
            r = self._random(field)  # NB: failure r=0 with probability less than 2**-60
        a = await self.gather(a)
        if field.order.bit_length() <= 60:
            z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), 1)
            b = a * r + z[0]
        else:
            b = a * r
        c = await self.output(b, threshold=2*t)
        return c == 0

    @mpc_coro_no_pc
    async def neg(self, a):
        """Secure negation (additive inverse) of a."""
        stype = type(a)
        if not stype.frac_length:
            await returnType(stype)
        else:
            await returnType((stype, a.integral))
        a = await self.gather(a)
        return -a

    @mpc_coro_no_pc
    async def pos(self, a):
        """Secure unary + applied to a."""
        stype = type(a)
        if not stype.frac_length:
            await returnType(stype)
        else:
            await returnType((stype, a.integral))
        a = await self.gather(a)
        return +a

    @mpc_coro_no_pc
    async def add(self, a, b):
        """Secure addition of a and b."""
        stype = type(a)
        if not stype.frac_length:
            await returnType(stype)
        else:
            await returnType((stype, a.integral and b.integral))
        a, b = await self.gather(a, b)
        return a + b

    @mpc_coro_no_pc
    async def sub(self, a, b):
        """Secure subtraction of a and b."""
        stype = type(b) if isinstance(b, SecureObject) else type(a)
        if not stype.frac_length:
            await returnType(stype)
        else:
            await returnType((stype, a.integral and b.integral))
        a, b = await self.gather(a, b)
        return a - b

    @mpc_coro
    async def mul(self, a, b):
        """Secure multiplication of a and b."""
        stype = type(a)
        f = stype.frac_length
        if not f:
            await returnType(stype)
        else:
            a_integral = a.integral
            b_integral = isinstance(b, int) or isinstance(b, SecureObject) and b.integral
            if isinstance(b, float):
                b = round(b * 2**f)
            await returnType((stype, a_integral and b_integral))

        shb = isinstance(b, SecureObject)
        if not shb:
            a = await self.gather(a)
        elif a is b:
            a = b = await self.gather(a)
        else:
            a, b = await self.gather(a, b)
        if f and b_integral:
            a, b = b, a
        c = a * b
        if f and (a_integral or b_integral) and not isinstance(a, int):
            c >>= f  # NB: in-place rshift
        if shb:
            c = self._reshare(c)
        if f and not (a_integral or b_integral):
            c = self.trunc(stype(c))  # TODO: compare to approach in in_prod
        return c

    def div(self, a, b):
        """Secure division of a by b, for nonzero b."""
        b_is_SecureObject = isinstance(b, SecureObject)
        stype = type(b) if b_is_SecureObject else type(a)
        field = stype.field
        f = stype.frac_length
        if b_is_SecureObject:
            if f:
                c = self._rec(b)
            else:
                c = self.reciprocal(b)
            return self.mul(c, a)

        # isinstance(a, SecureObject) ensured
        if f:
            if isinstance(b, (int, float)):
                c = 1/b
                if c.is_integer():
                    c = round(c)
            else:
                c = b.reciprocal() << f
        else:
            if not isinstance(b, field):
                b = field(b)
            c = b.reciprocal()
        return self.mul(a, c)

    @mpc_coro
    async def reciprocal(self, a):
        """Secure reciprocal (multiplicative inverse) of a, for nonzero a."""
        stype = type(a)
        field = stype.field
        await returnType(stype)
        a = await self.gather(a)
        while True:
            r = self._random(field)
            ar = await self.output(a * r, threshold=2*self.threshold)
            if ar:
                break
        r <<= stype.frac_length
        return r / ar

    def pow(self, a, b):
        """Secure exponentiation a raised to the power of b, for public integer b."""
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

    def lt(self, a, b):
        """Secure comparison a < b."""
        return self.sgn(a - b, LT=True)

    def eq(self, a, b):
        """Secure comparison a == b."""
        return self.is_zero(a - b)

    def ge(self, a, b):
        """Secure comparison a >= b."""
        return 1 - self.sgn(a - b, LT=True)

    def abs(self, a):
        """Secure absolute value of a."""
        return (-2*self.sgn(a, LT=True) + 1) * a

    def is_zero(self, a):
        """Secure zero test a == 0."""
        if isinstance(a, SecureFiniteField):
            return 1 - self.pow(a, a.field.order - 1)

        if a.bit_length/2 > self.options.sec_param >= 8 and a.field.order%4 == 3:
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
        a, u2, z = await self.gather(a, u2, z)
        a = a.value
        r = self._randoms(Zp, k)
        c = [Zp(a * r[i].value + (1-(z[i].value << 1)) * u2[i].value) for i in range(k)]
        # -1 is nonsquare for Blum p, u_i !=0 w.v.h.p.
        # If a == 0, c_i is square mod p iff z[i] == 0.
        # If a != 0, c_i is square mod p independent of z[i].
        c = await self.output(c, threshold=2*self.threshold)
        for i in range(k):
            if c[i] == 0:
                c[i] = Zp(1)
            else:
                c[i] = 1-z[i] if c[i].is_sqr() else z[i]
        e = await self.all(c)
        e <<= stype.frac_length
        return e

    @mpc_coro
    async def sgn(self, a, l=None, LT=False, EQ=False):
        """Secure sign(um) of a, return -1 if a < 0 else 0 if a == 0 else 1.

        If integer flag l=L is set, it is assumed that -2^(L-1) <= a < 2^(L-1)
        to save work (compared to the default l=type(a).bit_length).

        If Boolean flag LT is set, perform a secure less than zero test instead, and
        return 1 if a < 0 else 0, saving the work for a secure equality test.
        If Boolean flag EQ is set, perform a secure equal to zero test instead, and
        return 1 if a == 0 else 0, saving the work for a secure comparison.
        """
        assert not (LT and EQ)
        stype = type(a)
        await returnType((stype, True))
        Zp = stype.field

        l = l or stype.bit_length
        r_bits = await self.random_bits(Zp, l)
        r_modl = 0
        for r_i in reversed(r_bits):
            r_modl <<= 1
            r_modl += r_i.value
        a = await self.gather(a)
        a_rmodl = a + ((1<<l) + r_modl)
        k = self.options.sec_param
        r_divl = self._random(Zp, 1<<k).value
        c = await self.output(a_rmodl + (r_divl << l))
        c = c.value % (1<<l)

        if not EQ:  # a la Toft
            s_sign = (await self.random_bits(Zp, 1, signed=True))[0].value
            e = [None] * (l+1)
            sumXors = 0
            for i in range(l-1, -1, -1):
                c_i = (c >> i) & 1
                r_i = r_bits[i].value
                e[i] = Zp(s_sign + r_i - c_i + 3*sumXors)
                sumXors += 1 - r_i if c_i else r_i
            e[l] = Zp(s_sign - 1 + 3*sumXors)
            g = await self.is_zero_public(stype(self.prod(e)))
            h = 3 - s_sign if g else 3 + s_sign
            z = (c - a_rmodl + (h << l-1)) / (1<<l)

        if not LT:
            h = self.all(r_bits[i] if (c >> i) & 1 else 1-r_bits[i] for i in range(l))
            h = await h
            if EQ:
                z = h
            else:
                z = (h - 1) * (2*z - 1)
                z = await self._reshare(z)

        z <<= stype.frac_length
        return z

    def min(self, *x, key=None):
        """Secure minimum of all given elements in x, similar to Python's built-in min().

        See runtime.sorted() for details on key etc.
        """
        if len(x) == 1:
            x = x[0]
        if iter(x) is x:
            x = list(x)
        n = len(x)
        if not n:
            raise ValueError('min() arg is an empty sequence')

        if n == 1:
            return x[0]

        if key is None:
            key = lambda a: a
        min0 = self.min(x[:n//2], key=key)
        min1 = self.min(x[n//2:], key=key)
        return self.if_else(key(min0) < key(min1), min0, min1)

    def max(self, *x, key=None):
        """Secure maximum of all given elements in x, similar to Python's built-in max().

        See runtime.sorted() for details on key etc.
        """
        if len(x) == 1:
            x = x[0]
        if iter(x) is x:
            x = list(x)
        n = len(x)
        if not n:
            raise ValueError('max() arg is an empty sequence')

        if n == 1:
            return x[0]

        if key is None:
            key = lambda a: a
        max0 = self.max(x[:n//2], key=key)
        max1 = self.max(x[n//2:], key=key)
        return self.if_else(key(max0) < key(max1), max1, max0)

    def min_max(self, *x, key=None):
        """Secure minimum and maximum of all given elements in x.

        Saves 25% compared to calling min(x) and max(x) separately.
        Total number of comparisons is only (3n-3)//2, compared to 2n-2 for the obvious approach.
        This is optimal as shown by Ira Pohl in "A sorting problem and its complexity",
        Communications of the ACM 15(6), pp. 462-464, 1972.
        """
        if len(x) == 1:
            x = x[0]
        x = list(x)
        n = len(x)
        if not n:
            raise ValueError('min_max() arg is an empty sequence')

        if key is None:
            key = lambda a: a
        for i in range(n//2):
            a, b = x[i], x[-1-i]
            x[i], x[-1-i] = self.if_swap(a >= b, a, b)
        # NB: x[n//2] both in x[:(n+1)//2] and in x[n//2:] if n odd
        return self.min(x[:(n+1)//2], key=key), self.max(x[n//2:], key=key)

    def argmin(self, *x, key=None):
        """Secure argmin of all given elements in x.

        See runtime.sorted() for details on key etc.
        """
        if len(x) == 1:
            x = x[0]
        if iter(x) is x:
            x = list(x)
        n = len(x)
        if not n:
            raise ValueError('argmin() arg is an empty sequence')

        if key is None:
            key = lambda a: a
        return self._argmin(x, key)

    def _argmin(self, x, key):
        n = len(x)
        if n == 1:
            m = x[0]
            stype = type(m[0]) if isinstance(m, list) else type(m)
            return stype(0), m  # NB: sets integral attr to True for SecureFixedPoint numbers

        i0, min0 = self._argmin(x[:n//2], key)
        i1, min1 = self._argmin(x[n//2:], key)
        i1 += n//2
        c = key(min0) < key(min1)
        a = self.if_else(c, i0, i1)
        m = self.if_else(c, min0, min1)  # TODO: merge if_else's once integral attr per list element
        return a, m

    def argmax(self, *x, key=None):
        """Secure argmax of all given elements in x.

        See runtime.sorted() for details on key etc.
        """
        if len(x) == 1:
            x = x[0]
        if iter(x) is x:
            x = list(x)
        n = len(x)
        if not n:
            raise ValueError('argmax() arg is an empty sequence')

        if key is None:
            key = lambda a: a
        return self._argmax(x, key)

    def _argmax(self, x, key):
        n = len(x)
        if n == 1:
            m = x[0]
            stype = type(m[0]) if isinstance(m, list) else type(m)
            return stype(0), m  # NB: sets integral attr to True for SecureFixedPoint numbers

        i0, max0 = self._argmax(x[:n//2], key)
        i1, max1 = self._argmax(x[n//2:], key)
        i1 += n//2
        c = key(max0) < key(max1)
        a = self.if_else(c, i1, i0)
        m = self.if_else(c, max1, max0)  # TODO: merge if_else's once integral attr per list element
        return a, m

    def sorted(self, x, key=None, reverse=False):
        """Return a new securely sorted list with elements from x in ascending order.

        Similar to Python's built-in sorted(), but not stable.

        Elements of x are either secure numbers or lists of secure numbers.

        Argument key specifies a function applied to all elements of x before comparing them, using
        only < comparisons (that is, using only the __lt__() method, as for Python's list.sort()).
        Default key compares elements of x directly (using identity function 'lambda a: a').
        """
        x = list(x)
        n = len(x)
        if n < 2:
            return x

        if key is None:
            key = lambda a: a
        self._sort(x, key)  # TODO: stable sort &  vectorization of <'s
        if reverse:
            x.reverse()
        return x

    def _sort(self, x, key):
        """Batcher's merge-exchange sort, see Knuth TAOCP Algorithm 5.2.2M.

        In-place sort in roughly 1/2(log_2 n)^2 rounds of about n/2 comparions each.
        """
        n = len(x)  # n >= 2
        t = (n-1).bit_length()
        p = 1 << t-1
        while p:
            d, q, r = p, 1 << t-1, 0
            while d:
                for i in range(n - d):  # NB: all n-d comparisons can be done in parallel
                    if i & p == r:
                        a, b = x[i], x[i + d]
                        x[i], x[i + d] = mpc.if_swap(key(a) >= key(b), a, b)
                d, q, r = q - p, q >> 1, p
            p >>= 1
        return x

    @mpc_coro
    async def lsb(self, a):
        """Secure least significant bit of a."""  # a la [ST06]
        stype = type(a)
        await returnType((stype, True))
        Zp = stype.field
        l = stype.bit_length
        k = self.options.sec_param
        f = stype.frac_length

        b = self.random_bit(stype)
        a, b = await self.gather(a, b)
        b >>= f
        r = self._random(Zp, 1 << (l + k - 1)).value
        c = await self.output(a + ((1<<l) + (r << 1) + b.value))
        x = 1 - b if c.value & 1 else b  # xor
        x <<= f
        return x

    @mpc_coro_no_pc
    async def mod(self, a, b):
        """Secure modulo reduction."""
        # TODO: optimize for integral a of type secfxp
        stype = type(a)
        await returnType(stype)
        b = await self.gather(b)
        b = b.value
        assert isinstance(b, int)
        if b == 2:
            r = self.lsb(a)
        elif not b & (b-1):
            r = self.from_bits(self.to_bits(a, b.bit_length() - 1))
        else:
            r = self._mod(a, b)
        f = stype.frac_length
        return r * 2**-f

    @mpc_coro
    async def _mod(self, a, b):
        """Secure modulo reduction, for public b."""  # a la [GMS10]
        stype = type(a)
        await returnType(stype)
        Zp = stype.field
        l = stype.bit_length
        k = self.options.sec_param
        f = stype.frac_length

        r_bits = self.random._randbelow(stype, b, bits=True)
        a, r_bits = await self.gather(a, r_bits)
        r_bits = [(r >> f).value for r in r_bits]
        r_modb = 0
        for r_i in reversed(r_bits):
            r_modb <<= 1
            r_modb += r_i
        r_modb = Zp(r_modb)
        r_divb = self._random(Zp, 1 << k).value
        c = await self.output(a + ((1<<l) - ((1<<l) % b) + b * r_divb - r_modb.value))
        c = c.value % b
        c_bits = [(c >> i) & 1 for i in range(len(r_bits))]
        c_bits.append(0)
        r_bits = [stype(r) for r in r_bits]
        r_bits.append(stype(0))
        z = stype(r_modb - (b - c)) >= 0  # TODO: avoid full comparison (use r_bits)
        return self.from_bits(self.add_bits(r_bits, c_bits)) - z * b

    @mpc_coro_no_pc
    async def sum(self, x, start=0):
        """Secure sum of all elements in x, similar to Python's built-in sum()."""
        if iter(x) is x:
            x = list(x)
        else:
            x = x[:]
        if x == []:
            return start

        x[0] = x[0] + start  # NB: also updates x[0].integral if applicable
        stype = type(x[0])  # all elts assumed of same type
        if not stype.frac_length:
            await returnType(stype)
        else:
            await returnType((stype, all(a.integral for a in x)))

        x = await self.gather(x)
        s = sum(a.value for a in x)
        return stype.field(s)

    @mpc_coro  # no_pc possible if no reshare and no trunc
    async def in_prod(self, x, y):
        """Secure dot product of x and y (one resharing)."""
        if x == []:
            return 0

        if x is y:
            x = x[:]
            y = x
        else:
            x, y = x[:], y[:]
        shx = isinstance(x[0], SecureObject)
        shy = isinstance(y[0], SecureObject)
        stype = type(x[0]) if shx else type(y[0])
        f = stype.frac_length
        if not f:
            await returnType(stype)
        else:
            x_integral = all(a.integral for a in x)
            y_integral = all(a.integral for a in y)
            await returnType((stype, x_integral and y_integral))

        if x is y:
            x = y = await self.gather(x)
        elif shx and shy:
            x, y = await self.gather(x, y)
        elif shx:
            x = await self.gather(x)
        else:
            y = await self.gather(y)
        s = sum(a.value * b.value for a, b in zip(x, y))
        s = stype.field(s)
        if f:
            if x_integral or y_integral:
                s >>= f
            else:
                s = stype(s)
        if shx and shy:
            s = self._reshare(s)
        if f and not x_integral and not y_integral:
            s = self.trunc(s)  # TODO: compare to approach in mul
        return s

    @mpc_coro
    async def prod(self, x, start=1):
        """Secure product of all elements in x, similar to Python's math.prod().

        Runs in log_2 len(x) rounds).
        """
        if iter(x) is x:
            x = list(x)
        else:
            x = x[:]
        if x == []:
            return start

        x[0] = x[0] * start  # NB: also updates x[0].integral if applicable
        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, SecureObject):
            f = sftype.frac_length
            if not f:
                await returnType(sftype)
            else:
                integral = [a.integral for a in x]
                await returnType((sftype, all(integral)))
            x = await self.gather(x)
        else:
            f = 0
            await returnType(Future)

        n = len(x)
        while n > 1:
            h = [x[i] * x[i+1] for i in range(n%2, n, 2)]
            x[n%2:] = await self._reshare(h)
            if f:
                z = []
                for i in range(n%2, n, 2):
                    j = (n%2 + i)//2
                    if not integral[i] and not integral[i+1]:
                        z.append(x[j])  # will be truncated
                    else:
                        x[j] >>= f  # NB: in-place rshift
                if z:
                    z = await self.trunc(z, f=f, l=sftype.bit_length)
                    for i in reversed(range(n%2, n, 2)):
                        j = (n%2 + i)//2
                        if not integral[i] and not integral[i+1]:
                            x[j] = z.pop()
                integral[n%2:] = [integral[i] and integral[i+1] for i in range(n%2, n, 2)]
            n = len(x)
        return x[0]

    @mpc_coro
    async def all(self, x):
        """Secure all of elements in x, similar to Python's built-in all().

        Elements of x are assumed to be either 0 or 1 (Boolean).
        Runs in log_2 len(x) rounds.
        """
        if iter(x) is x:
            x = list(x)
        else:
            x = x[:]
        if x == []:
            return 1

        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, SecureObject):
            f = sftype.frac_length
            if not f:
                await returnType(sftype)
            else:
                if not all(a.integral for a in x):
                    raise ValueError('nonintegral fixed-point number')

                await returnType((sftype, True))
            x = await self.gather(x)
        else:
            f = 0
            await returnType(Future)

        n = len(x)  # TODO: for sufficiently large n use mpc.eq(mpc.sum(x), n) instead
        while n > 1:
            h = [x[i] * x[i+1] for i in range(n%2, n, 2)]
            if f:
                for a in h:
                    a >>= f  # NB: in-place rshift
            h = await self._reshare(h)
            x[n%2:] = h
            n = len(x)
        return x[0]

    def any(self, x):
        """Secure any of elements in x, similar to Python's built-in any().

        Elements of x are assumed to be either 0 or 1 (Boolean).
        Runs in log_2 len(x) rounds.
        """
        return 1 - self.all(1-a for a in x)

    @mpc_coro_no_pc
    async def vector_add(self, x, y):
        """Secure addition of vectors x and y."""
        if x == []:
            return []

        x, y = x[:], y[:]
        stype = type(x[0])  # all elts assumed of same type
        n = len(x)
        if not stype.frac_length:
            await returnType(stype, n)
        else:
            y0_integral = isinstance(y[0], int) or isinstance(y[0], SecureObject) and y[0].integral
            await returnType((stype, x[0].integral and y0_integral), n)

        x, y = await self.gather(x, y)
        for i in range(n):
            x[i] = x[i] + y[i]
        return x

    @mpc_coro_no_pc
    async def vector_sub(self, x, y):
        """Secure subtraction of vectors x and y."""
        if x == []:
            return []

        x, y = x[:], y[:]
        stype = type(x[0])  # all elts assumed of same type
        n = len(x)
        if not stype.frac_length:
            await returnType(stype, n)
        else:
            y0_integral = isinstance(y[0], int) or isinstance(y[0], SecureObject) and y[0].integral
            await returnType((stype, x[0].integral and y0_integral), n)

        x, y = await self.gather(x, y)
        for i in range(n):
            x[i] = x[i] - y[i]
        return x

    @mpc_coro_no_pc
    async def matrix_add(self, A, B, tr=False):
        """Secure addition of matrices A and (transposed) B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        n1, n2 = len(A), len(A[0])
        await returnType(type(A[0][0]), n1, n2)
        A, B = await self.gather(A, B)
        for i in range(n1):
            for j in range(n2):
                A[i][j] = A[i][j] + (B[j][i] if tr else B[i][j])
        return A

    @mpc_coro_no_pc
    async def matrix_sub(self, A, B, tr=False):
        """Secure subtraction of matrices A and (transposed) B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        n1, n2 = len(A), len(A[0])
        await returnType(type(A[0][0]), n1, n2)
        A, B = await self.gather(A, B)
        for i in range(n1):
            for j in range(n2):
                A[i][j] = A[i][j] - (B[j][i] if tr else B[i][j])
        return A

    @mpc_coro
    async def scalar_mul(self, a, x):
        """Secure scalar multiplication of scalar a with vector x."""
        if x == []:
            return []

        x = x[:]
        n = len(x)
        stype = type(a)  # a and all elts of x assumed of same type
        f = stype.frac_length
        if not f:
            await returnType(stype, n)
        else:
            a_integral = a.integral
            await returnType((stype, a_integral and x[0].integral), n)

        a, x = await self.gather(a, x)
        if f and a_integral:
            a = a >> f  # NB: no in-place rshift!
        for i in range(n):
            x[i] = x[i] * a
        x = await self._reshare(x)
        if f and not a_integral:
            x = self.trunc(x, f=f, l=stype.bit_length)
            x = await self.gather(x)
        return x

    @mpc_coro
    async def _if_else_list(self, a, x, y):
        x, y = x[:], y[:]
        n = len(x)
        stype = type(a)  # all elts of x and y assumed of same type
        field = stype.field
        f = stype.frac_length
        if not f:
            await returnType(stype, n)
        else:  # NB: a is integral
            await returnType((stype, x[0].integral and y[0].integral), n)

        a, x, y = await self.gather(a, x, y)
        if f:
            a = a >> f  # NB: no in-place rshift!
        a = a.value
        for i in range(n):
            x[i] = field(a * (x[i].value - y[i].value) + y[i].value)
        x = await self._reshare(x)
        return x

    def if_else(self, c, x, y):
        '''Secure selection between x and y based on condition c.'''
        if isinstance(c, sectypes.SecureFixedPoint) and not c.integral:
            raise ValueError('condition must be integral')

        if x is y:  # introduced for github.com/meilof/oblif
            return x

        if isinstance(x, list):
            z = self._if_else_list(c, x, y)
        else:
            z = c * (x - y) + y
        return z

    @mpc_coro
    async def _if_swap_list(self, a, x, y):
        x, y = x[:], y[:]
        n = len(x)
        stype = type(a)  # all elts of x and y assumed of same type
        field = stype.field
        f = stype.frac_length
        if not f:
            await returnType(stype, 2, n)
        else:  # NB: a is integral
            await returnType((stype, x[0].integral and y[0].integral), 2, n)

        a, x, y = await self.gather(a, x, y)
        if f:
            a = a >> f  # NB: no in-place rshift!
        a = a.value
        d = [None] * n
        for i in range(n):
            d[i] = field(a * (y[i].value - x[i].value))
        d = await self._reshare(d)
        for i in range(n):
            x[i] = x[i] + d[i]
            y[i] = y[i] - d[i]
        return x, y

    def if_swap(self, c, x, y):
        '''Secure swap of x and y based on condition c.'''
        if isinstance(c, sectypes.SecureFixedPoint) and not c.integral:
            raise ValueError('condition must be integral')

        if isinstance(x, list):
            z = self._if_swap_list(c, x, y)
        else:
            d = c * (y - x)
            z = [x + d, y - d]
        return z

    @mpc_coro
    async def schur_prod(self, x, y):
        """Secure entrywise multiplication of vectors x and y."""
        if x == []:
            return []

        if x is y:
            x = x[:]
            y = x
        else:
            x, y = x[:], y[:]
        n = len(x)
        sftype = type(x[0])  # all elts of x and y assumed of same type
        if issubclass(sftype, SecureObject):
            f = sftype.frac_length
            if not f:
                await returnType(sftype, n)
            else:
                x_integral = x[0].integral
                y_integral = y[0].integral
                await returnType((sftype, x_integral and y_integral), n)
            if x is y:
                x = y = await self.gather(x)
            else:
                x, y = await self.gather(x, y)
        else:
            f = 0
            await returnType(Future)

        for i in range(n):
            x[i] = x[i] * y[i]
            if f and (x_integral or y_integral):
                x[i] >>= f  # NB: in-place rshift
        x = await self._reshare(x)
        if f and not x_integral and not y_integral:
            x = self.trunc(x, f=f, l=sftype.bit_length)
            x = await self.gather(x)
        return x

    @mpc_coro
    async def matrix_prod(self, A, B, tr=False):
        """Secure matrix product of A with (transposed) B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        shA = isinstance(A[0][0], SecureObject)
        shB = isinstance(B[0][0], SecureObject)
        stype = type(A[0][0]) if shA else type(B[0][0])
        field = stype.field
        f = stype.frac_length
        n1 = len(A)
        n2 = len(B) if tr else len(B[0])
        if not f:
            await returnType(stype, n1, n2)
        else:
            A_integral = A[0][0].integral
            B_integral = B[0][0].integral
            await returnType((stype, A_integral and B_integral), n1, n2)

        if shA and shB:
            A, B = await self.gather(A, B)
        elif shA:
            A = await self.gather(A)
        else:
            B = await self.gather(B)
        n = len(A[0])
        C = [None] * (n1 * n2)
        for ia in range(n1):
            for ib in range(n2):
                s = 0
                for i in range(n):
                    s += A[ia][i].value * (B[ib][i] if tr else B[i][ib]).value
                s = field(s)
                if f and (A_integral or B_integral):
                    s >>= f  # NB: in-place rshift
                C[ia * n2 + ib] = s
        if shA and shB:
            C = await self.gather(self._reshare(C))
        if f and not A_integral and not B_integral:
            C = [self.trunc(c, f=f, l=stype.bit_length) for c in C]
            C = await self.gather(C)
        return [[C[ia * n2 + ib] for ib in range(n2)] for ia in range(n1)]

    @mpc_coro
    async def gauss(self, A, d, b, c):
        """Secure Gaussian elimination A d - b c."""
        A, b, c = [r[:] for r in A], b[:], c[:]
        stype = type(A[0][0])
        field = stype.field
        n1, n2 = len(A), len(A[0])
        await returnType(stype, n1, n2)
        A, d, b, c = await self.gather(A, d, b, c)
        d = d.value
        for i in range(n1):
            b_i = b[i].value
            for j in range(n2):
                A[i][j] = field(A[i][j].value * d - b_i * c[j].value)
            A[i] = self._reshare(A[i])
        A = await self.gather(A)
        f = stype.frac_length
        if f:
            A = [self.trunc(a, f=f, l=stype.bit_length) for a in A]
            A = await self.gather(A)
        return A

    def _prss_uci(self):
        """Create unique common input for PRSS.

        Increments the program counter to ensure that consecutive calls
        to PRSS-related methods will use unique program counters.
        """
        self._program_counter[0] += 1
        return self._program_counter[0].to_bytes(8, 'little', signed=True)

    def _random(self, sftype, bound=None):
        """Secure random value of the given type in the given range."""
        return self._randoms(sftype, 1, bound)[0]

    def _randoms(self, sftype, n, bound=None):
        """n secure random values of the given type in the given range."""
        if issubclass(sftype, SecureObject):
            field = sftype.field
        else:
            field = sftype
        if bound is None:
            bound = field.order
        else:
            bound = 1 << max(0, (bound // self._bincoef).bit_length() - 1)  # NB: rounded power of 2
        m = len(self.parties)
        prfs = self.prfs(bound)
        shares = thresha.pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
        if issubclass(sftype, SecureObject):
            shares = [sftype(s) for s in shares]
        return shares

    def random_bit(self, stype, signed=False):
        """Secure uniformly random bit of the given type."""
        return self.random_bits(stype, 1, signed)[0]

    @mpc_coro
    async def random_bits(self, sftype, n, signed=False):
        """n secure uniformly random bits of the given type."""
        prss0 = False
        if issubclass(sftype, SecureObject):
            if issubclass(sftype, SecureFiniteField):
                prss0 = True
            await returnType((sftype, True), n)
            field = sftype.field
            f = sftype.frac_length
        else:
            await returnType(Future)
            field = sftype
            f = 0

        m = len(self.parties)
        if field.characteristic == 2:
            prfs = self.prfs(2)
            bits = thresha.pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
            return bits

        bits = [None] * n
        p = field.characteristic
        if not signed:
            modulus = field.modulus
            q = (p+1) >> 1  # q = 1/2 mod p
        prfs = self.prfs(field.order)
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
            r2s = await self.output(r2s, threshold=2*t)
            for r, r2 in zip(rs, r2s):
                if r2.value != 0:
                    h -= 1
                    s = r.value * r2.sqrt(INV=True).value
                    if not signed:
                        s %= modulus
                        s += 1
                        s *= q
                    bits[h] = field(s << f)
        return bits

    def add_bits(self, x, y):
        """Secure binary addition of bit vectors x and y."""
        x, y = x[:], y[:]

        def f(i, j, high=False):
            n = j - i
            if n == 1:
                c[i] = x[i] * y[i]
                if high:
                    d[i] = x[i] + y[i] - c[i]*2
            else:
                h = i + n//2
                f(i, h, high=high)
                f(h, j, high=True)
                c[h:j] = self.vector_add(c[h:j], self.scalar_mul(c[h-1], d[h:j]))
                if high:
                    d[h:j] = self.scalar_mul(d[h-1], d[h:j])
        n = len(x)
        c = [None] * n
        if n >= 1:
            d = [None] * n
            f(0, n)
        # c = prefix carries for addition of x and y
        for i in range(n-1, -1, -1):
            c[i] = x[i] + y[i] - c[i]*2 + (c[i-1] if i > 0 else 0)
        return c

    @mpc_coro
    async def to_bits(self, a, l=None):
        """Secure extraction of l (or all) least significant bits of a."""  # a la [ST06].
        stype = type(a)
        if l is None:
            l = stype.bit_length
        # TODO: check l <= stype.bit_length (+1 for SecureFiniteField)
        await returnType((stype, True), l)
        field = stype.field
        f = stype.frac_length
        rshift_f = f and a.integral  # optimization for integral fixed-point numbers
        if rshift_f:
            # f least significant bits of a are all 0
            if f >= l:
                return [field(0) for _ in range(l)]

            l -= f

        r_bits = await self.random_bits(field, l)
        r_modl = 0
        for r_i in reversed(r_bits):
            r_modl <<= 1
            r_modl += r_i.value

        if issubclass(stype, sectypes.SecureFiniteField):
            if field.characteristic == 2:
                a = await self.gather(a)
                c = await self.output(a + r_modl)
                c = int(c.value)
                return [r_bits[i] + ((c >> i) & 1) for i in range(l)]

            if field.ext_deg > 1:
                raise TypeError('Binary field or prime field required.')

            a = self.convert(a, self.SecInt(l=1+stype.field.order.bit_length()))
            a_bits = self.to_bits(a)
            return self.convert(a_bits, stype)

        k = self.options.sec_param
        r_divl = self._random(field, 1<<(stype.bit_length + k - l)).value
        a = await self.gather(a)
        if rshift_f:
            a = a >> f
        c = await self.output(a + ((1<<stype.bit_length) + (r_divl << l) - r_modl))
        c = c.value % (1<<l)
        c_bits = [(c >> i) & 1 for i in range(l)]
        r_bits = [stype(r.value) for r in r_bits]  # TODO: drop .value, fix secfxp(r) if r field elt
        a_bits = self.add_bits(r_bits, c_bits)
        if rshift_f:
            a_bits = [field(0) for _ in range(f)] + a_bits
        return a_bits

    @mpc_coro_no_pc
    async def from_bits(self, x):
        """Recover secure number from its binary representation x."""
        # TODO: also handle negative numbers with sign bit (NB: from_bits() in random.py)
        if x == []:
            return 0

        x = x[:]
        stype = type(x[0])
        await returnType((stype, True))
        x = await self.gather(x)
        s = 0
        for a in reversed(x):
            s <<= 1
            s += a.value
        return stype.field(s)

    def find(self, x, a, bits=True, e='len(x)', f=None, cs_f=None):
        """Return index ix of the first occurrence of a in list x.

        The elements of x and a are assumed to be in {0, 1}, by default.
        Set Boolean flag bits=False for arbitrary inputs.

        If a is not present in x, then ix=len(x) by default.
        Set flag e=E to get ix=E instead if a is not present in x.
        Set e=None to get the "raw" output as a pair (nf, ix), where
        the indicator bit nf=1 if and only if a is not found in x.

        For instance, E=-1 can be used to mimic Python's find() methods.
        If E is a string, i=eval(E) will be returned where E is
        an expression
        in terms of len(x). As a simple example, E='len(x)-1' will enforce
        that a is considered to be present in any case as the last element
        of x, if not earlier.

        The return value is index ix, by default.
        If function f is set, the value of f(ix) is returned instead.
        Even though index ix is a secure number, however, the computation of f(ix)
        does not incur any communication costs, requiring local computation only.
        For example, with f=lambda i: 2^i we would save the work for a secure
        exponentiation, which would otherwise require a call to runtime.to_bits(ix), say.

        Also, function cs_f can be set instead of specifying function f directly.
        Function cs_f(b, i) will take as input a (secure) bit b and a (public) index i.
        The relationship between function f and its "conditional-step function" cs_f
        is given by:

            cs_f(b, i) = f(i+b),                      for b in {0, 1}      (*)

        For example, for f(i) = i, we get cs_f(b, i) = i+b.
        And, for f(i) = 2^i, we get cs_f(b, i) = 2^(i+b) = (b+1) 2^i.
        In general, we have:

            cs_f(b, i) = b (f(i+1) - f(i)) + f(i),    for b in {0, 1}      (**)

        For this reason, cs_f(b, i) can be computed locally indeed.

        A few more examples:

            f(i)    =   i  |    2^i    |  n-i  |    2^-i        | (i, 2^i)

         cs_f(b, i) =  i+b | (b+1) 2^i | n-i-b | (2-b) 2^-(i+1) | (i+b, (b+1) 2^i)

        In the last example, f(i) is a tuple containing two values. In general, f(i)
        can be a single number or a tuple/list of numbers.

        Note that it suffices to specify either f or cs_f, as (*) implies that
        f(i) = cs_f(0, i). If cs_f is not set, it will be computed from f using (**),
        incurring some overhead.
        """
        if bits:
            if not isinstance(a, int):
                x = self.vector_add([a] * len(x), self.scalar_mul(1 - 2*a, x))
            elif a == 1:
                x = [1-b for b in x]
        else:
            x = [b != a for b in x]
        # Problem is now reduced to finding the index of the first 0 in x.

        if cs_f is None:
            if f is None:
                type_f = int
                f = lambda i: [i]
                cs_f = lambda b, i: [i + b]
            else:
                type_f = type(f(0))
                if issubclass(type_f, int):
                    _f = f
                    f = lambda i: [_f(i)]
                cs_f = lambda b, i: [b * (f_i1 - f_i) + f_i for f_i, f_i1 in zip(f(i), f(i+1))]
        else:
            if f is None:
                type_f = type(cs_f(0, 0))
                if issubclass(type_f, int):
                    _cs_f = cs_f
                    cs_f = lambda b, i: [_cs_f(b, i)]
                elif issubclass(type_f, tuple):
                    _cs_f = cs_f
                    cs_f = lambda b, i: list(_cs_f(b, i))
                f = lambda i: cs_f(0, i)
            else:
                pass  # TODO: check correctness f vs cs_f

        if isinstance(e, str):
            e = eval(e)

        if not x:
            if e is None:
                nf, y = 1, f(0)
            else:
                y = f(e)
        else:
            def cl(i, j):
                n = j - i
                if n == 1:
                    b = x[i]
                    return [b] + cs_f(b, i)

                h = i + n//2
                nf = cl(i, h)  # nf[0] <=> "0 is not found"
                return self.if_else(nf[0], cl(h, j), nf)

            nf, *f_ix = cl(0, len(x))
            if e is None:
                y = f_ix
            else:
                f_e = list(map(type(nf), f(e)))
                y = self.if_else(nf, f_e, f_ix)
        if issubclass(type_f, int):
            y = y[0]
        elif issubclass(type_f, tuple):
            y = tuple(y)
        return (nf, y) if e is None else y

    @mpc_coro
    async def indexOf(self, x, a, bits=False):
        """Return index of the first occurrence of a in x.

        Raise ValueError if a is not present.
        """
        if not x:
            raise ValueError('value is not in list')

        stype = type(x[0])  # all elts of x and y assumed of same type
        await self.returnType((stype, True))

        ix = self.find(x, a, e=-1, bits=bits)
        if await self.eq_public(ix, -1):
            raise ValueError('value is not in list')

        return ix

    def _norm(self, a):  # signed normalization factor
        l = type(a).bit_length
        f = type(a).frac_length
        x = self.to_bits(a)  # low to high bits
        b = x[-1]  # sign bit
        del x[-1]
        x.reverse()
        nf = self.find(x, 1-b, cs_f=lambda b, i: (b+1) << i)
        return (1 - b*2) * nf * (2**(f - (l-1)))  # NB: f <= l

    def _rec(self, a):  # enhance performance by reducing no. of truncs
        f = type(a).frac_length
        v = self._norm(a)
        b = a * v  # 1/2 <= b <= 1
        theta = int(math.ceil(math.log2((f+1)/3.54)))
        c = 2.9142135623731 - b*2
        for _ in range(theta):
            c *= 2 - c * b
        return c * v

    def unit_vector(self, a, n):
        """Length-n unit vector [0]*a + [1] + [0]*(n-1-a) for secret a, assuming 0 <= a < n.

        NB: If a = n, unit vector [1] + [0]*(n-1) is returned. See mpyc.statistics.
        """
        b = n - 1
        k = b.bit_length()
        f = type(a).frac_length
        if f and not a.integral:
            raise ValueError('nonintegral fixed-point number')

        x = self.to_bits(a, k + f)[f:]
        u = []
        for i in range(k-1, -1, -1):
            v = self.scalar_mul(x[i], u)  # v = x[i] * u
            w = self.vector_sub(u, v)  # w = (1-x[i]) * u
            u = [x[i] - self.sum(v)]
            u.extend(c for _ in zip(w, v) for c in _)
            if not (b >> i) & 1:
                u.pop()
        # u is (a-1)st unit vector of length n-1 (if 1<=a<n) or all-0 vector of length n-1 (if a=0).
        return [type(a)(1) - self.sum(u)] + u


class Party:
    """Information about a party in the MPC protocol."""

    __slots__ = 'pid', 'host', 'port', 'protocol'

    def __init__(self, pid, host=None, port=None):
        """Initialize a party with given party identity pid."""
        self.pid = pid
        self.host = host
        self.port = port
        self.protocol = None

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
                        help='show this help message for MPyC and exit')
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
    group.add_argument('--no-gmpy2', action='store_true',
                       default=False, help='disable use of gmpy2 package')
    group.add_argument('--no-prss', action='store_true',
                       default=False, help='disable use of PRSS (pseudorandom secret sharing)')
    group.add_argument('--mix32-64bit', action='store_true',
                       default=False, help='enable mix of 32-bit and 64-bit platforms')

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

    env_no_gmpy2 = os.getenv('MPYC_NOGMPY') == '1'  # check if variable MPYC_NOGMPY is set
    if not importlib.util.find_spec('gmpy2'):
        # gmpy2 package not available
        if not (options.no_gmpy2 or env_no_gmpy2):
            logging.info('Install package gmpy2 for better performance.')
    else:
        # gmpy2 package available
        if options.no_gmpy2 or env_no_gmpy2:
            logging.info('Use of package gmpy2 disabled.')
            if not env_no_gmpy2:
                os.environ['MPYC_NOGMPY'] = '1'  # NB: MPYC_NOGMPY also set for subprocesses
                from importlib import reload
                reload(mpyc.gmpy)  # stubs will be loaded this time

    env_mix32_64bit = os.getenv('MPYC_MIX32_64BIT') == '1'  # check if MPYC_MIX32_64BIT is set
    if options.mix32_64bit or env_mix32_64bit:
        logging.info('Mix of parties on 32-bit and 64-bit platforms enabled.')
        from hashlib import sha1

        def hop(a):
            """Simple and portable pseudorandom program counter hop for Python 3.6+.

            Compatible across all (mixes of) 32-bit and 64-bit Python 3.6+ versions. Let's
            you run MPyC with some parties on 64-bit platforms and others on 32-bit platforms.
            Useful when working with standard 64-bit installations on Linux/MacOS/Windows and
            installations currently restricted to 32-bit such as pypy3 on Windows and Python on
            Raspberry Pi OS.
            """
            return int.from_bytes(sha1(str(a).encode()).digest()[:8], 'little', signed=True)
        asyncoro._hop = hop

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
        m = len(parties)
        if pid is None:
            pid = options.index
    else:
        # use default port for each local party
        m = options.M or 1
        if m > 1 and options.index is None:
            import platform
            import subprocess
            # convert sys.flags into command line arguments
            flgmap = {'debug': 'd', 'inspect': 'i', 'interactive': 'i', 'optimize': 'O',
                      'dont_write_bytecode': 'B', 'no_user_site': 's', 'no_site': 'S',
                      'ignore_environment': 'E', 'verbose': 'v', 'bytes_warning': 'b',
                      'quiet': 'q', 'isolated': 'I', 'dev_mode': 'X dev', 'utf8_mode': 'X utf8'}
            if os.getenv('PYTHONHASHSEED') == '0':
                # -R flag needed only if hash randomization is not enabled by default
                flgmap['hash_randomization'] = 'R'
            flg = lambda a: getattr(sys.flags, a, 0)
            flags = ['-' + flg(a) * c for a, c in flgmap.items() if flg(a)]
            # convert sys._xoptions into command line arguments
            xopts = ['-X' + a + ('' if c is True else '=' + c) for a, c in sys._xoptions.items()]
            prog, args = argv[0], argv[1:]
            for i in range(m-1, 0, -1):
                cmd_line = [sys.executable] + flags + xopts + [prog, '-I', str(i)] + args
                if options.output_windows and platform.platform().startswith('Windows'):
                    subprocess.Popen(['start'] + cmd_line, shell=True)
                elif options.output_file:
                    with open(f'party{options.M}_{i}.log', 'a') as f:
                        f.write('\n')
                        f.write(f'$> {" ".join(cmd_line)}\n')
                        subprocess.Popen(cmd_line, stdout=f, stderr=subprocess.STDOUT)
                else:
                    subprocess.Popen(cmd_line, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        options.no_async = m == 1 and (options.no_async or not options.M)
        pid = options.index or 0
        base_port = options.base_port or 11365
        parties = [Party(i, 'localhost', base_port + i) for i in range(m)]

    if options.threshold is None:
        options.threshold = (m-1)//2
    assert 2*options.threshold < m, f'threshold {options.threshold} too large for {m} parties'

    rt = Runtime(pid, parties, options)
    sectypes.runtime = rt
    asyncoro.runtime = rt
    mpyc.random.runtime = rt
    mpyc.statistics.runtime = rt
    mpyc.seclists.runtime = rt
    return rt


if os.getenv('READTHEDOCS') != 'True':
    try:
        mpc = setup()
    except Exception as exc:
        # suppress exceptions for pydoc etc.
        print('MPyC runtime.setup() exception:', exc)

"""The MPyC runtime module is used to execute secure multiparty computations.

Parties perform computations on secret-shared values by exchanging messages.
Shamir's threshold secret sharing scheme is used for finite fields of any order
exceeding the number of parties. MPyC provides many secure data types, ranging
from numeric types to more advanced types, for which the corresponding operations
are made available through Python's mechanism for operator overloading.
"""

import os
import sys
import time
import datetime
import logging
import math
import secrets
import itertools
import functools
import configparser
from dataclasses import dataclass
import pickle
import asyncio
import ssl
from mpyc.numpy import np
from mpyc import finfields
from mpyc import thresha
from mpyc import sectypes
from mpyc import asyncoro
import mpyc.secgroups
import mpyc.random
import mpyc.statistics
import mpyc.seclists

Future = asyncio.Future
mpc_coro = asyncoro.mpc_coro
mpc_coro_no_pc = asyncoro._mpc_coro_no_pc


class Runtime:
    """MPyC runtime secure against passive attacks.

    The runtime maintains basic information such as a program counter, the list
    of parties, etc., and handles secret-shared objects of type SecureObject.

    1-party case is supported (with option to disable asynchronous evaluation).
    Threshold 0 (no corrupted parties) is supported for m-party case as well
    to enable distributed computation (without secret sharing).
    """

#    __slots__ = ('pid', 'parties', 'options', '_threshold', '_logging_enabled', '_program_counter',
#                 '_pc_level', '_loop', 'start_time', 'aggregate_load', '_prss_keys', '_bincoef')
    version = mpyc.__version__
    SecureObject = sectypes.SecureObject
    SecureNumber = sectypes.SecureNumber
    SecureFiniteField = sectypes.SecureFiniteField
    SecureInteger = sectypes.SecureInteger
    SecureFixedPoint = sectypes.SecureFixedPoint
    SecureFloat = sectypes.SecureFloat
    SecureFiniteGroup = mpyc.secgroups.SecureFiniteGroup
    SecureArray = sectypes.SecureArray
    SecureFiniteFieldArray = sectypes.SecureFiniteFieldArray
    SecureIntegerArray = sectypes.SecureIntegerArray
    SecureFixedPointArray = sectypes.SecureFixedPointArray
    SecFld = staticmethod(sectypes.SecFld)
    SecInt = staticmethod(sectypes.SecInt)
    SecFxp = staticmethod(sectypes.SecFxp)
    SecFlt = staticmethod(sectypes.SecFlt)
    coroutine = staticmethod(mpc_coro)
    returnType = staticmethod(asyncoro.returnType)
    seclist = mpyc.seclists.seclist
    SecGrp = staticmethod(mpyc.secgroups.SecGrp)
    SecSymmetricGroup = staticmethod(mpyc.secgroups.SecSymmetricGroup)
    SecQuadraticResidues = staticmethod(mpyc.secgroups.SecQuadraticResidues)
    SecSchnorrGroup = staticmethod(mpyc.secgroups.SecSchnorrGroup)
    SecEllipticCurve = staticmethod(mpyc.secgroups.SecEllipticCurve)
    SecClassGroup = staticmethod(mpyc.secgroups.SecClassGroup)
    random = mpyc.random
    statistics = mpyc.statistics

    def __init__(self, pid, parties, options):
        """Initialize runtime."""
        self.pid = pid
        self.parties = tuple(parties)
        self.options = options
        self.threshold = options.threshold
        self._logging_enabled = not options.no_log
        self._program_counter = [0, 0]  # [hopping-counter, program-depth]
        self._pc_level = 0  # used for implementation of barriers
        self._loop = asyncio.get_event_loop()  # cache event loop
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
        self._bincoef = math.comb(m, t)
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
        await self.barrier(name=name)

    def run(self, f):
        """Run the given coroutine or future until it is done."""
        if self._loop.is_running():
            if not asyncio.iscoroutine(f):
                f = asyncoro._wrap_in_coro(f)
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
        elapsed = time.time() - self.start_time
        logging.info(f'Stop MPyC runtime -- elapsed time: {datetime.timedelta(seconds=elapsed)}')
        m = len(self.parties)
        if m == 1:
            return

        # m > 1
        self.parties[self.pid].protocol = Future(loop=self._loop)
        logging.debug('Synchronize with all parties before shutdown')
        await self.gather(self.transfer(self.pid))

        # Close connections to all parties > self.pid.
        logging.debug('Closing connections with other parties')
        for peer in self.parties[self.pid + 1:]:
            peer.protocol.close_connection()
        await self.parties[self.pid].protocol

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
        if hasattr(stype, '_input'):
            return stype._input(x, senders)

        if isinstance(x[0], sectypes.SecureArray):
            field = x[0].sectype.field
            shape = x[0].shape  # TODO: consider multiple arrays
        else:
            field = stype.field
            shape = None
        if not stype.frac_length:
            if shape is None:
                rettype = stype
            else:
                rettype = (stype, shape)
        else:
            if shape is None:
                rettype = (stype, x[0].integral)
            else:
                rettype = (stype, x[0].integral, shape)
        await self.returnType(rettype, len(senders), len(x))

        shares = [None] * len(senders)
        for i, peer_pid in enumerate(senders):
            if peer_pid == self.pid:
                x = await self.gather(x)
                t = self.threshold
                m = len(self.parties)
                if shape is not None:
                    x = x[0].value.flat  # indexable iterator
                in_shares = thresha.random_split(field, x, t, m)
                for other_pid, data in enumerate(in_shares):
                    data = field.to_bytes(data)
                    if other_pid == self.pid:
                        shares[i] = data
                    else:
                        self._send_message(other_pid, data)
            else:
                shares[i] = self._receive_message(peer_pid)
        shares = await self.gather(shares)
        if shape is None:
            y = [[field(a) for a in field.from_bytes(r)] for r in shares]
        else:
            y = [[field.array(field.from_bytes(r), check=False).reshape(shape) for r in shares]]
        return y

    @mpc_coro
    async def output(self, x, receivers=None, threshold=None, raw=False):
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
        await self.returnType(Future)

        n = len(x)
        if not n:
            return []

        t = self.threshold if threshold is None else threshold
        m = len(self.parties)
        if receivers is None:
            receivers = range(m)  # default
        receivers = [receivers] if isinstance(receivers, int) else list(receivers)
        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, self.SecureObject):
            if hasattr(sftype, '_output'):
                y = await sftype._output(x, receivers, threshold)
                if not x_is_list:
                    y = y[0]
                return y

            x = await self.gather(x)

        if isinstance(x[0], finfields.FiniteFieldElement):
            field = type(x[0])
            x = [a.value for a in x]
            shape = None
        else:
            field = x[0].field
            x = x[0].value  # TODO: consider multiple arrays
            shape = x.shape
            x = x.flat  # indexable iterator

        # Send share x to all successors in receivers.
        share = None
        for peer_pid in receivers:
            if 0 < (peer_pid - self.pid) % m <= t:
                if share is None:
                    share = field.to_bytes(x)
                self._send_message(peer_pid, share)
        # Receive and recombine shares if this party is a receiver.
        if self.pid in receivers:
            shares = [self._receive_message((self.pid - t + j) % m) for j in range(t)]
            shares = await self.gather(shares)
            points = [((self.pid - t + j) % m + 1, field.from_bytes(shares[j])) for j in range(t)]
            points.append((self.pid + 1, x))
            y = thresha.recombine(field, points)
            if shape is None:
                y = [field(a) for a in y]
            else:
                y = [field.array(y).reshape(shape)]
            if issubclass(sftype, self.SecureObject):
                f = sftype._output_conversion
                if not raw and f is not None:
                    y = [f(a) for a in y]
        else:
            y = [None] * n
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
        if issubclass(sftype, self.SecureObject):
            if not sftype.frac_length:
                if issubclass(sftype, sectypes.SecureArray):
                    rettype = (sftype, x[0].shape)
                else:
                    rettype = sftype
            else:
                if issubclass(sftype, sectypes.SecureArray):
                    rettype = (sftype, x[0].integral, x[0].shape)
                else:
                    rettype = (sftype, x[0].integral)
            if x_is_list:
                await self.returnType(rettype, len(x))
            else:
                await self.returnType(rettype)
            x = await self.gather(x)
        else:
            await self.returnType(Future)

        t = self.threshold
        if t == 0:
            if not x_is_list:
                x = x[0]
            return x

        if isinstance(x[0], finfields.FiniteFieldElement):
            field = type(x[0])
            x = [a.value for a in x]
            shape = None
        else:
            field = x[0].field
            x = x[0].value  # TODO: consider multiple arrays, see e.g., np_prod(), np_all()
            shape = x.shape
            x = x.flat  # indexable iterator

        m = len(self.parties)
        in_shares = thresha.random_split(field, x, t, m)
        in_shares = [field.to_bytes(elts) for elts in in_shares]
        # Recombine the first 2t+1 output_shares.
        out_shares = await self.gather(self._exchange_shares(in_shares)[:2*t+1])
        points = [(j+1, field.from_bytes(s)) for j, s in enumerate(out_shares)]
        y = thresha.recombine(field, points)
        if shape is None:
            y = [field(a) for a in y]
        else:
            y = [field.array(y).reshape(shape)]
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

        if (isinstance(x[0], self.SecureFiniteField)
                and issubclass(ttype, self.SecureFiniteField)):
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
        await self.returnType((ttype, not stype.frac_length), n)  # target type
        m = len(self.parties)
        k = self.options.sec_param
        l = min(stype.bit_length, ttype.bit_length)
        if issubclass(stype, self.SecureFiniteField):
            bound = stype.field.order
        else:
            bound = (1<<(k + l)) // self._bincoef + 1
        prfs = self.prfs(bound)
        uci = self._prss_uci()  # NB: same uci in both calls for r below

        x = await self.gather(x)
        d = ttype.frac_length - stype.frac_length  # TODO: use integral attribute fxp
        if d < 0:
            x = await self.trunc(x, f=-d, l=stype.bit_length)  # TODO: take minimum with ttype or so
        if stype.field.is_signed:
            if issubclass(stype, self.SecureFiniteField):
                offset = stype.field.modulus // 2
            else:
                offset = 1 << l-1
        else:
            offset = 0
        r = thresha.pseudorandom_share(stype.field, m, self.pid, prfs, uci, n)
        for i in range(n):
            x[i] = x[i].value + offset + r[i]

        x = await self.output(x)
        r = thresha.pseudorandom_share(ttype.field, m, self.pid, prfs, uci, n)
        for i in range(n):
            x[i] = x[i].value - r[i]
            if issubclass(stype, self.SecureFiniteField):
                x[i] = self._mod(ttype(x[i]), stype.field.modulus)
            x[i] = x[i] - offset
        if d > 0 and not issubclass(stype, self.SecureFiniteField):
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
        if issubclass(sftype, self.SecureObject):
            if x_is_list:
                await self.returnType(sftype, n)
            else:
                await self.returnType(sftype)
            Zp = sftype.field
            l = l or sftype.bit_length
            if f is None:
                f = sftype.frac_length
        else:
            await self.returnType(Future)
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
        if issubclass(sftype, self.SecureObject):
            x = await self.gather(x)
        c = await self.output([a + ((1 << l-1 + f) + (q.value << f) + r.value)
                               for a, q, r in zip(x, r_divf, r_modf)])
        c = [c.value % (1<<f) for c in c]
        y = [(a - c + r.value) >> f for a, c, r in zip(x, c, r_modf)]
        if not x_is_list:
            y = y[0]
        return y

    @mpc_coro
    async def np_trunc(self, a, f=None, l=None):
        """Secure truncation of f least significant bits of (elements of) a.

        Probabilistic rounding of a / 2**f (elementwise).
        """
        n = a.size
        sftype = type(a)
        if issubclass(sftype, self.SecureObject):
            await self.returnType((sftype, a.shape))
            Zp = sftype.sectype.field
            l = l or sftype.sectype.bit_length
            if f is None:
                f = sftype.frac_length
        else:
            await self.returnType(Future)
            Zp = sftype.field

        k = self.options.sec_param
        r_bits = await self.np_random_bits(Zp, f * n)
        r_modf = np.sum(r_bits.value.reshape((n, f)) << np.arange(f), axis=1)
        r_modf = r_modf.reshape(a.shape)
        r_divf = self._np_randoms(Zp, n, 1 << k + l).value
        r_divf = r_divf.reshape(a.shape)
        if issubclass(sftype, self.SecureObject):
            a = await self.gather(a)
        c = await self.output(Zp.array(a.value + (1 << l-1 + f) + (r_divf << f) + r_modf))
        c = c.value & ((1<<f) - 1)
        y = Zp.array(a.value + r_modf - c) >> f
        return y

    def eq_public(self, a, b):
        """Secure public equality test of a and b."""
        return self.is_zero_public(a - b)

    @mpc_coro
    async def is_zero_public(self, a) -> Future:
        """Secure public zero test of a."""
        sftype = type(a)
        if issubclass(sftype, sectypes.SecureFloat):
            return await sftype.is_zero_public(a)

        if issubclass(sftype, self.SecureObject):
            field = sftype.field
        else:
            field = sftype

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

        if issubclass(sftype, self.SecureObject):
            a = await self.gather(a)
        if field.order.bit_length() <= 60:
            z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), 1)
            b = a * r + z[0]
        else:
            b = a * r
        c = await self.output(b, threshold=2*t)
        return c == 0

    @mpc_coro
    async def np_is_zero_public(self, a) -> Future:
        """Secure public zero test of a, elementwise."""
        # TODO: remove restriction to 1D arrays a (due to r,s,z below being 1D arrays)
        sftype = type(a)
        if issubclass(sftype, self.SecureArray):
            field = sftype.sectype.field
        else:
            field = sftype

        n = a.size
        m = len(self.parties)
        t = self.threshold
        if field.order.bit_length() <= 60:  # TODO: introduce MPyC parameter for failure probability
            prfs = self.prfs(field.order)
            while True:
                r = self._np_randoms(field, n)
                s = self._np_randoms(field, n)
                z = thresha.np_pseudorandom_share_0(field, m, self.pid, prfs, self._prss_uci(), n)
                if np.all(await self.output(r * s + z, threshold=2*t)):
                    # TODO: handle failures for cases of small sec. param k (like 8),
                    # small bit_length l (like 2) and large n (like 200); filter the 0s.
                    break
        else:
            r = self._np_randoms(field, n)  # NB: failure r=0 with probability less than 2**-60

        if issubclass(sftype, self.SecureObject):
            a = await self.gather(a)
        if field.order.bit_length() <= 60:
            z = thresha.np_pseudorandom_share_0(field, m, self.pid, prfs, self._prss_uci(), n)
            b = a * r + z
        else:
            b = a * r
        c = await self.output(b, threshold=2*t)
        return c == 0

    @mpc_coro_no_pc
    async def neg(self, a):
        """Secure negation (additive inverse) of a."""
        stype = type(a)
        if not stype.frac_length:
            await self.returnType(stype)
        else:
            await self.returnType((stype, a.integral))
        a = await self.gather(a)
        return -a

    @mpc_coro_no_pc
    async def pos(self, a):
        """Secure unary + applied to a."""
        stype = type(a)
        if not stype.frac_length:
            await self.returnType(stype)
        else:
            await self.returnType((stype, a.integral))
        a = await self.gather(a)
        return +a

    @mpc_coro_no_pc
    async def add(self, a, b):
        """Secure addition of a and b."""
        stype = type(a)
        if not stype.frac_length:
            await self.returnType(stype)
        else:
            await self.returnType((stype, a.integral and b.integral))
        a, b = await self.gather(a, b)
        return a + b

    @mpc_coro_no_pc
    async def sub(self, a, b):
        """Secure subtraction of a and b."""
        stype = type(a)
        if not stype.frac_length:
            await self.returnType(stype)
        else:
            await self.returnType((stype, a.integral and b.integral))
        a, b = await self.gather(a, b)
        return a - b

    @mpc_coro_no_pc
    async def np_add(self, a, b):
        stype = type(a)
        a_shape = getattr(a, 'shape', (1,))
        b_shape = getattr(b, 'shape', (1,))
        shape = np.broadcast_shapes(a_shape, b_shape)
        if not stype.frac_length:
            await self.returnType((stype, shape))
        else:
            await self.returnType((stype, a.integral and b.integral, shape))
        a, b = await self.gather(a, b)
        return a + b

    @mpc_coro_no_pc
    async def np_subtract(self, a, b):
        stype = type(b) if isinstance(b, self.SecureArray) else type(a)
        a_shape = getattr(a, 'shape', (1,))
        b_shape = getattr(b, 'shape', (1,))
        shape = np.broadcast_shapes(a_shape, b_shape)
        if not stype.frac_length:
            await self.returnType((stype, shape))
        else:
            await self.returnType((stype, a.integral and b.integral, shape))
        a, b = await self.gather(a, b)
        return a - b

    @mpc_coro
    async def mul(self, a, b):
        """Secure multiplication of a and b."""
        stype = type(a)
        shb = isinstance(b, self.SecureObject)
        f = stype.frac_length
        if not f:
            await self.returnType(stype)
        else:
            a_integral = a.integral
            b_integral = shb and b.integral
            z = 0
            if not shb:
                if isinstance(b, int):
                    z = f
                elif isinstance(b, float):
                    b = round(b * 2**f)
                    z = max(0, min(f, (b & -b).bit_length() - 1))
                    b >>= z  # remove trailing zeros
            await self.returnType((stype, a_integral and (b_integral or z == f)))

        if not shb:
            a = await self.gather(a)
        elif a is b:
            a = b = await self.gather(a)
        else:
            a, b = await self.gather(a, b)
        c = a * b
        if f and (a_integral or b_integral) and z != f:
            c >>= f - z  # NB: in-place rshift
        if shb:
            c = self._reshare(c)
        if f and not (a_integral or b_integral) and z != f:
            c = self.trunc(stype(c), f=f - z)
        return c

    @mpc_coro
    async def np_multiply(self, a, b):
        stype = type(a)
        shb = isinstance(b, self.SecureObject)
        a_shape = getattr(a, 'shape', (1,))
        b_shape = getattr(b, 'shape', (1,))
        shape = np.broadcast_shapes(a_shape, b_shape)
        f = stype.frac_length
        if not f:
            await self.returnType((stype, shape))
        else:
            a_integral = a.integral
            b_integral = shb and b.integral
            z = 0
            if not shb:
                if isinstance(b, int):
                    z = f
                elif isinstance(b, float):
                    b = round(b * 2**f)
                    z = max(0, min(f, (b & -b).bit_length() - 1))
                    b >>= z  # remove trailing zeros
                elif isinstance(b, np.ndarray):
                    if np.issubdtype(b.dtype, np.integer):
                        z = f
                    elif np.issubdtype(b.dtype, np.floating):
                        # NB: unlike for self.mul() no test if all entries happen to be integral
                        # Scale to Python int entries (by setting otypes='O', prevents overflow):
                        b = np.vectorize(round, otypes='O')(b * 2**f)
                    # TODO: handle b.dtype=object, checking if all elts are int
            await self.returnType((stype, a_integral and (b_integral or z == f), shape))

        if not shb:
            a = await self.gather(a)
        elif a is b:
            a = b = await self.gather(a)
        else:
            a, b = await self.gather(a, b)
        c = a * b
        if f and (a_integral or b_integral) and z != f:
            c >>= f - z  # NB: in-place rshift
        if shb:
            c = self._reshare(c)
        if f and not (a_integral or b_integral) and z != f:
            c = self.np_trunc(stype(c, shape=shape), f=f - z)
        return c

    def div(self, a, b):
        """Secure division of a by b, for nonzero b."""
        b_is_SecureObject = isinstance(b, self.SecureObject)
        stype = type(b) if b_is_SecureObject else type(a)
        field = stype.field
        f = stype.frac_length
        if b_is_SecureObject:
            if f:
                c = self._rec(b)
            else:
                c = self.reciprocal(b)
            return self.mul(c, a)

        # isinstance(a, self.SecureObject) ensured
        if f:
            if isinstance(b, (int, float)):
                c = 1/b
                if c.is_integer():
                    c = round(c)
            else:
                c = b.reciprocal() << f
        else:
            if isinstance(b, field):
                b = b.value
            c = field(field._reciprocal(b))
        return self.mul(a, c)

    def np_divide(self, a, b):
        b_is_SecureArray = isinstance(b, self.SecureArray)
        stype = type(b) if b_is_SecureArray else type(a)
        field = stype.sectype.field
        f = stype.frac_length
        if b_is_SecureArray:
            if f:
                c = self._rec(b)
            else:
                c = self.np_reciprocal(b)
            return self.np_multiply(c, a)

        # isinstance(a, self.SecureArray) ensured
        if f:
            if isinstance(b, (int, float)):
                c = 1/b
                if c.is_integer():
                    c = round(c)
            elif isinstance(b, self.SecureFixedPoint):
                c = self._rec(b)
            else:
                c = b.reciprocal() << f
        else:
            if not isinstance(b, field.array):
                b = field.array(b)  # TODO: see if this can be used for case f != 0 as well
            c = b.reciprocal()
        return self.np_multiply(a, c)

    @mpc_coro
    async def reciprocal(self, a):
        """Secure reciprocal (multiplicative field inverse) of a, for nonzero a."""
        stype = type(a)
        field = stype.field
        await self.returnType(stype)
        a = await self.gather(a)
        while True:
            r = self._random(field)
            ar = await self.output(a * r, threshold=2*self.threshold)
            if ar:
                break
        r <<= stype.frac_length
        return r / ar

    @mpc_coro
    async def np_reciprocal(self, a):
        """Secure reciprocal (multiplicative field inverse) of a, for nonzero a."""
        stype = type(a)
        shape = a.shape
        field = stype.sectype.field
        await self.returnType((stype, shape))
        a = await self.gather(a)
        while True:  # will only succeec for large fields or small arrays a
            r = self._np_randoms(field, a.size).reshape(shape)
            ar = await self.output(a * r, threshold=2*self.threshold)
            if (ar != 0).all():
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

    def np_pow(self, a, b):
        """Secure exponentiation a raised to the power of b, for public integer b."""
        # TODO: extend to non-scalar b
        if b == 254:  # addition chain for AES S-Box (11 multiplications in 9 rounds)
            d = a
            c = self.np_multiply(d, d)
            c = self.np_multiply(c, c)
            c = self.np_multiply(c, c)
            c = self.np_multiply(c, d)
            c = self.np_multiply(c, c)
            c, d = self.np_multiply(c, self.np_stack((c, d)))
            c, d = self.np_multiply(c, self.np_stack((c, d)))
            c = self.np_multiply(c, d)
            c = self.np_multiply(c, c)
            return c

        if b == 0:
            return type(a)(np.ones(a.shape, dtype='O'))

        if b < 0:
            a = self.np_reciprocal(a)
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
        return 1 - self.lt(a, b)  # TODO: deprecate, give warning maybe

    def abs(self, a, l=None):
        """Secure absolute value of a."""
        return (-2*self.sgn(a, l=l, LT=True) + 1) * a

    def is_zero(self, a):
        """Secure zero test a == 0."""
        if isinstance(a, self.SecureFiniteField):
            return 1 - self.pow(a, a.field.order - 1)

        if a.bit_length/2 > self.options.sec_param >= 8 and a.field.order%4 == 3:
            return self._is_zero(a)

        return self.sgn(a, EQ=True)

    @mpc_coro
    async def _is_zero(self, a):  # a la [NO07]
        """Probabilistic zero test."""
        stype = type(a)
        await self.returnType((stype, True))
        Zp = stype.field

        k = self.options.sec_param
        z = self.random_bits(Zp, k)
        u = self._randoms(Zp, k)
        u2 = self.schur_prod(u, u)
        a, u2, z = await self.gather(a, u2, z)
        a = a.value
        r = self._randoms(Zp, k)
        c = [Zp(a * r[i].value + (1-(z[i].value << 1)) * u2[i].value) for i in range(k)]
        # -1 is nonsquare for Blum p, u[i] !=0 w.v.h.p.
        # If a == 0, c[i] is square mod p iff z[i] == 0.
        # If a != 0, c[i] is square mod p independent of z[i].
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
        await self.returnType((stype, True))
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

        In-place sort in roughly 1/2(log_2 n)^2 rounds of about n/2 comparisons each.
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
                        x[i], x[i + d] = self.if_swap(key(a) >= key(b), a, b)
                d, q, r = q - p, q >> 1, p
            p >>= 1
        return x

    @mpc_coro
    async def lsb(self, a):
        """Secure least significant bit of a."""  # a la [ST06]
        stype = type(a)
        await self.returnType((stype, True))
        Zp = stype.field
        l = stype.bit_length
        k = self.options.sec_param
        f = stype.frac_length

        b = self.random_bit(stype)
        a, b = await self.gather(a, b)
        if f:
            b >>= f
        r = self._random(Zp, 1 << (l + k - 1)).value
        c = await self.output(a + ((1<<l) + (r << 1) + b.value))
        x = 1 - b if c.value & 1 else b  # xor
        if f:
            x <<= f
        return x

    @mpc_coro_no_pc
    async def mod(self, a, b):
        """Secure modulo reduction."""
        # TODO: optimize for integral a of type secfxp
        stype = type(a)
        await self.returnType(stype)
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
        await self.returnType(stype)
        Zp = stype.field
        f = stype.frac_length

        l = stype.bit_length
        k = self.options.sec_param
        r_bits = self.random._randbelow(stype, b, bits=True)
        r_bits = await self.gather(r_bits)
        r_bits = [(r >> f if f else r).value for r in r_bits]
        r_modb = 0
        for r_i in reversed(r_bits):
            r_modb <<= 1
            r_modb += r_i
        r_divb = self._random(Zp, 1 << k).value
        a = await self.gather(a)
        c = await self.output(a + ((1<<l) - ((1<<l) % b) + b * r_divb - r_modb))
        c = c.value % b

        # Secure comparison z <=> c + r_modb >= b <=> r_modb >= b - c:
        l = len(r_bits)
        s_sign = (await self.random_bits(Zp, 1, signed=True))[0].value
        e = [None] * (l+1)
        sumXors = 0
        for i in range(l-1, -1, -1):
            c_i = ((b - c) >> i) & 1
            r_i = r_bits[i]
            e[i] = Zp(s_sign + r_i - c_i + 3*sumXors)
            sumXors += 1 - r_i if c_i else r_i
        e[l] = Zp(s_sign + 1 + 3*sumXors)
        g = await self.is_zero_public(stype(self.prod(e)))
        z = Zp(1 - s_sign if g else 1 + s_sign)/2
        return (c + r_modb - z * b)<<f

    @mpc_coro
    async def trailing_zeros(self, a, l=None):
        """Secure extraction of l least significant (or all) bits of a,
        only correct up to and including the least significant 1 (if any).
        """
        secint = type(a)  # TODO: extend this to secure fixed-point numbers
        if l is None:
            l = secint.bit_length
        await self.returnType(secint, l)
        field = secint.field

        r_bits = await self.random_bits(field, l)
        r_modl = 0
        for r_i in reversed(r_bits):
            r_modl <<= 1
            r_modl += r_i.value
        k = self.options.sec_param
        r_divl = self._random(field, 1<<(secint.bit_length + k - l)).value
        a = await self.gather(a)
        c = await self.output(a + ((1<<secint.bit_length) + (r_divl << l) + r_modl))
        c = c.value % (1<<l)
        return [1-r if (c >> i)&1 else r for i, r in enumerate(r_bits)]

    def gcp2(self, a, b, l=None):
        """Secure greatest common power of 2 dividing a and b."""
        x = self.trailing_zeros(a, l=l)
        y = self.trailing_zeros(b, l=l)
        z = self.vector_sub(self.vector_add(x, y), self.schur_prod(x, y))  # bitwise or
        _, f_i = self.find(z, 1, e=None, cs_f=lambda b, i: (b+1) << i)  # 2**"index of first 1 in z"
        # TODO: consider keeping f_i in number range if z contains no 1, e.g., setting e='len(x)-1'
        return f_i

    def _iterations(self, l):
        """Number of required iterations for l-bit integers of Bernstein-Yang's divstep function."""
        return (49*l + (80 if l < 46 else 57)) // 17

    def _gcd(self, a, b, l=None):
        secint = type(a)
        if l is None:
            l = secint.bit_length

        # Step 1: remove all common factors of 2 from a and b.
        pow_of_2 = self.gcp2(a, b, l=l)
        a, b = self.scalar_mul(1/pow_of_2, [a, b])

        # Step 2: compute gcd for the case that (at least) one of the two integers is odd.
        g, f = (a%2).if_swap(a, b)
        # f is odd (or f=g=0), use stripped version of _divsteps(f, g, l) below
        delta = secint(1)
        for i in range(self._iterations(l)):
            delta_gt0 = 1 - self.sgn((delta-1-(i%2))/2, l=min(i, l).bit_length(), LT=True)
            # delta_gt0 <=> delta > 0, using |delta-1|<=min(i,l) and delta-1=i (mod 2) (for g!=0)
            g_0 = g%2
            delta, f, g = (delta_gt0 * g_0).if_else([-delta, g, -f], [delta, f, g])
            delta, g = delta+1, (g + g_0 * f)/2

        # Combine the results of both steps.
        return pow_of_2 * f

    def gcd(self, a, b, l=None):
        """Secure greatest common divisor of a and b.

        If provided, l should be an upper bound on the bit lengths of both a and b.
        """
        return self.abs(self._gcd(a, b, l=l), l=l)

    def lcm(self, a, b, l=None):
        """Secure least common multiple of a and b.

        If provided, l should be an upper bound on the bit lengths of both a and b.
        """
        g = self._gcd(a, b, l=l)
        return abs(a * (b / (g + (g == 0))))  # TODO: use l to optimize 0-test for g

    def _divsteps(self, a, b, l=None):
        """Secure extended GCD of a and b, assuming a is odd (or, a=b=0).

        Return f, v such that f = gcd(a, b) = u*a + v*b for some u.
        If f=0, then v=0 as well.

        The divstep function due to Bernstein and Yang is used, see Theorem 11.2 in
        "Fast constant-time gcd computation and modular inversion" (eprint.iacr.org/2019/266),
        however entirely avoiding the use of 2-adic arithmetic.
        """
        secint = type(a)
        if l is None:
            l = secint.bit_length
        delta, f, v, g, r = secint(1), a, secint(0), b, secint(1)
        for i in range(self._iterations(l)):
            delta_gt0 = 1 - self.sgn((delta-1-(i%2))/2, l=min(i, l).bit_length(), LT=True)
            # delta_gt0 <=> delta > 0, using |delta-1|<=min(i,l) and delta-1=i (mod 2) (for g!=0)
            g_0 = g%2
            delta, f, v, g, r = (delta_gt0 * g_0).if_else([-delta, g, r, -f, -v],
                                                          [delta, f, v, g, r])
            g, r = g_0.if_else([g + f, r + v], [g, r])  # ensure g is even
            r = (r%2).if_else(r + a, r)  # ensure r is even
            delta, g, r = delta+1, g/2, r/2
        return f, v

    def inverse(self, a, b, l=None):  # TODO: reconsider name inverse() vs invert()
        """Secure inverse of a modulo b, assuming a>=0, b>0, and gcd(a,b)=1.
        The result is nonnegative and less than b (inverse is 0 only when b=1).

        If provided, l should be an upper bound on the bit lengths of both a and b.

        To compute inverses for negative b, use -b instead of b, and
        to compute inverses for arbitrary nonzero b, use abs(b) instead of b.
        """
        c = 1 - a%2
        a, b_ = c.if_swap(a, b)  # NB: keep reference to b
        # a is odd
        g, t = self._divsteps(a, b_, l=l)
        # g == 1 or g == -1
        t = g * (t - a)
        s = (1 - t * b_) / a
        u = c.if_else(t, s)
        u = (u < 0).if_else(u + 2*b, u)
        u = (u >= b).if_else(u - b, u)
        return u

    def gcdext(self, a, b, l=None):
        """Secure extended GCD of secure integers a and b.
        Return triple (g, s, t) such that g = gcd(a,b) = s*a + t*b.

        If provided, l should be an upper bound on the bit lengths of a and b.
        """
        pow_of_2 = self.gcp2(a, b, l=l)
        a, b = self.scalar_mul(1/pow_of_2, [a, b])
        c = 1 - a%2
        a, b = c.if_swap(a, b)  # a is odd (or, a=0 if b=0 as well)
        g, t = self._divsteps(a, b, l=l)
        g0 = g%2  # NB: g0=1 <=> g odd <=> g!=0
        sgn_g = g0 - 2*self.sgn(g, l=l, LT=True)  # sign of g
        g, t = self.scalar_mul(sgn_g, [g, t])  # ensure g>=0
        s = (g - t * b) / (a + 1 - g0)  # avoid division by 0 if a=0 (and b=0)
        s, t = c.if_swap(s, t)
        # TODO: consider further reduction of coefficients s and t
        return pow_of_2 * g, s, t

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
            await self.returnType(stype)
        else:
            await self.returnType((stype, all(a.integral for a in x)))

        x = await self.gather(x)
        s = sum(a.value for a in x)
        return stype.field(s)

    @mpc_coro  # no_pc possible if no reshare and no trunc
    async def in_prod(self, x, y):
        """Secure dot product of x and y (one resharing)."""
        if x == []:
            return 0

        if x is y:
            x = y = x[:]
        else:
            x, y = x[:], y[:]
        shx = isinstance(x[0], self.SecureObject)
        shy = isinstance(y[0], self.SecureObject)
        stype = type(x[0]) if shx else type(y[0])
        f = stype.frac_length
        if not f:
            await self.returnType(stype)
        else:
            x_integral = all(a.integral for a in x)
            y_integral = all(a.integral for a in y)
            await self.returnType((stype, x_integral and y_integral))

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
        if f and (x_integral or y_integral):
            s >>= f  # NB: in-place rshift
        if shx and shy:
            s = self._reshare(s)
        if f and not (x_integral or y_integral):
            s = self.trunc(stype(s))
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
        if issubclass(sftype, self.SecureObject):
            f = sftype.frac_length
            if not f:
                await self.returnType(sftype)
            else:
                integral = [a.integral for a in x]
                await self.returnType((sftype, all(integral)))
            x = await self.gather(x)
        else:
            f = 0
            await self.returnType(Future)

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
    async def np_prod(self, x, start=1):
        """Secure product of all elements in x, similar to Python's math.prod().

        Elements of x are assumed to be arrays of the same shape.
        Runs in log_2 len(x) rounds).
        """
        # TODO: cover case of SecureArray  (incl. case f > 0
        if iter(x) is x:
            x = list(x)
        else:
            x = x[:]
        if x == []:
            return start

        x[0] = x[0] * start  # NB: also updates x[0].integral if applicable
        sftype = type(x[0])  # all elts assumed of same type and shape
        if issubclass(sftype, self.SecureObject):
            assert False
            f = sftype.frac_length
            if not f:
                await self.returnType((sftype, x[0].shape))
            else:
                integral = [a.integral for a in x]
                await self.returnType((sftype, all(integral)))
            x = await self.gather(x)
        else:
            f = 0
            await self.returnType(Future)

        n = len(x)
        while n > 1:
            h = [x[i] * x[i+1] for i in range(n%2, n, 2)]
            x[n%2:] = await self.gather([self._reshare(a) for a in h])
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
        if issubclass(sftype, self.SecureObject):
            f = sftype.frac_length
            if not f:
                await self.returnType(sftype)
            else:
                if not all(a.integral for a in x):
                    raise ValueError('nonintegral fixed-point number')

                await self.returnType((sftype, True))
            x = await self.gather(x)
        else:
            f = 0
            await self.returnType(Future)

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

    @mpc_coro
    async def np_all(self, x):
        """Secure all of elements in x, similar to Python's built-in all().

        Elements of x are assumed to be arrays of the same shape,
        containing 0s and 1s (Boolean).
        Runs in log_2 len(x) rounds.
        """
        # TODO: cover case of SecureArray  (incl. case f > 0
        if iter(x) is x:
            x = list(x)
        else:
            x = x[:]
        if x == []:
            return 1

        sftype = type(x[0])  # all elts assumed of same type and shape
        if issubclass(sftype, self.SecureObject):
            assert False  # TODO: cover this case, set shape
            f = sftype.frac_length
            if not f:
                await self.returnType(sftype)
            else:
                if not all(a.integral for a in x):
                    raise ValueError('nonintegral fixed-point number')

                await self.returnType((sftype, True))
            x = await self.gather(x)
        else:
            f = 0
            await self.returnType(Future)

        n = len(x)  # TODO: for sufficiently large n use mpc.eq(mpc.sum(x), n) instead
        while n > 1:
            h = [x[i] * x[i+1] for i in range(n%2, n, 2)]
            if f:
                for a in h:
                    a >>= f  # NB: in-place rshift
            x[n%2:] = await self.gather([self._reshare(a) for a in h])
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
            await self.returnType(stype, n)
        else:
            y0_integral = (isinstance(y[0], int) or
                           isinstance(y[0], self.SecureObject) and y[0].integral)
            await self.returnType((stype, x[0].integral and y0_integral), n)

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
            await self.returnType(stype, n)
        else:
            y0_integral = (isinstance(y[0], int) or
                           isinstance(y[0], self.SecureObject) and y[0].integral)
            await self.returnType((stype, x[0].integral and y0_integral), n)

        x, y = await self.gather(x, y)
        for i in range(n):
            x[i] = x[i] - y[i]
        return x

    @mpc_coro_no_pc
    async def matrix_add(self, A, B, tr=False):
        """Secure addition of matrices A and (transposed) B."""
        A, B = [r[:] for r in A], [r[:] for r in B]
        n1, n2 = len(A), len(A[0])
        await self.returnType(type(A[0][0]), n1, n2)
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
        await self.returnType(type(A[0][0]), n1, n2)
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
            await self.returnType(stype, n)
        else:
            a_integral = a.integral
            await self.returnType((stype, a_integral and x[0].integral), n)

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
            await self.returnType(stype, n)
        else:  # NB: a is integral
            await self.returnType((stype, x[0].integral and y[0].integral), n)

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
            await self.returnType(stype, 2, n)
        else:  # NB: a is integral
            await self.returnType((stype, x[0].integral and y[0].integral), 2, n)

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
            x = y = x[:]
        else:
            x, y = x[:], y[:]
        n = len(x)
        sftype = type(x[0])  # all elts of x and y assumed of same type
        if issubclass(sftype, self.SecureObject):
            f = sftype.frac_length
            if not f:
                await self.returnType(sftype, n)
            else:
                x_integral = x[0].integral
                y_integral = y[0].integral
                await self.returnType((sftype, x_integral and y_integral), n)
            if x is y:
                x = y = await self.gather(x)
            else:
                x, y = await self.gather(x, y)
        else:
            f = 0
            await self.returnType(Future)

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
        if A is B:
            A = B = [r[:] for r in A]
        else:
            A, B = [r[:] for r in A], [r[:] for r in B]
        shA = isinstance(A[0][0], self.SecureObject)
        shB = isinstance(B[0][0], self.SecureObject)
        stype = type(A[0][0]) if shA else type(B[0][0])
        field = stype.field
        f = stype.frac_length
        n1 = len(A)
        n2 = len(B) if tr else len(B[0])
        if not f:
            await self.returnType(stype, n1, n2)
        else:
            A_integral = A[0][0].integral
            B_integral = B[0][0].integral
            await self.returnType((stype, A_integral and B_integral), n1, n2)

        if A is B:
            A = B = await self.gather(A)
        elif shA and shB:
            A, B = await self.gather(A, B)
        elif shA:
            A = await self.gather(A)
        else:
            B = await self.gather(B)
        n = len(A[0])
        C_symmetric = A is B and tr  # C = A A^T is symmetric
        C = [None] * (n1*(n1+1)//2 if C_symmetric else n1 * n2)
        for i in range(n1):
            ni = i*(i+1)//2 if C_symmetric else i * n2
            for j in range(i+1 if C_symmetric else n2):
                s = 0
                for k in range(n):
                    s += A[i][k].value * (B[j][k] if tr else B[k][j]).value
                s = field(s)
                if f and (A_integral or B_integral):
                    s >>= f  # NB: in-place rshift
                C[ni + j] = s
        if shA and shB:
            C = await self.gather(self._reshare(C))
        if f and not A_integral and not B_integral:
            C = self.trunc(C, f=f, l=stype.bit_length)
            C = await self.gather(C)
        if C_symmetric:
            C = [[C[i*(i+1)//2 + j if j < i else j*(j+1)//2 + i]
                  for j in range(n1)] for i in range(n1)]
        else:
            C = [C[ni:ni + n2] for ni in range(0, n1 * n2, n2)]
        return C

    @mpc_coro
    async def np_matmul(self, A, B):
        shA = isinstance(A, self.SecureObject)
        shB = isinstance(B, self.SecureObject)
        stype = type(A) if shA else type(B)
        shape = np._matmul_shape(A.shape, B.shape)
        f = stype.frac_length
        if not f:
            if shape is None:
                rettype = stype.sectype
            else:
                rettype = (stype, shape)
        else:
            A_integral = A.integral
            B_integral = B.integral
            if shape is None:
                rettype = (stype.sectype, A_integral and B_integral)
            else:
                rettype = (stype, A_integral and B_integral, shape)
            # TODO: handle A or B public integral value
        await self.returnType(rettype)

        if A is B:
            A = B = await self.gather(A)
        elif shA and shB:
            A, B = await self.gather(A, B)
        elif shA:
            A = await self.gather(A)
        else:
            B = await self.gather(B)
        C = A @ B
        if f and (A_integral or B_integral):
            C >>= f  # NB: in-place rshift
        if shA and shB:
            C = self._reshare(C)
        if f and not A_integral and not B_integral:
            if shape is None:
                C = self.trunc(stype.sectype(C))
            else:
                C = self.np_trunc(stype(C, shape=shape))
        return C

    @mpc_coro
    async def np_outer(self, a, b):
        """Outer product of vectors a and b.

        Input arrays a and b are flattened if not already 1d.
        """
        sha = isinstance(a, self.SecureObject)
        shb = isinstance(b, self.SecureObject)
        stype = type(a) if sha else type(b)
        shape = (a.size, b.size)
        f = stype.frac_length
        if not f:
            rettype = (stype, shape)
        else:
            a_integral = a.integral
            b_integral = b.integral
            rettype = (stype, a_integral and b_integral, shape)
            # TODO: handle a or b public integral value
        await self.returnType(rettype)

        if a is b:
            a = b = await self.gather(a)
        elif sha and shb:
            a, b = await self.gather(a, b)
        elif sha:
            a = await self.gather(a)
        else:
            b = await self.gather(b)
        c = np.outer(a, b)  # NB: flattens a and/or b
        if f and (a_integral or b_integral):
            c >>= f  # NB: in-place rshift
        if sha and shb:
            c = self._reshare(c)
        if f and not a_integral and not b_integral:
            c = self.np_trunc(stype(c, shape=shape))
        return c

    @mpc_coro_no_pc
    async def np_getitem(self, a, key):
        """SecureArray a, index/slice key."""
        stype = type(a)
        shape = np._item_shape(a.shape, key)
        if not shape:
            stype = stype.sectype
        if issubclass(type(a), self.SecureFixedPointArray):
            if not shape:
                stype = (stype, a.integral)
            else:
                stype = (stype, a.integral, shape)
        elif shape:
            stype = (stype, shape)
        await self.returnType(stype)
        a = await self.gather(a)
        return a.__getitem__(key)

    # TODO: investigate possibility for np_setitem, for now use np_update() below, see sha3 demo

    @mpc_coro_no_pc
    async def np_update(self, a, key, value):
        """Return secure array a modified by update a[key]=value.

        Also value can be a secure array or object.
        But key is in the clear.

        Differs from __setitem__() which works in-place, returning None.
        """
        await self.returnType((type(a), a.shape))
        if isinstance(value, self.SecureObject):
            value = await self.gather(value)
        a = await self.gather(a)
        a.__setitem__(key, value)
        return a

    @mpc_coro_no_pc
    async def np_flatten(self, a, order):
        if isinstance(a, self.SecureFixedPointArray):
            assert a.integral is not None
            await self.returnType((type(a), a.integral, (a.size,)))
        else:
            await self.returnType((type(a), (a.size,)))
        a = await self.gather(a)
        return a.flatten(order)

    @mpc_coro_no_pc
    async def np_tolist(self, a):
        stype = type(a).sectype
        if issubclass(stype, self.SecureFixedPoint):
            assert a.integral is not None
            await self.returnType((stype, a.integral), *a.shape)
        else:
            await self.returnType(stype, *a.shape)
        a = await self.gather(a)
        return a.tolist()

    @mpc_coro_no_pc
    async def np_fromlist(self, x):
        """List of secure numbers to array."""
        stype = type(x[0])
        shape = (len(x),)
        if issubclass(stype, self.SecureFixedPoint):
            integral = all(a.integral for a in x)
            await self.returnType((stype.array, integral, shape))
        else:
            await self.returnType((stype.array, shape))
        x = await self.gather(x)
        return stype.field.array([a.value for a in x], check=False)

    @mpc_coro_no_pc
    async def np_reshape(self, a, shape, order='C'):
        stype = type(a)
        if isinstance(shape, int):
            shape = (shape,)  # ensure shape is a tuple
        if -1 in shape:
            if shape.count(-1) > 1:
                raise ValueError('can only specify one unknown dimension')

            if (n := a.size) % (n1 := -math.prod(shape)) != 0:
                raise ValueError(f'cannot reshape array of size {n} into shape {shape}')

            i = shape.index(-1)
            shape = list(shape)
            shape[i] = n // n1
            shape = tuple(shape)

        if issubclass(stype, self.SecureFixedPointArray):
            await self.returnType((stype, a.integral, shape))
        else:
            await self.returnType((stype, shape))
        a = await self.gather(a)
        return a.reshape(shape, order=order)

    @mpc_coro_no_pc
    async def np_copy(self, a, order='K'):
        # Note that numpy.copy() puts order='K', but ndarray.copy() puts order='C'.
        # Therefore, we put order='K' here and let SecureArray.copy() call np_copy() with order='C'.
        # TODO: a can be a scalar, should be wrapped in 0D array
        stype = type(a)
        if issubclass(stype, self.SecureFixedPointArray):
            assert a.integral is not None
            await self.returnType((stype, a.integral, a.shape))
        else:
            await self.returnType((stype, a.shape))
        a = await self.gather(a)
        return a.copy(order=order)

    @mpc_coro_no_pc
    async def np_transpose(self, a, axes=None):
        stype = type(a)
        if axes is None:
            perm = range(a.ndim)[::-1]
        else:
            perm = axes
        shape = tuple(a.shape[perm[i]] for i in range(a.ndim))
        if issubclass(stype, self.SecureFixedPointArray):
            assert a.integral is not None
            await self.returnType((stype, a.integral, shape))
        else:
            await self.returnType((stype, shape))
        a = await self.gather(a)
        return a.transpose(perm)

    @mpc_coro_no_pc
    async def np_swapaxes(self, a, axis1, axis2):
        shape = list(a.shape)
        shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
        await self.returnType((type(a), tuple(shape)))
        a = await self.gather(a)
        return a.swapaxes(axis1, axis2)

    @mpc_coro_no_pc
    async def np_concatenate(self, arrays, axis=0):
        """Join a sequence of arrays along an existing axis.

        If axis is None, arrays are flattened before use.
        Default axis is 0.
        """
        # TODO: handle array_like input arrays
        # TODO: integral attr
        if axis is None:
            shape = (sum(a.size for a in arrays),)
        else:
            shape = list(arrays[0].shape)
            # same shape for all arrays except for dimension axis
            shape[axis] = sum(a.shape[axis] for a in arrays)
            shape = tuple(shape)
        i = 0
        while not isinstance(a := arrays[i], sectypes.SecureArray):
            i += 1
        await self.returnType((type(a), shape))
        arrays = await self.gather(arrays)
        return np.concatenate(arrays, axis=axis)

    @mpc_coro_no_pc
    async def np_stack(self, arrays, axis=0):
        a = arrays[0]
        shape = list(a.shape)
        shape.insert(axis, len(arrays))
        await self.returnType((type(a), tuple(shape)))
        arrays = await self.gather(arrays)
        return np.stack(arrays, axis=axis)

    @mpc_coro_no_pc
    async def np_block(self, arrays):
        def extract_type(s):
            if isinstance(s, list):
                for a in s:
                    if cls := extract_type(a):
                        break
            elif isinstance(s, sectypes.SecureObject):
                cls = type(s)
            else:
                cls = None
            return cls

        sectype = extract_type(arrays)
        if not issubclass(sectype, sectypes.SecureArray):
            sectype = sectype.array

        def block_ndim(a, depth=0):
            if not isinstance(a, list):
                ndm = max(getattr(a, 'ndim', 0), depth)
            else:
                ndm = max(block_ndim(_, depth+1) for _ in a)
            return ndm

        def _block_shape(a, ndm):
            if not isinstance(a, list):
                shape = getattr(a, 'shape', ())
                shape = [1]*(ndm - len(shape)) + list(shape)  # pad all shapes with leading 1s
                height = 1
            else:
                # Same shape (except for axis -h) and height for all blocks in a:
                shape, h = _block_shape(a[0], ndm)
                # Sum sizes for dimension -h:
                shape[-h] = sum(_block_shape(_, ndm)[0][-h] for _ in a)
                height = h + 1
            return shape, height

        def block_shape(a):
            return tuple(_block_shape(a, block_ndim(a))[0])  # TODO: move this to mpyc.numpy module

        await self.returnType((sectype, block_shape(arrays)))
        arrays = await self.gather(arrays)  # TODO: handle secfxp
        return np.block(arrays)

    @mpc_coro_no_pc
    async def np_vstack(self, tup):
        a = tup[0]
        shape = list(a.shape) if a.ndim >= 2 else [1, a.shape[0]]
        shape[0] = sum(a.shape[0] if a.shape[1:] else 1 for a in tup)
        await self.returnType((type(a), tuple(shape)))
        tup = await self.gather(tup)
        return np.vstack(tup)

    @mpc_coro_no_pc
    async def np_hstack(self, tup):
        """Stack arrays in sequence horizontally (column wise).

        This is equivalent to concatenation along the second axis,
        except for 1-D arrays where it concatenates along the first
        axis. Rebuilds arrays divided by hsplit.
        """
        a = tup[0]
        shape = list(a.shape)
        if a.ndim == 1:
            shape[0] = sum(a.shape[0] for a in tup)
        else:
            shape[1] = sum(a.shape[1] for a in tup)
        await self.returnType((type(a), tuple(shape)))
        tup = await self.gather(tup)
        return np.hstack(tup)

    @mpc_coro_no_pc
    async def np_dstack(self, tup):
        """Stack arrays in sequence depth wise (along third axis).

        This is equivalent to concatenation along the third axis
        after 2-D arrays of shape (M,N) have been reshaped to
        (M,N,1) and 1-D arrays of shape (N,) have been reshaped
        to (1,N,1). Rebuilds arrays divided by dsplit.
        """
        a = tup[0]
        if a.ndim == 1:
            shape = (1, a.shape[0], len(tup))
        if a.ndim == 2:
            shape = (a.shape[0], a.shape[1], len(tup))
        else:
            shape = list(a.shape)
            shape[2] = sum(a.shape[2] for a in tup)
            shape = tuple(shape)
        await self.returnType((type(a), shape))
        tup = await self.gather(tup)
        return np.dstack(tup)

    @mpc_coro_no_pc
    async def np_column_stack(self, tup):
        a = tup[0]
        shape_0 = a.shape[0]
        shape_1 = sum(a.shape[1] if a.shape[1:] else 1 for a in tup)
        shape = (shape_0, shape_1)
        await self.returnType((type(a), shape))
        tup = await self.gather(tup)
        return np.column_stack(tup)

    np_row_stack = np_vstack

    @mpc_coro_no_pc
    async def np_split(self, ary, indices_or_sections, axis=0):
        """Split an array into multiple sub-arrays as views into ary."""
        shape = list(ary.shape)
        if isinstance(indices_or_sections, int):
            N = indices_or_sections
        else:
            N = indices_or_sections.shape[axis]
        shape[axis] //= N
        shape = tuple(shape)
        await self.returnType((type(ary), shape), N)
        ary = await self.gather(ary)
        return np.split(ary, indices_or_sections, axis)

    # TODO: array_split() returning arrays of different shapes -- not yet supported by returnType()
    # array_split(ary, indices_or_sections[, axis]) Split an array into multiple sub-arrays.

    def np_dsplit(self, ary, indices_or_sections):
        """Split array into multiple sub-arrays along the 3rd axis (depth)."""
        return self.np_split(ary, indices_or_sections, axis=2)

    def np_hsplit(self, ary, indices_or_sections):
        """Split an array into multiple sub-arrays horizontally (column-wise)."""
        return self.np_split(ary, indices_or_sections, axis=1)

    def np_vsplit(self, ary, indices_or_sections):
        """Split an array into multiple sub-arrays vertically (row-wise)."""
        return self.np_split(ary, indices_or_sections, axis=0)

    def np_append(self, arr, values, axis=None):
        """Append values to the end of array arr.

        If axis is None (default), arr and values are flattened first.
        Otherwise, arr and values must all be of the same shape, except along the given axis.
        """
        return self.np_concatenate((arr, values), axis=axis)

    @mpc_coro_no_pc
    async def np_fliplr(self, a):
        """Reverse the order of elements along axis 1 (left/right).

        For a 2D array, this flips the entries in each row in the left/right direction.
        Columns are preserved, but appear in a different order than before.
        """
        await self.returnType((type(a), a.shape))
        a = await self.gather(a)
        return np.fliplr(a)

    def np_minimum(self, a, b):
        return b + (a < b) * (a - b)

    def np_maximum(self, a, b):
        return a + (a < b) * (b - a)

    def np_where(self, c, a, b):
        return c * (a - b) + b

    def np_amax(self, a, axis=None):
        assert axis is None  # TODO: handle other axis (axes) and other kwargs like keepdims
        return self.max(a.flatten().tolist())

    def np_amin(self, a, axis=None):
        assert axis is None  # TODO: handle other axis (axes) and other kwargs like keepdims
        return self.min(a.flatten().tolist())

    @mpc_coro_no_pc
    async def np_sum(self, a, axis=None):
        # TODO: handle multiple axes and other kwargs like keepdims
        if axis is None:
            await self.returnType(type(a).sectype)
        else:
            shape = a.shape[:axis] + a.shape[axis+1:]
            await self.returnType((type(a), shape))
        a = await self.gather(a)
        return np.sum(a, axis=axis)

    @mpc_coro_no_pc
    async def np_roll(self, a, shift, axis=None):
        await self.returnType((type(a), a.shape))
        a = await self.gather(a)
        return np.roll(a, shift, axis)

    @mpc_coro_no_pc
    async def np_negative(self, a):
        if not a.frac_length:
            await self.returnType((type(a), a.shape))
        else:
            await self.returnType((type(a), a.integral, a.shape))
        a = await self.gather(a)
        return -a

    def np_absolute(self, a, l=None):
        """Secure absolute value of a."""
        return (-2*self.np_sgn(a, l=l, LT=True) + 1) * a

    def np_less(self, a, b):
        """Secure comparison a < b."""
        return self.np_sgn(a - b, LT=True)

    def np_equal(self, a, b):
        """Secure comparison a == b."""
        d = a - b
        stype = d.sectype
        if issubclass(stype, self.SecureFiniteField):
            return 1 - self.np_pow(d, stype.field.order - 1)

        if stype.bit_length/2 > self.options.sec_param >= 8 and stype.field.order%4 == 3:
            return self._np_is_zero(d)

        return self.np_sgn(d, EQ=True)

    @mpc_coro
    async def _np_is_zero(self, a):
        """Probabilistic zero test, elementwise."""
        stype = type(a)
        shape = a.shape
        await self.returnType((stype, True, shape))
        Zp = stype.sectype.field

        n = a.size
        k = self.options.sec_param
        z = self.np_random_bits(Zp, k * n)
        r = self._np_randoms(Zp, k * n)
        u2 = self._reshare(r * r)
        r = self._np_randoms(Zp, k * n)
        a, u2, z = await self.gather(a, u2, z)
        a = a.value.reshape((n,))
        r = r.value.reshape((k, n))
        z = z.value.reshape((k, n))
        u2 = u2.value.reshape((k, n))

        c = Zp.array(a * r + (1-(z << 1)) * u2)
        del a, r, u2
        # -1 is nonsquare for Blum p, u2[i,j] !=0 w.v.h.p.
        # If a[j] == 0, c[i,j] is square mod p iff z[i,j] == 0.
        # If a[j] != 0, c[i,j] is square mod p independent of z[i,j].
        c = await self.output(c, threshold=2*self.threshold)
        z = np.where(c.value == 0, 0, z)
        c = np.where(c.is_sqr(), 1 - z, z)
        del z
        e = await self.np_all(map(Zp.array, c))
        e <<= stype.frac_length
        e = e.reshape(shape)
        return e

    @mpc_coro
    async def np_sgn(self, a, l=None, LT=False, EQ=False):
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
        await self.returnType((stype, True, a.shape))
        Zp = stype.sectype.field

        n = a.size
        l = l or stype.sectype.bit_length
        k = self.options.sec_param
        r_bits = (await self.np_random_bits(Zp, (l + int(not EQ)) * n)).value
        if not EQ:
            s_sign = (r_bits[-n:] << 1) - 1
        r_bits = r_bits[:l*n].reshape((n, l))
        shifts = np.arange(l-1, -1, -1)
        r_modl = np.sum(r_bits << shifts, axis=1)
        r_divl = self._np_randoms(Zp, n, 1<<k).value
        a = await self.gather(a)
        a_r = a.value.reshape((n,)) + (1<<l) + r_modl
        c = await self.output(Zp.array(a_r + (r_divl << l)))
        c = c.value & ((1<<l) - 1)
        z = c - a_r

        c_bits = np.right_shift.outer(c, shifts).T & 1
        r_bits = r_bits.T
        Xor = c_bits + r_bits - (c_bits*r_bits << 1)  # shape (l, n)

        if not EQ:  # a la Toft
            zeros = np.zeros((1, n), dtype=object)
            ones = np.ones((1, n), dtype=object)
            SumXors = np.cumsum(np.vstack((zeros, Xor)), axis=0)
            if LT:
                del Xor
            e = s_sign - np.vstack((c_bits - r_bits, ones)) + 3*SumXors
            del c_bits, r_bits, SumXors
            e = self.np_prod(map(Zp.array, e))
            g = await self.np_is_zero_public(stype(e, shape=(n,)))
            h = (1 - (g << 1)) * s_sign + 3
            z = Zp.array(z + (h << l-1)) >> l

        if not LT:
            h = self.np_all(map(Zp.array, 1 - Xor))
            del Xor
            h = await self.gather(h)
            if EQ:
                z = h
            else:
                h = h.value
                z = z.value
                z = Zp.array((h - 1) * ((z << 1) - 1))
                z = await self._reshare(z)

        z <<= stype.frac_length
        z = z.reshape(a.shape)
        return z

    @mpc_coro
    async def np_det(self, A):
        """Secure determinant for nonsingular matrices."""
        # TODO: allow case det(A)=0 (obliviously)
        # TODO: support higher dimensional A than A.ndim = 2
        # TODO: support fixed-point
        secnum = type(A).sectype
        await self.returnType(secnum)

        n = A.shape[-1]
        while True:
            U = self._np_randoms(secnum.field, n**2).reshape(n, n)
            detU = self.prod(np.diag(U).tolist())  # detU != 0 with high probability
            detU = secnum(detU)
            if not await self.is_zero_public(detU):
                break

        U = U.value
        L = np.diag(np.ones(n, dtype='O'))
        L[np.triu_indices(n, 1)] = 0
        L[np.tril_indices(n, -1)] = U[np.tril_indices(n, -1)]
        U[np.tril_indices(n, -1)] = 0
        L = secnum.array(L)
        U = secnum.array(U)
        LUA = L @ (U @ A)
        LUA = await self.output(LUA, raw=True)
        detLUA = np.linalg.det(LUA)
        detA = detLUA / detU
        return detA

    @mpc_coro
    async def gauss(self, A, d, b, c):
        """Secure Gaussian elimination A d - b c."""
        n1, n2 = len(A), len(A[0])
        A, b, c = [_ for r in A for _ in r], b[:], c[:]  # flat copy of A
        stype = type(A[0])
        field = stype.field
        await self.returnType(stype, n1, n2)
        A, d, b, c = await self.gather(A, d, b, c)
        d = d.value
        for i in range(n1):
            ni = i * n2
            b_i = b[i].value
            for j in range(n2):
                A[ni + j] = field(A[ni + j].value * d - b_i * c[j].value)
        A = await self.gather(self._reshare(A))
        f = stype.frac_length
        if f:
            A = self.trunc(A, f=f, l=stype.bit_length)
            A = await self.gather(A)
        A = [A[ni:ni + n2] for ni in range(0, n1 * n2, n2)]
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
        if issubclass(sftype, self.SecureObject):
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
        if issubclass(sftype, self.SecureObject):
            shares = [sftype(s) for s in shares]
        return shares

    def _np_randoms(self, sftype, n, bound=None):
        """n secure random values of the given type in the given range."""
        if issubclass(sftype, self.SecureArray):
            field = sftype.sectype.field
        elif issubclass(sftype, self.SecureObject):
            field = sftype.field
        else:
            field = sftype
        if bound is None:
            bound = field.order
        else:
            bound = 1 << max(0, (bound // self._bincoef).bit_length() - 1)  # NB: rounded power of 2
        m = len(self.parties)
        prfs = self.prfs(bound)
        shares = thresha.np_pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
        if issubclass(sftype, self.SecureObject):
            shares = sftype(shares)
        return shares

    def random_bit(self, stype, signed=False):
        """Secure uniformly random bit of the given type."""
        return self.random_bits(stype, 1, signed)[0]

    @mpc_coro
    async def random_bits(self, sftype, n, signed=False):
        """n secure uniformly random bits of the given type."""
        prss0 = False
        if issubclass(sftype, self.SecureObject):
            if issubclass(sftype, self.SecureFiniteField):
                prss0 = True
            await self.returnType((sftype, True), n)
            field = sftype.field
            f = sftype.frac_length
        else:
            await self.returnType(Future)
            field = sftype
            f = 0

        m = len(self.parties)
        if field.characteristic == 2:
            prfs = self.prfs(2)
            bits = thresha.pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
            return bits

        bits = [None] * n
        if not signed:
            p = field.characteristic
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
                    s = r.value * field._sqrt(r2.value, INV=True)
                    if not signed:
                        s %= modulus
                        s += 1
                        s *= q
                    bits[h] = field(s << f)
        return bits

    @mpc_coro
    async def np_random_bits(self, sftype, n, signed=False):
        """Return shape-(n,) secure array of given type with uniformly random bits."""
        # TODO: extend to arbitrary shapes
        prss0 = False
        if issubclass(sftype, self.SecureObject):
            if issubclass(sftype, self.SecureFiniteField):
                prss0 = True
            await self.returnType((sftype.array, True, (n,)))
            field = sftype.field
            f = sftype.frac_length
        else:
            await self.returnType(Future)
            field = sftype
            f = 0

        m = len(self.parties)
        if field.characteristic == 2:
            prfs = self.prfs(2)
            bits = thresha.np_pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
            return bits

        if not signed:
            p = field.characteristic
            modulus = field.modulus
            q = (p+1) >> 1  # q = 1/2 mod p
        prfs = self.prfs(field.order)
        t = self.threshold
        h = n
        while h > 0:
            r = thresha.np_pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), h)
            # Compute and open the squares and compute square roots.
            r2 = r * r
            if prss0:
                z = thresha.np_pseudorandom_share_0(field, m, self.pid, prfs, self._prss_uci(), h)
                r2 += z
            r2 = await self.output(r2, threshold=2*t)
            h = 0  # TODO: handle case that r2 constains 0s, e.g.. using np.any(r2.value == 0)
            s = r.value * field.array._sqrt(r2.value, INV=True)
            if not signed:
                s %= modulus
                s += 1
                s *= q
            bits = s << f
        return field.array(bits)

    def add_bits(self, x, y):
        """Secure binary addition of bit vectors x and y."""
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
        assert l <= stype.bit_length + stype.frac_length
        await self.returnType((stype, True), l)
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

        if issubclass(stype, self.SecureFiniteField):
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

    @mpc_coro
    async def np_to_bits(self, a, l=None):
        """Secure extraction of l (or all) least significant bits of a."""  # a la [ST06].
        # TODO: other cases than characteristic=2 case
        stype = type(a).sectype
        if l is None:
            l = stype.bit_length
        assert l <= stype.bit_length + stype.frac_length
#        await self.returnType((stype, True), l)
        shape = a.shape + (l,)
        await self.returnType((type(a), True, shape))
        field = stype.field
        # f = stype.frac_length
        # rshift_f = f and a.integral  # optimization for integral fixed-point numbers
        # if rshift_f:
            # # f least significant bits of a are all 0
            # if f >= l:
                # return [field(0) for _ in range(l)]

            # l -= f

        n = a.size
        r_bits = await self.np_random_bits(field, n * l)
        r_bits = r_bits.reshape(shape)
        shifts = np.arange(l)
        r_modl = np.sum(r_bits.value << shifts, axis=a.ndim)

        # r_modl = 0
        # for r_i in reversed(r_bits):
            # r_modl <<= 1
            # r_modl += r_i.value

        if issubclass(stype, self.SecureFiniteField):
            if field.characteristic == 2:
                a = await self.gather(a)
                c = await self.output(a + r_modl)
                c = np.vectorize(int, otypes='O')(c.value)
                c_bits = np.right_shift.outer(c, shifts) & 1
                return c_bits + r_bits

            # if field.ext_deg > 1:
                # raise TypeError('Binary field or prime field required.')

            # a = self.convert(a, self.SecInt(l=1+stype.field.order.bit_length()))
            # a_bits = self.to_bits(a)
            # return self.convert(a_bits, stype)

        # k = self.options.sec_param
        # r_divl = self._random(field, 1<<(stype.bit_length + k - l)).value
        # a = await self.gather(a)
        # if rshift_f:
            # a = a >> f
        # c = await self.output(a + ((1<<stype.bit_length) + (r_divl << l) - r_modl))
        # c = c.value % (1<<l)
        # c_bits = [(c >> i) & 1 for i in range(l)]
        # r_bits = [stype(r.value) for r in r_bits]  # TODO: drop .value, fix secfxp(r) if r field elt
        # a_bits = self.add_bits(r_bits, c_bits)
        # if rshift_f:
            # a_bits = [field(0) for _ in range(f)] + a_bits
        # return a_bits

    @mpc_coro_no_pc
    async def from_bits(self, x):
        """Recover secure number from its binary representation x."""
        # TODO: also handle negative numbers with sign bit (NB: from_bits() in random.py)
        if x == []:
            return 0

        x = x[:]
        stype = type(x[0])
        await self.returnType((stype, True))
        x = await self.gather(x)
        s = 0
        for a in reversed(x):
            s <<= 1
            s += a.value
        return stype.field(s)

    @mpc_coro_no_pc
    async def np_from_bits(self, x):
        """Recover secure numbers from their binary representations in x."""
        # TODO: also handle negative numbers with sign bit (NB: from_bits() in random.py)
        *shape, l = x.shape
        await self.returnType((type(x), True, tuple(shape)))
        x = await self.gather(x)
        shifts = np.arange(l)
        s = np.sum(x.value << shifts, axis=x.ndim-1)
        return type(x).field.array(s)

    def find(self, x, a, bits=True, e='len(x)', f=None, cs_f=None):
        """Return index ix of the first occurrence of a in list x.

        The elements of x and a are assumed to be in {0, 1}, by default.
        Set Boolean flag bits=False for arbitrary inputs.

        If a is not present in x, then ix=len(x) by default.
        Set flag e=E to get ix=E instead if a is not present in x.
        Set e=None to get the "raw" output as a pair (nf, ix), where
        the indicator bit nf=1 if and only if a is not found in x.

        For instance, E=-1 can be used to mimic Python's find() methods.
        If E is a string, i=eval(E) will be returned where E is an expression
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


@dataclass
class Party:
    """Information about party with identity pid in the MPC protocol."""

    pid: int  # party identity
    host: str = None
    port: int = None
    protocol = None

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
    parser = mpyc.get_arg_parser()
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

    env_mix32_64bit = os.getenv('MPYC_MIX32_64BIT') == '1'  # check if MPYC_MIX32_64BIT is set
    if options.mix32_64bit or env_mix32_64bit:
        logging.info('Mix of parties on 32-bit and 64-bit platforms enabled.')
        from hashlib import sha1

        def hop(a):
            """Simple and portable pseudorandom program counter hop.

            Compatible across all (mixes of) 32-bit and 64-bit supported Python versions. Let's
            you run MPyC with some parties on 64-bit platforms and others on 32-bit platforms.
            Useful when working with standard 64-bit installations on Linux/MacOS/Windows and
            32-bit installations on Raspberry Pi OS, for instance.
            """
            return int.from_bytes(sha1(str(a).encode()).digest()[:8], 'little', signed=True)
        asyncoro._hop = hop

    if options.config or options.parties:
        # use host:port for each local or remote party
        addresses = []
        if options.config:
            # from ini configuration file
            config = configparser.ConfigParser()
            with open(os.path.join('.config', options.config), 'r') as f:
                config.read_file(f)
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
            for i in range(m-1, 0, -1):
                cmd_line = [sys.executable] + flags + xopts + argv + ['-I', str(i)]
                if options.output_windows and sys.platform.startswith(('win32', 'linux', 'darwin')):
                    if sys.platform.startswith('win32'):
                        subprocess.Popen(['start'] + cmd_line, shell=True)
                    elif sys.platform.startswith('linux'):
                        # TODO: check for other common Linux terminals
                        subprocess.Popen(['gnome-terminal', '--'] + cmd_line)
                    elif sys.platform.startswith('darwin'):
                        subprocess.Popen(['open', '-a', 'Terminal', '--args'] + cmd_line)
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
    mpyc.seclists.runtime = rt
    mpyc.secgroups.runtime = rt
    mpyc.random.runtime = rt
    mpyc.statistics.runtime = rt
    return rt


if os.getenv('READTHEDOCS') != 'True':
    try:
        logging.debug('Run MPyC runtime.setup()')
        mpc = setup()
    except Exception as exc:
        # suppress exceptions for pydoc etc.
        print('MPyC runtime.setup() exception:', exc)

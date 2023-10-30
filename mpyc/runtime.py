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
from mpyc.numpy import np
from mpyc import finfields
from mpyc import thresha
from mpyc import sectypes
from mpyc import asyncoro
from mpyc import mpctools
import mpyc.secgroups
import mpyc.random
import mpyc.statistics
import mpyc.seclists

Future = asyncio.Future


class Runtime:
    """MPyC runtime secure against passive attacks.

    The runtime maintains basic information such as a program counter, the list
    of parties, etc., and handles secret-shared objects of type SecureObject.

    1-party case is supported (with option to disable asynchronous evaluation).
    Threshold 0 (no corrupted parties) is supported for m-party case as well
    to enable distributed computation (without secret sharing).
    """

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
    gather = asyncoro.gather_shares
    coroutine = staticmethod(asyncoro.mpc_coro)
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
        self._threshold = t
        if self.options.no_prss:
            return

        m = len(self.parties)
        # generate new PRSS keys
        self.prfs.cache_clear()
        keys = {}
        for subset in itertools.combinations(range(m), m - t):
            if subset[0] == self.pid:
                keys[subset] = secrets.token_bytes(16)  # 128-bit key
        self._prss_keys = keys

    @functools.cache
    def prfs(self, bound):
        """PRFs with codomain range(bound) for pseudorandom secret sharing.

        Return a mapping from sets of parties to PRFs.
        """
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
        m = len(self.parties)
        if m == 1:
            self.start_time = time.time()
            return

        # m > 1
        loop = self._loop
        for peer in self.parties:
            peer.protocol = Future(loop=loop) if peer.pid == self.pid else None
        if self.options.ssl:
            import ssl  # NB: avoid "dependency" for PyScript (ssl unvendored in Pyodide)
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
                await asyncio.sleep(0.1)

        await self.parties[self.pid].protocol
        if self.options.ssl:
            logging.info(f'All {m} parties connected via SSL.')
        else:
            logging.info(f'All {m} parties connected.')
        if self.pid:
            server.close()
        self.start_time = time.time()

    async def shutdown(self):
        """Shutdown the MPyC runtime.

        Close all connections, if any.
        """
        # Wait for all parties behind a barrier.
        while self._pc_level > self._program_counter[1]:
            await asyncio.sleep(0)
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))  # format: YYYY-MM-DDTHH:MM:SS[.ffffff]
        elapsed = elapsed[:-3] if elapsed[-7] == '.' else elapsed + '.000'  # keep milliseconds .fff
        nbytes = [peer.protocol.nbytes_sent if peer.pid != self.pid else 0 for peer in self.parties]
        logging.info(f'Stop MPyC -- elapsed time: {elapsed}|bytes sent: {sum(nbytes)}')
        logging.debug(f'Bytes sent per party: {" ".join(map(str, nbytes))}')
        m = len(self.parties)
        if m == 1:
            return

        # m > 1
        self.parties[self.pid].protocol = Future(loop=self._loop)
        logging.debug('Synchronize with all parties before shutdown')
        await self.transfer(self.pid)

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

    @asyncoro.mpc_coro
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
        if isinstance(y, Future):
            return y

        if senders_is_int:
            y = y[0]
            if not x_is_list:
                y = y[0]
        else:
            if not x_is_list:
                y = [a[0] for a in y]
        return y

    @asyncoro.mpc_coro
    async def _distribute(self, x, senders):
        """Distribute shares for each x provided by a sender."""
        if x == []:
            return [[] for _ in senders]

        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, self.SecureObject):
            if hasattr(sftype, '_input'):
                return sftype._input(x, senders)

            if issubclass(sftype, self.SecureArray):
                field = sftype.sectype.field
                shape = x[0].shape  # TODO: consider multiple arrays
            else:
                field = sftype.field
                shape = None
        elif issubclass(sftype, finfields.FiniteFieldElement):
            field = sftype
            shape = None
        else:  # finfields.FiniteFieldArray
            field = sftype.field
            shape = x[0].shape

        if issubclass(sftype, self.SecureObject):
            if not sftype.frac_length:
                if shape is None:
                    rettype = sftype
                else:
                    rettype = (sftype, shape)
            else:
                if shape is None:
                    rettype = (sftype, x[0].integral)
                else:
                    rettype = (sftype, x[0].integral, shape)
            await self.returnType(rettype, len(senders), len(x))
        else:
            await self.returnType(Future)

        if shape is None or self.options.mix32_64bit:
            random_split = thresha.random_split
            marshal = field.to_bytes
            unmarshal = field.from_bytes
        else:
            random_split = thresha.np_random_split
            marshal = pickle.dumps
            unmarshal = pickle.loads
        shares = [None] * len(senders)
        for i, peer_pid in enumerate(senders):
            if peer_pid == self.pid:
                if issubclass(sftype, self.SecureObject):
                    x = await self.gather(x)
                t = self.threshold
                m = len(self.parties)
                if shape is not None:
                    x = x[0].reshape(-1)  # in-place flatten
                in_shares = random_split(field, x, t, m)
                for other_pid, data in enumerate(in_shares):
                    data = marshal(data)
                    if other_pid == self.pid:
                        shares[i] = data
                    else:
                        self._send_message(other_pid, data)
            else:
                shares[i] = self._receive_message(peer_pid)
        shares = await self.gather(shares)
        if shape is None:
            y = [[field(a) for a in unmarshal(r)] for r in shares]
        else:
            y = [[field.array(unmarshal(r), check=False).reshape(shape)] for r in shares]
        return y

    @asyncoro.mpc_coro
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
        else:  # finfields.FiniteFieldArray
            field = x[0].field
            x = x[0].value  # TODO: consider multiple arrays
            shape = x.shape
            x = x.reshape(-1)  # in-place flatten

        if shape is None or self.options.mix32_64bit:
            recombine = thresha.recombine
            marshal = field.to_bytes
            unmarshal = field.from_bytes
        else:
            recombine = thresha.np_recombine
            marshal = pickle.dumps
            unmarshal = pickle.loads
        # Send share x to all successors in receivers.
        share = None
        for peer_pid in receivers:
            if 0 < (peer_pid - self.pid) % m <= t:
                if share is None:
                    share = marshal(x)
                self._send_message(peer_pid, share)
        # Receive and recombine shares if this party is a receiver.
        if self.pid in receivers:
            shares = [self._receive_message((self.pid - t + j) % m) for j in range(t)]
            shares = await self.gather(shares)
            points = [((self.pid - t + j) % m + 1, unmarshal(shares[j])) for j in range(t)]
            points.append((self.pid + 1, x))
            y = recombine(field, points)
            if shape is None:
                y = [field(a) for a in y]
            elif self.options.mix32_64bit:
                y = [field.array(y).reshape(shape)]
            else:
                y = [y.reshape(shape)]
            if issubclass(sftype, self.SecureObject):
                f = sftype._output_conversion
                if not raw and f is not None:
                    y = [f(a) for a in y]
        else:
            y = [None] * n
        if not x_is_list:
            y = y[0]
        return y

    @asyncoro.mpc_coro
    async def _reshare(self, x):
        x_is_list = isinstance(x, list)
        if not x_is_list:
            x = [x]
        if x == []:
            return []

        sftype = type(x[0])  # all elts assumed of same type
        if issubclass(sftype, self.SecureObject):
            if not sftype.frac_length:
                if issubclass(sftype, self.SecureArray):
                    rettype = (sftype, x[0].shape)
                else:
                    rettype = sftype
            else:
                if issubclass(sftype, self.SecureArray):
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
            x = x[0].value
            shape = x.shape
            x = x.reshape(-1)  # in-place flatten

        m = len(self.parties)
        if shape is None or self.options.mix32_64bit:
            shares = thresha.random_split(field, x, t, m)
            shares = [field.to_bytes(elts) for elts in shares]
        else:
            shares = thresha.np_random_split(field, x, t, m)
            shares = [pickle.dumps(elts) for elts in shares]
        # Recombine the first 2t+1 output_shares.
        shares = self._exchange_shares(shares)
        shares = await self.gather(shares[:2*t+1])
        if shape is None or self.options.mix32_64bit:
            points = [(j+1, field.from_bytes(s)) for j, s in enumerate(shares)]
            y = thresha.recombine(field, points)
        else:
            points = [(j+1, pickle.loads(s)) for j, s in enumerate(shares)]
            y = thresha.np_recombine(field, points)
        if shape is None:
            y = [field(a) for a in y]
        elif self.options.mix32_64bit:
            y = [field.array(y).reshape(shape)]
        else:
            y = [y.reshape(shape)]
        if not x_is_list:
            y = y[0]
        return y

    def convert(self, x, t_type):
        """Secure conversion of (elements of) x to given t_type.

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

        s_type = type(x[0])  # all elts assumed of same type
        if (issubclass(s_type, self.SecureFiniteField) and
                issubclass(t_type, self.SecureFiniteField)):
            # conversion via secure integers
            size = max(s_type.field.order, t_type.field.order)
            l = max(32, size.bit_length())
            secint = self.SecInt(l=l)
            y = self._convert(self._convert(x, secint), t_type)
        else:
            y = self._convert(x, t_type)

        if not x_is_list:
            y = y[0]
        return y

    @asyncoro.mpc_coro
    async def _convert(self, x, t_type):
        s_type = type(x[0])  # source type
        n = len(x)
        await self.returnType((t_type, not s_type.frac_length), n)  # target type

        m = len(self.parties)
        t = self.threshold
        s_field = s_type.field
        t_field = t_type.field
        s_is_SecureFiniteField = issubclass(s_type, self.SecureFiniteField)
        if s_is_SecureFiniteField:
            bound = s_field.order
        else:
            k = self.options.sec_param
            l = min(s_type.bit_length, t_type.bit_length)
            if self.options.no_prss:
                bound = (1<<(k + l)) // (t+1) + 1
            else:
                bound = (1<<(k + l)) // math.comb(m, t) + 1

        if self.options.no_prss:
            uci = self._program_counter[0] % m
            senders = tuple((uci + i) % m for i in range(t+1))  # TODO: sort out load balancing
            if self.pid in senders:
                r = [secrets.randbelow(bound) for _ in range(n)]
                s_r = [s_field(a) for a in r]
                t_r = [t_field(a) for a in r]
                del r
            else:
                s_r = [s_field(0)] * n
                t_r = [t_field(0)] * n
            s_r = self.input(s_r, senders=senders)
            t_r = self.input(t_r, senders=senders)
            s_r, t_r = await self.gather(s_r, t_r)
            s_r = list(map(sum, zip(*s_r)))
            t_r = list(map(sum, zip(*t_r)))
        else:
            prfs = self.prfs(bound)
            uci = self._prss_uci()  # NB: same uci in calls for s_r and t_r
            s_r = thresha.pseudorandom_share(s_field, m, self.pid, prfs, uci, n)
            t_r = thresha.pseudorandom_share(t_field, m, self.pid, prfs, uci, n)

        x = await self.gather(x)
        d = t_type.frac_length - s_type.frac_length  # TODO: use integral attribute fxp
        if d < 0:
            x = await self.trunc(x, f=-d, l=s_type.bit_length)  # TODO: take minimum with t_type?
        if s_field.is_signed:
            if s_is_SecureFiniteField:
                offset = s_field.order // 2
            else:
                offset = 1 << l-1
        else:
            offset = 0
        for i in range(n):
            x[i] = x[i].value + offset + s_r[i]
        del s_r

        x = await self.output(x)
        for i in range(n):
            x[i] = x[i].value - t_r[i]
            if s_is_SecureFiniteField:
                x[i] = self._mod(t_type(x[i]), s_field.modulus)
            x[i] = x[i] - offset
        if d > 0 and not s_is_SecureFiniteField:
            for i in range(n):
                x[i] <<= d
        return x

    @asyncoro.mpc_coro
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
        if self.options.no_prss:
            r_divf = await r_divf
        if issubclass(sftype, self.SecureObject):
            x = await self.gather(x)
        c = await self.output([a + ((1 << l-1 + f) + (q.value << f) + r.value)
                               for a, q, r in zip(x, r_divf, r_modf)])
        c = [c.value % (1<<f) for c in c]
        y = [(a - c + r.value) >> f for a, c, r in zip(x, c, r_modf)]
        if not x_is_list:
            y = y[0]
        return y

    @asyncoro.mpc_coro
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
        r_divf = self._np_randoms(Zp, n, 1 << k + l)
        if self.options.no_prss:
            r_divf = await r_divf
        r_divf = r_divf.value
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

    @asyncoro.mpc_coro
    async def is_zero_public(self, a) -> Future:
        """Secure public zero test of a."""
        sftype = type(a)
        if issubclass(sftype, self.SecureFloat):
            return await sftype.is_zero_public(a)

        if issubclass(sftype, self.SecureObject):
            field = sftype.field
        else:
            field = sftype
        field_relative_size = field.order.bit_length() // self.options.sec_param
        if field_relative_size == 0 and self.options.no_prss:
            threshold = self.threshold  # will suffice due to reshare below
        else:
            threshold = 2 * self.threshold

        if field_relative_size >= 2:  # large fields
            r = self._random(field)  # nonzero r with high probability
            if self.options.no_prss:
                r = (await r)[0]
        else:  # small and medium-sized fields
            while True:
                r_s = self._randoms(field, 2)
                if self.options.no_prss:
                    r_s = await r_s
                r, s = r_s
                rs = r * s
                if field_relative_size == 0:  # small fields
                    if self.options.no_prss:
                        rs = await self._reshare(rs)
                    else:
                        m = len(self.parties)
                        prfs = self.prfs(field.order)
                        z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs,
                                                            self._prss_uci(), 1)
                        rs += z[0]
                if await self.output(rs, threshold=threshold):
                    break  # nonzero r ensured because rs is nonzero

        if issubclass(sftype, self.SecureObject):
            a = await self.gather(a)
        b = a * r
        if field_relative_size == 0:  # small fields
            if self.options.no_prss:
                b = await self._reshare(b)
            else:
                z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), 1)
                b += z[0]
        c = await self.output(b, threshold=threshold)
        return c == 0

    @asyncoro.mpc_coro
    async def np_is_zero_public(self, a) -> Future:
        """Secure public zero test of a, elementwise."""
        sftype = type(a)
        if issubclass(sftype, self.SecureArray):
            field = sftype.sectype.field
        else:
            field = sftype
        field_relative_size = field.order.bit_length() // self.options.sec_param
        if field_relative_size == 0 and self.options.no_prss:
            threshold = self.threshold  # will suffice due to reshare below
        else:
            threshold = 2 * self.threshold

        n = a.size
        if field_relative_size >= 2:  # large fields
            r = self._np_randoms(field, n)  # nonzero r with high probability
            if self.options.no_prss:
                r = await r
        else:  # small and medium-sized fields
            while True:
                r = self._np_randoms(field, n)
                s = self._np_randoms(field, n)
                if self.options.no_prss:
                    r, s = await self.gather(r, s)
                rs = r * s
                if field_relative_size == 0:  # small fields
                    if self.options.no_prss:
                        rs = await self._reshare(rs)
                    else:
                        m = len(self.parties)
                        prfs = self.prfs(field.order)
                        rs += thresha.np_pseudorandom_share_0(field, m, self.pid, prfs,
                                                              self._prss_uci(), n)
                if np.all(await self.output(rs, threshold=threshold)):
                    break  # nonzero r ensured because rs is nonzero
                    # TODO: handle cases with low success probability (considering alternatives
                    # such as multiplying t+1 uniformly random nonzero private input values in
                    # log_2 (t+1) rounds, or producing extra candidates such that n successes
                    # remain with high probability).

        if issubclass(sftype, self.SecureObject):
            a = await self.gather(a)
        shape = a.shape
        if len(shape) > 1:
            a = a.reshape(-1)
        b = a * r
        if field_relative_size == 0:  # small fields
            if self.options.no_prss:
                b = await self._reshare(b)
            else:
                b += thresha.np_pseudorandom_share_0(field, m, self.pid, prfs, self._prss_uci(), n)
        c = await self.output(b, threshold=threshold)
        if len(shape) > 1:
            c = c.reshape(shape)
        return c == 0

    @asyncoro.mpc_coro_no_pc
    async def neg(self, a):
        """Secure negation (additive inverse) of a."""
        stype = type(a)
        if not stype.frac_length:
            await self.returnType(stype)
        else:
            await self.returnType((stype, a.integral))
        a = await self.gather(a)
        return -a

    @asyncoro.mpc_coro_no_pc
    async def pos(self, a):
        """Secure unary + applied to a."""
        stype = type(a)
        if not stype.frac_length:
            await self.returnType(stype)
        else:
            await self.returnType((stype, a.integral))
        a = await self.gather(a)
        return +a

    @asyncoro.mpc_coro_no_pc
    async def add(self, a, b):
        """Secure addition of a and b."""
        stype = type(a)
        if not stype.frac_length:
            await self.returnType(stype)
        else:
            await self.returnType((stype, a.integral and b.integral))
        a, b = await self.gather(a, b)
        return a + b

    @asyncoro.mpc_coro_no_pc
    async def sub(self, a, b):
        """Secure subtraction of a and b."""
        stype = type(a)
        if not stype.frac_length:
            await self.returnType(stype)
        else:
            await self.returnType((stype, a.integral and b.integral))
        a, b = await self.gather(a, b)
        return a - b

    @asyncoro.mpc_coro_no_pc
    async def np_add(self, a, b):
        """Secure addition of a and b, elementwise with broadcast."""
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

    @asyncoro.mpc_coro_no_pc
    async def np_subtract(self, a, b):
        """Secure subtraction of a and b, elementwise with broadcast."""
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

    @asyncoro.mpc_coro
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

    @asyncoro.mpc_coro
    async def np_multiply(self, a, b):
        """Secure multiplication of a and b, elementwise with broadcast."""
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
        """Secure division of a and b, for nonzero b, elementwise with broadcast."""
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

    @asyncoro.mpc_coro
    async def reciprocal(self, a):
        """Secure reciprocal (multiplicative field inverse) of a, for nonzero a."""
        stype = type(a)
        await self.returnType(stype)
        field = stype.field
        while True:
            r = self._random(field)
            if self.options.no_prss:
                r = (await r)[0]
            a = await self.gather(a)
            ar = a * r
            threshold = 2 * self.threshold
            if field.order.bit_length() < self.options.sec_param:  # TODO: use separate parameter
                if self.options.no_prss:
                    ar = await self._reshare(ar)
                    threshold //= 2  # suffices due to reshare
                else:
                    m = len(self.parties)
                    prfs = self.prfs(field.order)
                    z = thresha.pseudorandom_share_zero(field, m, self.pid, prfs,
                                                        self._prss_uci(), 1)
                    ar += z[0]
            ar = await self.output(ar, threshold=threshold)
            if ar:  # happens with probability 1 - 1/field.order, which is usually close to 1
                break
        r <<= stype.frac_length  # TODO: sort out secfxp case (also see self.pow())
        return r / ar

    @asyncoro.mpc_coro
    async def np_reciprocal(self, a):
        """Secure elementwise reciprocal (multiplicative field inverse) of a, for nonzero a."""
        sftype = type(a)
        shape = a.shape
        if issubclass(sftype, self.SecureArray):
            await self.returnType((sftype, shape))
            field = sftype.sectype.field
            f = sftype.frac_length
        else:  # for recursive calls
            await self.returnType(Future)
            field = sftype.field
            f = 0

        n = a.size
        r = self._np_randoms(field, n)
        if self.options.no_prss:
            r = await r
        if issubclass(sftype, self.SecureArray):
            a = await self.gather(a)
            a = a.reshape(-1)
        ar = a * r
        threshold = 2 * self.threshold
        if field.order.bit_length() < self.options.sec_param:  # TODO: use separate parameter
            if self.options.no_prss:
                ar = await self._reshare(ar)
                threshold //= 2  # suffices due to reshare
            else:
                m = len(self.parties)
                prfs = self.prfs(field.order)
                ar += thresha.np_pseudorandom_share_0(field, m, self.pid, prfs, self._prss_uci(), n)
        ar = await self.output(ar, threshold=threshold)
        if np.count_nonzero(ar.value) < n:
            b = np.empty(n, dtype='O')
            ar_nonzero = ar != 0
            b[ar_nonzero] = (r[ar_nonzero] / ar[ar_nonzero]).value
            del r, ar
            b[~ar_nonzero] = (await self.np_reciprocal(a[~ar_nonzero])).value
            b = field.array(b, check=False)
        else:
            b = r / ar
        if f:
            b <<= f  # TODO: sort out secfxp case (also see self.pow())
        b = b.reshape(shape)
        return b

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
        """Secure elementwise exponentiation a raised to the power of b, for public integer b."""
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

    @asyncoro.mpc_coro
    async def _is_zero(self, a):  # a la [NO07]
        """Probabilistic zero test."""
        stype = type(a)
        await self.returnType((stype, True))
        Zp = stype.field
        k = self.options.sec_param

        z = self.random_bits(Zp, k)
        r = self._randoms(Zp, k)
        if self.options.no_prss:
            r = await r
        u2 = self.schur_prod(r, r)
        r = self._randoms(Zp, k)
        a, u2, z = await self.gather(a, u2, z)
        if self.options.no_prss:
            r = await r
        a = a.value
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

    @asyncoro.mpc_coro
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
        k = self.options.sec_param

        r_bits = self.random_bits(Zp, l)
        r_divl = self._random(Zp, 1<<k)
        r_bits = await r_bits
        r_modl = 0
        for r_i in reversed(r_bits):
            r_modl <<= 1
            r_modl += r_i.value
        if self.options.no_prss:
            r_divl = (await r_divl)[0]
        r_divl = r_divl.value
        a = await self.gather(a)
        a_rmodl = a + ((1<<l) + r_modl)
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
        In case of multiple occurrences of the minimum values,
        the index of the first occurrence is returned.
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
        c = key(min1) < key(min0)
        a = self.if_else(c, i1, i0)
        m = self.if_else(c, min1, min0)  # TODO: merge if_else's once integral attr per list element
        return a, m

    def argmax(self, *x, key=None):
        """Secure argmax of all given elements in x.

        See runtime.sorted() for details on key etc.
        In case of multiple occurrences of the maximum values,
        the index of the first occurrence is returned.
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
                        x[i], x[i + d] = self.if_swap(key(a) < key(b), b, a)
                d, q, r = q - p, q >> 1, p
            p >>= 1
        return x

    def np_sort(self, a, axis=-1, key=None):
        """"Returns new array sorted along given axis.

        By default, axis=-1.
        If axis is None, the array is flattened.

        Same sorting network as in self._sort().
        """
        if axis is None:
            a = self.np_flatten(a)
            axis = 0
        else:
            a = self.np_copy(a)
        if key is None:
            key = lambda a: a
        n = a.shape[axis]
        if a.size == 0 or n <= 1:
            return a

        # n >= 2
        a = self.np_swapaxes(a, axis, -1)  # switch to last axis
        t = (n-1).bit_length()
        p = 1 << t-1
        while p:
            d, q, r = p, 1 << t-1, 0
            while d:
                I = np.fromiter((i for i in range(n - d) if i & p == r), dtype=int)
                b0 = a[..., I]
                b1 = a[..., I + d]
                h = (key(b1) < key(b0)) * (b1 - b0)
                b0, b1 = b0 + h, b1 - h
                a = self.np_update(a, (..., I), b0)
                a = self.np_update(a, (..., I + d), b1)
                d, q, r = q - p, q >> 1, p
            p >>= 1
        a = self.np_swapaxes(a, axis, -1)  # restore original axis
        return a

    @asyncoro.mpc_coro
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
        r = self._random(Zp, 1 << (l + k - 1))
        if self.options.no_prss:
            r = (await r)[0]
        r = r.value
        c = await self.output(a + ((1<<l) + (r << 1) + b.value))
        x = 1 - b if c.value & 1 else b  # xor
        if f:
            x <<= f
        return x

    @asyncoro.mpc_coro_no_pc
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
        else:
            r = self._mod(a, b)
        f = stype.frac_length
        return r * 2**-f

    @asyncoro.mpc_coro
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
        r_divb = self._random(Zp, 1 << k)
        if self.options.no_prss:
            r_divb = (await r_divb)[0]
        r_divb = r_divb.value
        a = await self.gather(a)
        c = await self.output(a + ((1<<l) - ((1<<l) % b) + b * r_divb - r_modb))
        c = c.value % b
        if c == 0:
            c = b  # NB: needed if b is an integral power of 2

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

    @asyncoro.mpc_coro
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
        r_divl = self._random(field, 1<<(secint.bit_length + k - l))
        if self.options.no_prss:
            r_divl = (await r_divl)[0]
        r_divl = r_divl.value
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

    @asyncoro.mpc_coro_no_pc
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

    @asyncoro.mpc_coro  # no_pc possible if no reshare and no trunc
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

    @asyncoro.mpc_coro
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

    @asyncoro.mpc_coro
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

    def any(self, x):
        """Secure any of elements in x, similar to Python's built-in any().

        Elements of x are assumed to be either 0 or 1 (Boolean).
        Runs in log_2 len(x) rounds.
        """
        return 1 - self.all(1-a for a in x)

    def np_prod(self, a, axis=None):
        """Secure product of array elements over a given axis (or axes)."""
        if axis is None:
            # Flatten a to 1D array:
            a = self.np_reshape(a, (-1,))
        elif isinstance(axis, tuple):
            axis = tuple(i % a.ndim for i in axis)
            # Move specified axes to front:
            axes = axis + tuple(i for i in range(a.ndim) if i not in axis)
            a = self.np_transpose(a, axes=axes)
            # Flatten specified axes to one dimension:
            a = self.np_reshape(a, (-1,) + a.shape[len(axis):])
        elif axis := axis % a.ndim:
            # Move nonzero axis to front:
            axes = (axis,) + tuple(range(axis)) + tuple(range(axis+1, a.ndim))
            a = self.np_transpose(a, axes=axes)

        while (n := a.shape[0]) > 1:
            n0 = n%2
            m = a[n0:(n+1)//2] * a[(n+1)//2:]
            if n0:
                m = self.np_concatenate((a[:1], m), axis=0)
            a = m
        return a[0]

    def np_all(self, a, axis=None):
        """Secure all-predicate for array a, entirely or along the given axis (or axes).

        If axis is None (default) all is evaluated over the entire array (returning a scalar).
        If axis is an int or a tuple of ints, all is evaluated along all specified axes.
        The shape of the result is the shape of a with all specified axes removed
        (converted to a scalar if no dimensions remain).
        """
        return self.np_prod(a, axis=axis)

    def np_any(self, a, axis=None):
        """Secure any-predicate for array a, entirely or along the given axis (or axes).

        If axis is None (default) any is evaluated over the entire array (returning a scalar).
        If axis is an int or a tuple of ints, any is evaluated along all specified axes.
        The shape of the result is the shape of a with all specified axes removed
        (converted to a scalar if no dimensions remain).
        """
        return 1 - self.np_prod(1 - a, axis=axis)

    @asyncoro.mpc_coro_no_pc
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

    @asyncoro.mpc_coro_no_pc
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

    @asyncoro.mpc_coro_no_pc
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

    @asyncoro.mpc_coro_no_pc
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

    @asyncoro.mpc_coro
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
            x = await self.trunc(x, f=f, l=stype.bit_length)
        return x

    @asyncoro.mpc_coro
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
        if isinstance(c, self.SecureFixedPoint) and not c.integral:
            raise ValueError('condition must be integral')

        if x is y:  # introduced for github.com/meilof/oblif
            return x

        if isinstance(x, list):
            z = self._if_else_list(c, x, y)
        else:
            z = c * (x - y) + y
        return z

    @asyncoro.mpc_coro
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
        if isinstance(c, self.SecureFixedPoint) and not c.integral:
            raise ValueError('condition must be integral')

        if isinstance(x, list):
            z = self._if_swap_list(c, x, y)
        else:
            d = c * (y - x)
            z = [x + d, y - d]
        return z

    @asyncoro.mpc_coro
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
            x = await self.trunc(x, f=f, l=sftype.bit_length)
        return x

    @asyncoro.mpc_coro
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
            C = await self._reshare(C)
        if f and not A_integral and not B_integral:
            C = self.trunc(C, f=f, l=stype.bit_length)
            C = await self.gather(C)
        if C_symmetric:
            C = [[C[i*(i+1)//2 + j if j < i else j*(j+1)//2 + i]
                  for j in range(n1)] for i in range(n1)]
        else:
            C = [C[ni:ni + n2] for ni in range(0, n1 * n2, n2)]
        return C

    @asyncoro.mpc_coro
    async def np_matmul(self, A, B):
        """Secure matrix product of arrays A and B, with broadcast."""
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

    @asyncoro.mpc_coro
    async def np_outer(self, a, b):
        """Secure outer product of vectors a and b.

        Input arrays a and b are flattened if not already 1D.
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

    @asyncoro.mpc_coro_no_pc
    async def np_getitem(self, a, key):
        """Secure array a, index/slice key."""
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

    # TODO: investigate options for np_setitem, for now use np_update(), see sha3 demo and np_sort()

    @asyncoro.mpc_coro_no_pc
    async def np_update(self, a, key, value):
        """Return secure array modified by update a[key]=value.

        Also value can be a secure array or object.
        But key is in the clear.

        Differs from __setitem__() which works in-place, returning None.
        MUST be used as follows: a = np_update(a, key, value).
        """
        stype = type(a)
        shape = a.shape
        if issubclass(stype, self.SecureFixedPointArray):
            # TODO: value without integral attribute
            rettype = (stype, a.integral and value.integral, shape)
        else:
            rettype = (stype, shape)
        await self.returnType(rettype)

        a = await self.gather(a)
        if isinstance(value, self.SecureObject):
            value = await self.gather(value)
        a.__setitem__(key, value)
        return a

    @asyncoro.mpc_coro_no_pc
    async def np_flatten(self, a, order='C'):
        """Return 1D copy of a.

        Default 'C' for row-major order (C style).
        Alternative 'F' for column-major order (Fortran style).
        """
        if isinstance(a, self.SecureFixedPointArray):
            assert a.integral is not None
            await self.returnType((type(a), a.integral, (a.size,)))
        else:
            await self.returnType((type(a), (a.size,)))
        a = await self.gather(a)
        return a.flatten(order)

    @asyncoro.mpc_coro_no_pc
    async def np_tolist(self, a):
        """Return array a as an nested list of Python scalars.

        The nested list is a.ndim levels deep (scalar if a.ndim is zero).
        """
        stype = type(a).sectype
        if issubclass(stype, self.SecureFixedPoint):
            assert a.integral is not None
            await self.returnType((stype, a.integral), *a.shape)
        else:
            await self.returnType(stype, *a.shape)
        a = await self.gather(a)
        return a.tolist()

    @asyncoro.mpc_coro_no_pc
    async def np_fromlist(self, x):
        """List of secure numbers to 1D array."""
        stype = type(x[0])
        shape = (len(x),)
        if issubclass(stype, self.SecureFixedPoint):
            integral = all(a.integral for a in x)
            await self.returnType((stype.array, integral, shape))
        else:
            await self.returnType((stype.array, shape))
        x = await self.gather(x)
        return stype.field.array([a.value for a in x], check=False)

    @asyncoro.mpc_coro_no_pc
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

    @asyncoro.mpc_coro_no_pc
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

    @asyncoro.mpc_coro_no_pc
    async def np_transpose(self, a, axes=None):
        """Reverse (default) or permute the axes of array a.

        For 2D arrays, default result is the usual matrix transpose.
        Parameter axes can be any permutation of 0,...,n-1 for n-dimensional array a.
        """
        if a.ndim == 1:
            return a

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
        return a.transpose(axes)

    @asyncoro.mpc_coro_no_pc
    async def np_swapaxes(self, a, axis1, axis2):
        """Interchange two given axes of array a.

        For 2D arrays, same as the usual matrix transpose.
        """
        if a.ndim and axis1 - axis2 in (0, a.ndim, -a.ndim):
            return a

        stype = type(a)
        shape = list(a.shape)
        shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
        shape = tuple(shape)
        if issubclass(stype, self.SecureFixedPointArray):
            rettype = (stype, a.integral, shape)
        else:
            rettype = (stype, shape)
        await self.returnType(rettype)
        a = await self.gather(a)
        return a.swapaxes(axis1, axis2)

    @asyncoro.mpc_coro_no_pc
    async def np_concatenate(self, arrays, axis=0):
        """Join a sequence of arrays along an existing axis.

        If axis is None, arrays are flattened before use.
        Default axis is 0.
        """
        # TODO: handle array_like input arrays
        if axis is None:
            shape = (sum(a.size for a in arrays),)
        else:
            shape = list(arrays[0].shape)
            # same shape for all arrays except for dimension axis
            shape[axis] = sum(a.shape[axis] for a in arrays)
            shape = tuple(shape)
        i = 0
        while not isinstance(a := arrays[i], self.SecureArray):
            i += 1
        stype = type(a)
        if issubclass(stype, self.SecureFixedPointArray):
            integral = all(a.integral if isinstance(a, stype) else stype(a).integral
                           for a in arrays)
            await self.returnType((stype, integral, shape))
        else:
            await self.returnType((stype, shape))
        arrays = await self.gather(arrays)
        return np.concatenate(arrays, axis=axis)

    @asyncoro.mpc_coro_no_pc
    async def np_stack(self, arrays, axis=0):
        """Join a sequence of arrays along a new axis.

        The axis parameter specifies the index of the new axis in the shape of the result.
        For example, if axis=0 it will be the first dimension and if axis=-1 it will be
        the last dimension.
        """
        i = 0
        while not isinstance(a := arrays[i], self.SecureArray):
            i += 1
        shape = list(a.shape)
        shape.insert(axis, len(arrays))
        shape = tuple(shape)
        stype = type(a)
        if issubclass(stype, self.SecureFixedPointArray):
            integral = all(a.integral if isinstance(a, stype) else stype(a).integral
                           for a in arrays)
            await self.returnType((stype, integral, shape))
        else:
            await self.returnType((stype, shape))
        arrays = await self.gather(arrays)
        return np.stack(arrays, axis=axis)

    @asyncoro.mpc_coro_no_pc
    async def np_block(self, arrays):
        """Assemble an array from nested lists of blocks given by arrays.

        Blocks in the innermost lists are concatenated along the last axis,
        then these are concatenated along the second to last axis,
        and so on until the outermost list is reached.
        """
        def extract_type(s):
            if isinstance(s, list):
                for a in s:
                    if cls := extract_type(a):
                        break
            elif isinstance(s, self.SecureObject):
                cls = type(s)
            else:
                cls = None
            return cls

        sectype = extract_type(arrays)
        if not issubclass(sectype, self.SecureArray):
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

        def extract_integral(s):
            if isinstance(s, list):
                integral = all(extract_integral(a) for a in s)
            elif isinstance(s, self.SecureObject):
                integral = s.integral
            else:  # TODO: extend to cleartext numpy arrays?
                raise NotImplementedError

            return integral

        if issubclass(sectype, self.SecureFixedPointArray):
            rettype = (sectype, extract_integral(arrays), block_shape(arrays))
        else:
            rettype = (sectype, block_shape(arrays))
        await self.returnType(rettype)
        arrays = await self.gather(arrays)  # TODO: handle secfxp
        return np.block(arrays)

    @asyncoro.mpc_coro_no_pc
    async def np_vstack(self, tup):
        a = tup[0]
        stype = type(a)
        shape = list(a.shape) if a.ndim >= 2 else [1, a.shape[0]]
        shape[0] = sum(a.shape[0] if a.shape[1:] else 1 for a in tup)

        shape = tuple(shape)
        if issubclass(stype, self.SecureFixedPointArray):
            integral = all(a.integral if isinstance(a, stype) else stype(a).integral for a in tup)
            await self.returnType((stype, integral, shape))
        else:
            await self.returnType((stype, shape))
        tup = await self.gather(tup)
        return np.vstack(tup)

    @asyncoro.mpc_coro_no_pc
    async def np_hstack(self, tup):
        """Stack arrays in sequence horizontally (column wise).

        This is equivalent to concatenation along the second axis,
        except for 1D arrays where it concatenates along the first
        axis. Rebuilds arrays divided by hsplit.
        """
        i = 0
        while not isinstance(a := tup[i], self.SecureArray):
            i += 1
        stype = type(a)
        shape = list(a.shape)
        if a.ndim == 1:
            shape[0] = sum(a.shape[0] for a in tup)
        else:
            shape[1] = sum(a.shape[1] for a in tup)
        shape = tuple(shape)
        if issubclass(stype, self.SecureFixedPointArray):
            integral = all(a.integral if isinstance(a, stype) else stype(a).integral for a in tup)
            await self.returnType((stype, integral, shape))
        else:
            await self.returnType((stype, shape))
        tup = await self.gather(tup)
        return np.hstack(tup)

    @asyncoro.mpc_coro_no_pc
    async def np_dstack(self, tup):
        """Stack arrays in sequence depth wise (along third axis).

        This is equivalent to concatenation along the third axis
        after 2D arrays of shape (M,N) have been reshaped to
        (M,N,1) and 1D arrays of shape (N,) have been reshaped
        to (1,N,1). Rebuilds arrays divided by dsplit.
        """
        a = tup[0]
        if a.ndim == 1:
            shape = (1, a.shape[0], len(tup))
        elif a.ndim == 2:
            shape = (a.shape[0], a.shape[1], len(tup))
        else:
            shape = list(a.shape)
            shape[2] = sum(a.shape[2] for a in tup)
            shape = tuple(shape)
        await self.returnType((type(a), shape))
        tup = await self.gather(tup)
        return np.dstack(tup)

    @asyncoro.mpc_coro_no_pc
    async def np_column_stack(self, tup):
        i = 0
        while not isinstance(a := tup[i], self.SecureArray):
            i += 1
        stype = type(a)
        shape_0 = a.shape[0]
        shape_1 = sum(a.shape[1] if a.shape[1:] else 1 for a in tup)
        shape = (shape_0, shape_1)
        if issubclass(stype, self.SecureFixedPointArray):
            integral = all(a.integral if isinstance(a, stype) else stype(a).integral for a in tup)
            await self.returnType((stype, integral, shape))
        else:
            await self.returnType((stype, shape))
        tup = await self.gather(tup)
        return np.column_stack(tup)

    np_row_stack = np_vstack

    @asyncoro.mpc_coro_no_pc
    async def np_split(self, ary, indices_or_sections, axis=0):
        """Split an array into multiple sub-arrays as views into ary."""
        stype = type(ary)
        shape = list(ary.shape)
        if isinstance(indices_or_sections, int):
            N = indices_or_sections
        else:
            N = indices_or_sections.shape[axis]
        shape[axis] //= N
        shape = tuple(shape)
        if issubclass(stype, self.SecureFixedPointArray):
            rettype = (stype, ary.integral, shape)
        else:
            rettype = (stype, shape)
        await self.returnType(rettype, N)
        ary = await self.gather(ary)
        return np.split(ary, indices_or_sections, axis=axis)

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

    @asyncoro.mpc_coro_no_pc
    async def np_fliplr(self, a):
        """Reverse the order of elements along axis 1 (left/right).

        For a 2D array, this flips the entries in each row in the left/right direction.
        Columns are preserved, but appear in a different order than before.
        """
        stype = type(a)
        if issubclass(stype, self.SecureFixedPointArray):
            rettype = (stype, a.integral, a.shape)
        else:
            rettype = (stype, a.shape)
        await self.returnType(rettype)
        a = await self.gather(a)
        return np.fliplr(a)

    def np_minimum(self, a, b):
        """Secure elementwise minimum of a and b.

        If a and b are of different shapes, they must be broadcastable to a common shape
        (which is scalar if both a and b are scalars).
        """
        return b + (a < b) * (a - b)

    def np_maximum(self, a, b):
        """Secure elementwise maximum of a and b.

        If a and b are of different shapes, they must be broadcastable to a common shape
        (which is scalar if both a and b are scalars).
        """
        return a + (a < b) * (b - a)

    def np_where(self, c, a, b):
        """Return elements chosen from a or b depending on condition c.

        The shapes of a, b, and c are broadcast together.
        """
        return c * (a - b) + b

    def np_amin(self, a, axis=None, keepdims=False):
        """Secure minimum of array a, entirely or along the given axis (or axes).

        If axis is None (default) the minimum of the array is returned.
        If axis is an int or a tuple of ints, the minimum along all specified axes is returned.

        If keepdims is not set (default), the shape of the result is the shape of a with all
        specified axes removed (converted to a scalar if no dimensions remain).
        Otherwise, if keepdims is set, the axes along which the minimum is taken
        are left in the result as dimensions of size 1.
        """
        # TODO: other kwargs like initial
        if axis is None:
            if keepdims:
                shape = [1] * a.ndim
            # Flatten a to 1D array:
            a = self.np_reshape(a, (-1,))
        elif isinstance(axis, tuple):
            axis = tuple(i % a.ndim for i in axis)
            if keepdims:
                shape = list(a.shape)
                for i in axis:
                    shape[i] = 1
            # Move specified axes to front:
            axes = axis + tuple(i for i in range(a.ndim) if i not in axis)
            a = self.np_transpose(a, axes=axes)
            # Flatten specified axes to one dimension:
            a = self.np_reshape(a, (-1,) + a.shape[len(axis):])
        elif axis := axis % a.ndim:
            if keepdims:
                shape = list(a.shape)
                shape[axis] = 1
            # Move nonzero axis to front:
            axes = (axis,) + tuple(range(axis)) + tuple(range(axis+1, a.ndim))
            a = self.np_transpose(a, axes=axes)

        while (n := a.shape[0]) > 1:
            n0 = n%2
            a1, a2 = a[n0:(n+1)//2], a[(n+1)//2:]
            m = a1 + (a2 < a1) * (a2 - a1)
            if n0:
                m = self.np_concatenate((a[:1], m), axis=0)
            a = m

        if keepdims:
            return self.np_reshape(a, tuple(shape))

        return a[0]

    def np_amax(self, a, axis=None, keepdims=False):
        """Secure maximum of array a, entirely or along the given axis (or axes).

        If axis is None (default) the maximum of the array is returned.
        If axis is an int or a tuple of ints, the minimum along all specified axes is returned.

        If keepdims is not set (default), the shape of the result is the shape of a with all
        specified axes removed (converted to a scalar if no dimensions remain).
        Otherwise, if keepdims is set, the axes along which the maximum is taken
        are left in the result as dimensions of size 1.
        """
        # TODO: other kwargs like initial
        if axis is None:
            if keepdims:
                shape = [1] * a.ndim
            # Flatten a to 1D array:
            a = self.np_reshape(a, (-1,))
        elif isinstance(axis, tuple):
            axis = tuple(i % a.ndim for i in axis)
            if keepdims:
                shape = list(a.shape)
                for i in axis:
                    shape[i] = 1
            # Move specified axes to front:
            axes = axis + tuple(i for i in range(a.ndim) if i not in axis)
            a = self.np_transpose(a, axes=axes)
            # Flatten specified axes to one dimension:
            a = self.np_reshape(a, (-1,) + a.shape[len(axis):])
        elif axis := axis % a.ndim:
            if keepdims:
                shape = list(a.shape)
                shape[axis] = 1
            # Move nonzero axis to front:
            axes = (axis,) + tuple(range(axis)) + tuple(range(axis+1, a.ndim))
            a = self.np_transpose(a, axes=axes)

        while (n := a.shape[0]) > 1:
            n0 = n%2
            a1, a2 = a[n0:(n+1)//2], a[(n+1)//2:]
            m = a1 + (a1 < a2) * (a2 - a1)
            if n0:
                m = self.np_concatenate((a[:1], m), axis=0)
            a = m

        if keepdims:
            return self.np_reshape(a, tuple(shape))

        return a[0]

    @asyncoro.mpc_coro_no_pc
    async def np_sum(self, a, axis=None, keepdims=False, initial=0):
        """Secure sum of array elements over a given axis (or axes)."""
        sectype = type(a).sectype
        if not isinstance(initial, sectype):
            initial = sectype(initial)
        if axis is None:
            if keepdims:
                shape = (1,) * a.ndim
            else:
                shape = ()
        else:
            if not isinstance(axis, tuple):
                axis = (axis,)
            axes = [i % a.ndim for i in axis]
            if keepdims:
                shape = tuple(s if i not in axes else 1 for i, s in enumerate(a.shape))
            else:
                shape = tuple(s for i, s in enumerate(a.shape) if i not in axes)
        if shape == ():
            if isinstance(a, self.SecureFixedPointArray):
                rettype = (sectype, a.integral)
            else:
                rettype = sectype
        else:
            if isinstance(a, self.SecureFixedPointArray):
                rettype = (type(a), a.integral, shape)
            else:
                rettype = (type(a), shape)
        await self.returnType(rettype)
        a, initial = await self.gather(a, initial)
        return np.sum(a, axis=axis, keepdims=keepdims, initial=initial.value)
        # TODO: handle switch from initial (field elt) to initial.value inside finfields.py

    @asyncoro.mpc_coro_no_pc
    async def np_roll(self, a, shift, axis=None):
        """Roll array elements (cyclically) along a given axis.

        If axis is None (default), array is flattened before cyclic shift,
        and original shape is restored afterwards.
        """
        await self.returnType((type(a), a.shape))
        a = await self.gather(a)
        return np.roll(a, shift, axis=axis)

    @asyncoro.mpc_coro_no_pc
    async def np_negative(self, a):
        """Secure elementwise negation -a (additive inverse) of a."""
        if not a.frac_length:
            await self.returnType((type(a), a.shape))
        else:
            await self.returnType((type(a), a.integral, a.shape))
        a = await self.gather(a)
        return -a

    def np_absolute(self, a, l=None):
        """Secure elementwise absolute value of a."""
        return (-2*self.np_sgn(a, l=l, LT=True) + 1) * a

    def np_less(self, a, b):
        """Secure comparison a < b, elementwise with broadcast."""
        return self.np_sgn(a - b, LT=True)

    def np_equal(self, a, b):
        """Secure comparison a == b, elementwise with broadcast."""
        d = a - b
        stype = d.sectype
        if issubclass(stype, self.SecureFiniteField):
            return 1 - self.np_pow(d, stype.field.order - 1)

        if stype.bit_length/2 > self.options.sec_param >= 8 and stype.field.order%4 == 3:
            return self._np_is_zero(d)

        return self.np_sgn(d, EQ=True)

    @asyncoro.mpc_coro
    async def _np_is_zero(self, a):
        """Probabilistic elementwise zero test of array a.

        Return 1 if a == 0 else 0.
        """
        stype = type(a)
        shape = a.shape
        await self.returnType((stype, True, shape))
        Zp = stype.sectype.field
        n = a.size
        k = self.options.sec_param

        z = self.np_random_bits(Zp, k * n)
        r = self._np_randoms(Zp, k * n)
        if self.options.no_prss:
            r = await r
        u2 = self._reshare(r * r)
        r = self._np_randoms(Zp, k * n)
        if self.options.no_prss:
            r = await r
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
        e = self.np_all(stype(c), axis=0)
        e = await self.gather(e)
        e = e.reshape(shape)
        return e

    @asyncoro.mpc_coro
    async def np_sgn(self, a, l=None, LT=False, EQ=False):
        """Secure elementwise sign(um) of array a.

        Return -1 if a < 0 else 0 if a == 0 else 1.

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

        r_bits = self.np_random_bits(Zp, (l + int(not EQ)) * n)
        r_divl = self._np_randoms(Zp, n, 1<<k)
        r_bits = (await r_bits).value
        if not EQ:
            s_sign = (r_bits[-n:] << 1) - 1
        r_bits = r_bits[:l*n].reshape((n, l))
        shifts = np.arange(l-1, -1, -1)
        r_modl = np.sum(r_bits << shifts, axis=1)
        if self.options.no_prss:
            r_divl = await r_divl
        r_divl = r_divl.value
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
            e = self.np_prod(stype(e), axis=0)
            g = await self.np_is_zero_public(e)
            h = (1 - (g << 1)) * s_sign + 3
            z = Zp.array(z + (h << l-1)) >> l

        if not LT:
            h = self.np_all(stype(1 - Xor), axis=0)
            del Xor
            h = await self.gather(h)
            h >>= stype.frac_length
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

    def np_argmin(self, a, axis=None, keepdims=False, key=None, arg_unary=False, arg_only=True):
        """Returns the indices of the minimum values along an axis.

        Default behavior similar to np.argmin() for NumPy arrays:

         - the indices are returned as numbers (not as unit vectors),
         - only the indices are returned (minimum values omitted).

        NB: Different defaults than for method call a.argmin().

        If no axis is given (default), array a is flattened first.

        If the indices are returned as unit vectors in an array u say,
        then u is always of the same shape as the (possibly flattened) input array a.

        If the indices are returned as numbers, the shape of the array of indices
        is controlled by parameter keepdims. If keepdims is not set (default),
        the shape of the indices is the shape of a with the specified axis removed
        (converted to a scalar if no dimensions remain). Otherwise, if keepdims is
        set, the axis along which the minimum is taken is left in the result as
        dimension of size 1.

        If the minimum values are returned as well, the shape of this part of the output is
        also controlled by parameter keepdims. If keepdims is not set (default), a 1D array
        of minimum values is returned with one entry per element of the given array a with
        the given axis removed; if axis is None, the minimum is returned as a scalar.
        Otherwise, if keepdims is set, the array of minimum values is of the same shape
        as the given array a except that the dimension of the given axis is reduced to 1;
        if axis is None, all axes are present as dimensions of size 1.
        """
        if key is None:
            key = lambda a: a

        key_size = getattr(key, 'size', 1)
        assert key_size in (1, a.shape[-1])

        shape = a.shape
        ndim = a.ndim - key_size + 1  # number of dimensions corrected for key size
        if axis is None:
            if key_size == 1:
                a = self.np_reshape(a, (1, a.size))
            else:
                a = self.np_reshape(a, (1, a.size // key_size, key_size))
        else:
            if key_size == 1:
                # Move axis to last position:
                a = self.np_swapaxes(a, axis, -1)
                # Collapse all dimensions except the last one:
                a = self.np_reshape(a, (-1, a.shape[-1]))
            else:
                # Last axis is of dimension equal to key size, and not allowed as value for axis:
                assert (axis + 1) % a.ndim != 0
                # Move axis to last but one position:
                a = self.np_swapaxes(a, axis, -2)
                # Collapse all dimensions except the last two:
                a = self.np_reshape(a, (-1, a.shape[-2], key_size))
        u, m = self._np_argmin(a, key)
        # u is a 2D array
        if not arg_unary:
            iv = np.arange(u.shape[1])
            if isinstance(a, self.SecureFixedPointArray):
                iv = type(a)(iv)  # TODO: remove once @ handles integral attrb for public values
            u = u @ iv
        if axis is None:
            if not arg_unary and keepdims:
                # convert shape u from (1,) to (1,...,1)
                u = self.np_reshape(u, (1,) * ndim)
            else:
                u = u[0]
        else:
            shape = list(shape)
            if key_size > 1:
                del shape[-1]
            if arg_unary:
                shape[axis], shape[-1] = shape[-1], shape[axis]
            else:
                if keepdims:
                    shape[axis] = 1
                else:
                    del shape[axis]
            u = self.np_reshape(u, tuple(shape))
            if arg_unary:
                u = self.np_swapaxes(u, axis, -1)
        if arg_only:
            return u

        if axis is None:
            if keepdims:
                m = self.np_reshape(m, (1,) * ndim)
            else:
                m = m[0][0]
        elif keepdims:
            if arg_unary:
                shape[axis], shape[-1] = shape[-1], shape[axis]  # NB: restore shape
                shape[axis] = 1
            m = self.np_reshape(m, tuple(shape))
        return u, m

    def _np_argmin(self, a, key):
        """Return first occurrence of minimum (as unit vector) and minimum itself."""
        n = a.shape[1]
        if n == 1:
            u = type(a)(np.array([[1]]))
            m = a
        elif n == 2:
            # Redundant case, except for some small savings.
            a1, a2 = a[:, :1], a[:, 1:]
            c = key(a2) < key(a1)
            u = self.np_concatenate((1 - c, c), axis=1)
            m = c * (a2 - a1) + a1
        else:
            n0 = n%2
            a1, a2 = a[:, n0::2], a[:, n0 + 1::2]  # NB: odd-even split to return first occurrence
            c = key(a2) < key(a1)
            cc = c if c.ndim == a.ndim else c.reshape(*c.shape, 1)  # TODO: use c[..., np.newaxis]?
            m = cc * (a2 - a1) + a1
            if n0:
                m = self.np_concatenate((a[:, :1], m), axis=1)
            u, m = self._np_argmin(m, key)
            if n0:
                u0, u = u[:, :1], u[:, 1:]
            u2 = u * c
            u = self.np_concatenate((u - u2, u2), axis=0)
            u = self.np_reshape(u, (len(c), 2*c.shape[1]), order='F')
            if n0:
                u = self.np_concatenate((u0, u), axis=1)
        return u, m

    def np_argmax(self, a, axis=None, keepdims=False, key=None, arg_unary=False, arg_only=True):
        """Returns the indices of the maximum values along an axis.

        Default behavior similar to np.argmax() for NumPy arrays:

         - the indices are returned as numbers (not as unit vectors),
         - only the indices are returned (maximum values omitted).

        NB: Different defaults than for method call a.argmax().

        If no axis is given (default), array a is flattened first.

        If the indices are returned as unit vectors in an array u say,
        then u is always of the same shape as the (possibly flattened) input array a.

        If the indices are returned as numbers, the shape of the array of indices
        is controlled by parameter keepdims. If keepdims is not set (default),
        the shape of the indices is the shape of a with the specified axis removed
        (converted to a scalar if no dimensions remain). Otherwise, if keepdims is
        set, the axis along which the maximum is taken is left in the result as
        dimension of size 1.

        If the maximum values are returned as well, the shape of this part of the output is
        also controlled by parameter keepdims. If keepdims is not set (default), a 1D array
        of maximum values is returned with one entry per element of the given array a with
        the given axis removed; if axis is None, the maximum is returned as a scalar.
        Otherwise, if keepdims is set, the array of maximum values is of the same shape
        as the given array a except that the dimension of the given axis is reduced to 1;
        if axis is None, all axes are present as dimensions of size 1.
        """
        if key is None:
            key = lambda a: a

        key_size = getattr(key, 'size', 1)
        assert key_size in (1, a.shape[-1])

        shape = a.shape
        ndim = a.ndim - key_size + 1  # number of dimensions corrected for key size
        if axis is None:
            if key_size == 1:
                a = self.np_reshape(a, (1, a.size))
            else:
                a = self.np_reshape(a, (1, a.size // key_size, key_size))
        else:
            if key_size == 1:
                # Move axis to last position:
                a = self.np_swapaxes(a, axis, -1)
                # Collapse all dimensions except the last one:
                a = self.np_reshape(a, (-1, a.shape[-1]))
            else:
                # Last axis is of dimension equal to key size, and not allowed as value for axis:
                assert (axis + 1) % a.ndim != 0
                # Move axis to last but one position:
                a = self.np_swapaxes(a, axis, -2)
                # Collapse all dimensions except the last two:
                a = self.np_reshape(a, (-1, a.shape[-2], key_size))
        u, m = self._np_argmax(a, key)
        # u is a 2D array
        if not arg_unary:
            iv = np.arange(u.shape[1])
            if isinstance(a, self.SecureFixedPointArray):
                iv = type(a)(iv)  # TODO: remove once @ handles integral attrb for public values
            u = u @ iv
        if axis is None:
            if not arg_unary and keepdims:
                # convert shape u from (1,) to (1,...,1)
                u = self.np_reshape(u, (1,) * ndim)
            else:
                u = u[0]
        else:
            shape = list(shape)
            if key_size > 1:
                del shape[-1]
            if arg_unary:
                shape[axis], shape[-1] = shape[-1], shape[axis]
            else:
                if keepdims:
                    shape[axis] = 1
                else:
                    del shape[axis]
            u = self.np_reshape(u, tuple(shape))
            if arg_unary:
                u = self.np_swapaxes(u, axis, -1)
        if arg_only:
            return u

        if axis is None:
            if keepdims:
                m = self.np_reshape(m, (1,) * ndim)
            else:
                m = m[0][0]
        elif keepdims:
            if arg_unary:
                shape[axis], shape[-1] = shape[-1], shape[axis]  # NB: restore shape
                shape[axis] = 1
            m = self.np_reshape(m, tuple(shape))
        return u, m

    def _np_argmax(self, a, key):
        """Return first occurrence of maximum (as unit vector) and maximum itself."""
        n = a.shape[1]
        if n == 1:
            u = type(a)(np.array([[1]]))
            m = a
        elif n == 2:
            # Redundant case, except for some small savings.
            a1, a2 = a[:, :1], a[:, 1:]
            c = key(a1) < key(a2)
            u = self.np_concatenate((1 - c, c), axis=1)
            m = c * (a2 - a1) + a1
        else:
            n0 = n%2
            a1, a2 = a[:, n0::2], a[:, n0 + 1::2]  # NB: odd-even split to return first occurrence
            c = key(a1) < key(a2)
            cc = c if c.ndim == a.ndim else c.reshape(*c.shape, 1)  # TODO: use c[..., np.newaxis]?
            m = cc * (a2 - a1) + a1
            if n0:
                m = self.np_concatenate((a[:, :1], m), axis=1)
            u, m = self._np_argmax(m, key)
            if n0:
                u0, u = u[:, :1], u[:, 1:]
            u2 = u * c
            u = self.np_concatenate((u - u2, u2), axis=0)
            u = self.np_reshape(u, (len(c), 2*c.shape[1]), order='F')
            if n0:
                u = self.np_concatenate((u0, u), axis=1)
        return u, m

    @asyncoro.mpc_coro
    async def np_det(self, A):
        """Secure determinant for nonsingular matrices."""
        # TODO: allow case det(A)=0 (obliviously)
        # TODO: support higher dimensional A than A.ndim = 2
        # TODO: support fixed-point
        secnum = type(A).sectype
        await self.returnType(secnum)

        n = A.shape[-1]
        while True:
            U = self._np_randoms(secnum.field, n**2)
            if self.options.no_prss:
                U = await U
            U = U.reshape(n, n)
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

    @asyncoro.mpc_coro
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
        A = await self._reshare(A)
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
        x = self._randoms(sftype, 1, bound)
        if not isinstance(x, Future):
            x = x[0]
        return x

    def _randoms(self, sftype, n, bound=None):
        """Return n secure random values of the given type in the given range."""
        if issubclass(sftype, self.SecureObject):
            field = sftype.field
        else:
            field = sftype
        m = len(self.parties)
        t = self.threshold
        if bound is None:
            bound = field.order
        else:
            d = t+1 if self.options.no_prss else math.comb(m, t)
            bound = 1 << max(0, (bound // d).bit_length() - 1)  # NB: rounded power of 2
        if self.options.no_prss:
            uci = self._program_counter[0] % m
            senders = tuple((uci + i) % m for i in range(t+1))  # TODO: sort out load balancing
            if self.pid in senders:
                x = [field(secrets.randbelow(bound)) for _ in range(n)]
            else:
                x = [field(0)] * n
            x = self.input(x, senders=senders)

            @asyncoro.mpc_coro_no_pc
            async def add_shares(x):
                if issubclass(sftype, self.SecureObject):
                    await self.returnType(sftype, n)
                else:
                    await self.returnType(Future)
                x = await x
                x = [field(sum(a.value for a in _)) for _ in zip(*x)]
                return x

            return add_shares(x)

        x = thresha.pseudorandom_share(field, m, self.pid, self.prfs(bound), self._prss_uci(), n)
        if issubclass(sftype, self.SecureObject):
            x = [sftype(s) for s in x]
        return x

    def _np_randoms(self, sftype, n, bound=None):
        """Secure shape-(n,) array with random values of the given type in the given range."""
        # TODO: extend to arbitrary shapes
        if issubclass(sftype, self.SecureObject):
            field = sftype.field
        else:
            field = sftype
        m = len(self.parties)
        t = self.threshold
        if bound is None:
            bound = field.order
        else:
            d = t+1 if self.options.no_prss else math.comb(m, t)
            bound = 1 << max(0, (bound // d).bit_length() - 1)  # NB: rounded power of 2
        if self.options.no_prss:
            uci = self._program_counter[0] % m
            senders = tuple((uci + i) % m for i in range(t+1))  # TODO: sort out load balancing
            if self.pid in senders:
                x = field.array([secrets.randbelow(bound) for _ in range(n)])
            else:
                x = field.array(np.zeros(n, dtype=object), check=False)
            x = self.input(x, senders=senders)

            @asyncoro.mpc_coro_no_pc
            async def add_shares(x):
                if issubclass(sftype, self.SecureObject):
                    await self.returnType((sftype.array, (n,)))
                else:
                    await self.returnType(Future)
                x = await x
                x = (a[0].value for a in x)
                x = np.sum(np.fromiter(x, 'O', count=t+1))
                x = field.array(x)
                return x

            return add_shares(x)

        x = thresha.np_pseudorandom_share(field, m, self.pid, self.prfs(bound), self._prss_uci(), n)
        if issubclass(sftype, self.SecureObject):
            x = sftype.array(x)
        return x

    def random_bit(self, stype, signed=False):
        """Secure uniformly random bit of the given type."""
        return self.random_bits(stype, 1, signed)[0]

    @asyncoro.mpc_coro
    async def random_bits(self, sftype, n, signed=False):
        """Return n secure uniformly random bits of the given type."""
        if issubclass(sftype, self.SecureObject):
            await self.returnType((sftype, True), n)
            field = sftype.field
            f = sftype.frac_length
        else:
            await self.returnType(Future)
            field = sftype
            f = 0
        if not n:
            return []

        m = len(self.parties)
        t = self.threshold
        p = field.characteristic
        if p == 2:
            if self.options.no_prss:
                uci = self._program_counter[0] % m
                senders = tuple((uci + i) % m for i in range(t+1))  # TODO: sort out load balancing
                if self.pid in senders:
                    bits = [field(secrets.randbits(1)) for _ in range(n)]
                else:
                    bits = [field(0)] * n
                bits = self.input(bits, senders=senders)
                bits = await bits
                bits = list(map(sum, zip(*bits)))
                return bits

            prfs = self.prfs(2)
            bits = thresha.pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
            return bits

        if self.options.no_prss:
            # Multiply t+1 uniformly random ±1 private input values in log_2 (t+1) rounds.
            # Alternative: uniformly random secret value r, squared and opened as in PRSS case
            # in 3 rounds, with break-even point at t=7, hence advantageous for m >= 15.
            uci = self._program_counter[0] % m
            senders = tuple((uci + i) % m for i in range(t+1))  # TODO: sort out load balancing
            if self.pid in senders:
                bits = [field(2*secrets.randbits(1)-1) for _ in range(n)]
            else:
                bits = [field(0)] * n
            bits = self.input(bits, senders=senders)
            bits = await bits
            bits = list(map(list, zip(*bits)))
            bits = [self.prod(x) for x in bits]
            bits = await self.gather(bits)
            for i in range(n):
                bits[i] = bits[i].value
        else:
            bits = [None] * n
            prfs = self.prfs(field.order)
            h = n
            while h > 0:
                rs = thresha.pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), h)
                zs = thresha.pseudorandom_share_zero(field, m, self.pid, prfs, self._prss_uci(), h)
                # Compute and open the squares and compute square roots.
                r2s = [field(r.value**2 + z.value) for r, z in zip(rs, zs)]
                r2s = await self.output(r2s, threshold=2*t)
                for r, r2 in zip(rs, r2s):
                    if r2.value != 0:
                        h -= 1
                        bits[h] = r.value * field._sqrt(r2.value, INV=True)
                        if not signed:
                            bits[h] %= field.modulus

        if not signed:
            q = (p+1) >> 1  # q = 1/2 mod p
            for i in range(n):
                bits[i] = (bits[i] + 1) * q
        for i in range(n):
            if f:
                bits[i] <<= f
            bits[i] = field(bits[i])
        return bits

    @asyncoro.mpc_coro
    async def np_random_bits(self, sftype, n, signed=False):
        """Return shape-(n,) secure array with uniformly random bits of given type."""
        # TODO: extend to arbitrary shapes
        if issubclass(sftype, self.SecureObject):
            await self.returnType((sftype.array, True, (n,)))
            field = sftype.field
            f = sftype.frac_length
        else:
            await self.returnType(Future)
            field = sftype
            f = 0
        if not n:
            return field.array([])

        m = len(self.parties)
        t = self.threshold
        p = field.characteristic
        if p == 2:
            if self.options.no_prss:
                uci = self._program_counter[0] % m
                senders = tuple((uci + i) % m for i in range(t+1))  # TODO: sort out load balancing
                if self.pid in senders:
                    bits = field.array([secrets.randbits(1) for _ in range(n)], check=False)
                else:
                    bits = field.array(np.zeros(n, dtype=object), check=False)
                bits = self.input(bits, senders=senders)
                bits = await bits
                bits = [a[0] for a in bits]
                bits = sum(bits)
                return bits

            prfs = self.prfs(2)
            bits = thresha.np_pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), n)
            return bits

        if self.options.no_prss:
            uci = self._program_counter[0] % m
            senders = tuple((uci + i) % m for i in range(t+1))  # TODO: sort out load balancing
            if self.pid in senders:
                bits = field.array(np.fromiter((2*secrets.randbits(1)-1 for _ in range(n)),
                                               'O', count=n))
            else:
                bits = field.array(np.zeros(n, dtype=object), check=False)
            bits = self.input(bits, senders=senders)
            bits = await bits
            bits = [a[0] for a in bits]
            bits = np.stack(bits)
            while (_n := bits.shape[0]) > 1:
                n0 = _n%2
                _s = bits[n0:(_n+1)//2] * bits[(_n+1)//2:]
                _s = await self._reshare(_s)
                if n0:
                    _s = np.concatenate((bits[:1], _s), axis=0)
                bits = _s
            bits = bits[0].value
        else:
            prfs = self.prfs(field.order)
            r = np.array([], dtype='O')
            r2 = np.array([], dtype='O')
            h = n
            while h:
                _r = thresha.np_pseudorandom_share(field, m, self.pid, prfs, self._prss_uci(), h)
                z = thresha.np_pseudorandom_share_0(field, m, self.pid, prfs, self._prss_uci(), h)
                # Compute and open the squares and later compute square roots.
                _r2 = field.array(_r.value**2 + z.value)
                _r2 = await self.output(_r2, threshold=2*t)
                mask = _r2.value != 0
                h -= np.count_nonzero(mask)
                if h:
                    r = np.append(r, _r.value[mask])
                    r2 = np.append(r2, _r2.value[mask])
                # else: fast path for h == 0
            if len(r):
                r = np.append(r, _r.value)
                r2 = np.append(r2, _r2.value)
            else:  # fast path
                r = _r.value
                r2 = _r2.value
            bits = r * field.array._sqrt(r2, INV=True)
            if not signed:
                bits %= field.modulus

        if not signed:
            bits += 1
            bits *= (p + 1) >> 1  # divide by 2
        bits <<= f
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

    @asyncoro.mpc_coro
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
        r_divl = self._random(field, 1<<(stype.bit_length + k - l))
        if self.options.no_prss:
            r_divl = (await r_divl)[0]
        r_divl = r_divl.value
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

    @asyncoro.mpc_coro
    async def np_to_bits(self, a, l=None):
        """Secure extraction of l (or all) least significant bits of a."""  # a la [ST06].
        # TODO: other cases than characteristic=2 case, see self.to_bits()
        stype = type(a).sectype
        if l is None:
            l = stype.bit_length
        assert l <= stype.bit_length + stype.frac_length
        shape = a.shape + (l,)
        await self.returnType((type(a), True, shape))
        field = stype.field

        if issubclass(stype, self.SecureFiniteField):
            if field.characteristic == 2:
                n = a.size
                r_bits = await self.np_random_bits(field, n * l)
                r_bits = r_bits.reshape(shape)
                shifts = np.arange(l)
                r_modl = np.sum(r_bits.value << shifts, axis=a.ndim)
                a = await self.gather(a)
                c = await self.output(a + r_modl)
                c = np.vectorize(int, otypes='O')(c.value)
                c_bits = np.right_shift.outer(c, shifts) & 1
                return c_bits + r_bits

            if field.ext_deg > 1:
                raise TypeError('Binary field or prime field required.')

    @asyncoro.mpc_coro_no_pc
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

    @asyncoro.mpc_coro_no_pc
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

    @asyncoro.mpc_coro
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
        if isinstance(a, self.SecureArray):
            # TODO: vectorized version of _norm()
            v = list(map(self._norm, self.np_tolist(a.flatten())))
            return mpc.np_fromlist(v).reshape(*a.shape)

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

    @asyncoro.mpc_coro
    async def _cpx_mul(self, x, y):
        """Secure complex product of 2-tuples x and y (one resharing)."""
        # NB: ad hoc implementation for use in self.sincos() below
        shx = isinstance(x[0], self.SecureObject)
        shy = isinstance(y[0], self.SecureObject)
        stype = type(x[0]) if shx else type(y[0])
        f = stype.frac_length
        x_integral = True
        y_integral = True
        x = list(x)
        y = list(y)
        if isinstance(x[0], float):
            x[0] = round(x[0] * 2**f)
            x_integral = False
        if isinstance(x[1], float):
            x[1] = round(x[1] * 2**f)
            x_integral = False
        if isinstance(y[0], float):
            y[0] = round(y[0] * 2**f)
            y_integral = False
        if isinstance(y[1], float):
            y[1] = round(y[1] * 2**f)
            y_integral = False
        x_integral = x_integral and x[0].integral and x[1].integral
        y_integral = y_integral and y[0].integral and y[1].integral
        await self.returnType((stype, x_integral and y_integral), 2)

        if shx and shy:
            x, y = await self.gather(x, y)
        elif shx:
            x = await self.gather(x)
        else:
            y = await self.gather(y)
        a, b = x
        c, d = y
        z = [a * c - b * d, a * d + b * c]  # TODO: support complex multiplication in finite fields
        if f and (x_integral or y_integral):
            # NB: in-place rshifts
            z[0] >>= f
            z[1] >>= f
        if shx and shy:
            z = await self._reshare(z)
        if f and not (x_integral or y_integral):
            z = await self.trunc(z, f=f, l=stype.bit_length)
        return z

    @asyncoro.mpc_coro
    async def sincos(self, a):
        """Secure sine and cosine of fixed-point number a.

        See "New Approach for Sine and Cosine in Secure Fixed-Point Arithmetic"
        by Stan Korzilius and Berry Schoenmakers, to appear in the proceedings of
        CSCML 2023, 7th International Symposium on Cyber Security Cryptography and
        Machine Learning, LNCS 13914, Springer.
        """
        secfxp = type(a)
        await self.returnType(secfxp, 2)

        f = secfxp.frac_length
        k = f + 6
        secfxp2 = self.SecFxp(2*k)  # TODO: tune bit length and fractional length
        n = 2**k
        r_bits = self.random_bits(secfxp2, k)
        psi = 0
        for r_i in r_bits:
            psi <<= 1
            psi += r_i
        r12 = r_bits[1] * r_bits[2]
        s0 = 1 - 2*r_bits[0]
        c = s0 * (1 - r_bits[1] - r_bits[2] + r12 + (r_bits[2] - 2*r12)/math.sqrt(2))
        s = s0 * (r_bits[1] - r12 + r_bits[2]/math.sqrt(2))
        cs_psi = [(c, -s)]
        for i in range(3, k):
            theta_i = math.pi / 2**i
            c_i = 1 + r_bits[i] * (math.cos(theta_i) - 1)
            s_i = r_bits[i] * -math.sin(theta_i)
            cs_psi.append((c_i, s_i))
        cs_psi = mpctools.reduce(self._cpx_mul, cs_psi)
        R = self._random(secfxp2, 2**self.options.sec_param) << k

        a = self.convert(a, secfxp2)
        a = (a / (2*math.pi)) * n
        a = self.trunc(a) << k
        chi = await mpc.output(a + psi + R * n, raw=True)
        chi = chi.value >> k
        chi = (chi % n) * 2*math.pi/n
        c, s = math.cos(chi), math.sin(chi)
        c, s = self._cpx_mul(cs_psi, (c, s))
        c, s = self.convert([c, s], secfxp)
        return s, c

    def sin(self, a):
        """Secure sine of fixed-point number a."""
        return self.sincos(a)[0]

    def cos(self, a):
        """Secure cosine of fixed-point number a."""
        return self.sincos(a)[1]

    def tan(self, a):
        """Secure tangent of fixed-point number a."""
        s, c = self.sincos(a)
        return s / c

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

    @asyncoro.mpc_coro
    async def np_unit_vector(self, a, n):
        """Length-n unit vector [0]*a + [1] + [0]*(n-1-a) for secret a, assuming 0 <= a < n.

        Unit vector returned as secure NumPy array.

        NB: Value of a is reduced modulo n (almost for free).
        """
        # TODO: allow large range for a
        # TODO: extend to arrays for a
        await self.returnType((type(a).array, True, (n,)))
        # TODO: conversion GF(p) to secint, 0<=a<n, n < p/2 needed? Like for self.unit_vector(a, n).
        u = self.np_fromlist(self.random.random_unit_vector(type(a), n))
        u = await self.gather(u)
        r = u @ np.arange(n)
        f = type(a).frac_length
        r >>= f
        a = await self.gather(a)
        a >>= f
        R = self._random(type(a), 1<<self.options.sec_param)
        if self.options.no_prss:
            R = (await R)[0]
        R += 1
        c = await self.output(a - r + R * n)
        c = c.value % n
        # rotate u over c positions to the right
        u = np.roll(u, c)
        return u


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
    if options.VERSION:
        print(f'MPyC {mpyc.__version__}')
        sys.exit()

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
                        cmd_line = ' '.join(cmd_line)
                        cmd_line = f'tell application "Terminal" to do script "{cmd_line}"'
                        subprocess.Popen(['osascript', '-e', cmd_line])
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

    options.no_prss = options.no_prss or os.getenv('MPYC_NOPRSS') == '1'  # check if MPYC_NOPRSS set
    if options.no_prss:
        logging.info('Use of PRSS (pseudorandom secret sharing) disabled.')

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

"""This module provides the basic support for asynchronous communication and
computation of secret-shared values.
"""

import sys
import traceback
import struct
import itertools
import functools
import typing
from asyncio import Protocol, Future, Task
from mpyc.sectypes import SecureObject


class MessageExchanger(Protocol):
    """Send and receive messages.

    Bidirectional connection with one of the other parties (peers).
    """

    __slots__ = 'runtime', 'peer_pid', 'bytes', 'buffers', 'transport'

    def __init__(self, runtime, peer_pid=None):
        self.runtime = runtime
        self.peer_pid = peer_pid
        self.bytes = bytearray()
        self.buffers = {}
        self.transport = None

    def _key_transport_done(self):
        self.runtime.parties[self.peer_pid].protocol = self
        if all(p.protocol is not None for p in self.runtime.parties):
            self.runtime.parties[self.runtime.pid].protocol.set_result(self.runtime)

    def connection_made(self, transport):
        """Called when a connection is made.

        If the party is a client for this connection, it sends its identity
        to the peer as well as any PRSS keys.
        """
        self.transport = transport
        if self.peer_pid is not None:  # party is client (peer is server)
            m = len(self.runtime.parties)
            t = self.runtime.threshold
            pid_keys = [self.runtime.pid.to_bytes(2, 'little')]  # send pid
            if not self.runtime.options.no_prss:
                for subset in itertools.combinations(range(m), m - t):
                    if subset[0] == self.runtime.pid and self.peer_pid in subset:
                        pid_keys.append(self.runtime._prss_keys[subset])  # send PRSS keys
            transport.writelines(pid_keys)
            self._key_transport_done()

    def send(self, pc, payload):
        """Send payload labeled with pc to the peer.

        Message format consists of three parts:
         1. pc (8 bytes signed int)
         2. payload_size (4 bytes unsigned int)
         3. payload (byte string of length payload_size).
        """
        payload_size = len(payload)
        fmt = f'<qI{payload_size}s'
        self.transport.write(struct.pack(fmt, pc, payload_size, payload))

    def data_received(self, data):
        """Called when data is received from the peer.

        Received bytes are unpacked as a program counter and the payload
        (actual data). The payload is passed to the appropriate Future, if any.

        First message from peer is processed differently if peer is a client.
        """
        self.bytes.extend(data)
        if self.peer_pid is None:  # peer is client (party is server)
            if len(self.bytes) < 2:
                return

            peer_pid = int.from_bytes(self.bytes[:2], 'little')
            len_packet = 2
            if not self.runtime.options.no_prss:
                m = len(self.runtime.parties)
                t = self.runtime.threshold
                for subset in itertools.combinations(range(m), m - t):
                    if subset[0] == peer_pid and self.runtime.pid in subset:
                        len_packet += 16
                if len(self.bytes) < len_packet:
                    return

            # record new protocol peer
            self.peer_pid = peer_pid
            if not self.runtime.options.no_prss:
                # store keys received from peer
                len_packet = 2
                for subset in itertools.combinations(range(m), m - t):
                    if subset[0] == peer_pid and self.runtime.pid in subset:
                        self.runtime._prss_keys[subset] = self.bytes[len_packet:len_packet + 16]
                        len_packet += 16
            del self.bytes[:len_packet]
            self._key_transport_done()

        while self.bytes:
            if len(self.bytes) < 12:
                return

            pc, payload_size = struct.unpack_from('<qI', self.bytes)
            len_packet = payload_size + 12
            if len(self.bytes) < len_packet:
                return

            payload = struct.unpack_from(f'<{payload_size}s', self.bytes, 12)[0]
            del self.bytes[:len_packet]
            if pc in self.buffers:
                self.buffers.pop(pc).set_result(payload)
            else:
                self.buffers[pc] = payload

    def receive(self, pc):
        """Receive payload labeled with given pc from the peer."""
        payload = self.buffers.pop(pc, None)
        if payload is None:
            # Data not yet received from peer.
            payload = self.buffers[pc] = Future(loop=self.runtime._loop)
        return payload

    def connection_lost(self, exc):
        """Called when the connection with the peer is lost or closed.

        If the connection is closed normally (during shutdown) then exc is None.
        Otherwise, if the connection is lost unexpectedly, exc may indicate the
        cause (but exc is None is still possible).
        """
        if exc:
            raise exc
        # TODO: also raise an exception if exc is None and no shutdown in progress

    def close_connection(self):
        """Close connection with the peer."""
        self.transport.close()


class _AwaitableFuture:
    """Cheap replacement of a Future."""

    __slots__ = 'value'

    def __init__(self, value):
        self.value = value

    def __await__(self):
        return self.value
        yield  # NB: makes __await__ iterable


class _SharesCounter(Future):
    """Count and gather all futures (shares) recursively for a given object."""

    __slots__ = 'counter', 'obj'

    def __init__(self, loop, obj):
        super().__init__(loop=loop)
        self.counter = 0
        self._add_callbacks(obj)
        if not self.counter:
            self.set_result(_get_results(obj))
        else:
            self.obj = obj

    def _decrement(self, _):
        self.counter -= 1
        if not self.counter:
            self.set_result(_get_results(self.obj))

    def _add_callbacks(self, obj):
        if isinstance(obj, SecureObject):
            if isinstance(obj.share, Future):
                if obj.share.done():
                    obj.share = obj.share.result()
                else:
                    self.counter += 1
                    obj.share.add_done_callback(self._decrement)
        elif isinstance(obj, Future) and not obj.done():
            self.counter += 1
            obj.add_done_callback(self._decrement)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                self._add_callbacks(x)


def _get_results(obj):
    if isinstance(obj, SecureObject):
        if isinstance(obj.share, Future):
            return obj.share.result()

        return obj.share

    if isinstance(obj, Future):
        return obj.result()

    if isinstance(obj, (list, tuple)):
        return type(obj)(map(_get_results, obj))

    return obj


def gather_shares(rt, *obj):
    """Gather all results for the given futures (shares)."""
    if len(obj) == 1:
        obj = obj[0]
    if obj is None:
        return _AwaitableFuture(None)

    if isinstance(obj, Future):
        return obj

    if isinstance(obj, SecureObject):
        if isinstance(obj.share, Future):
            return obj.share

        return _AwaitableFuture(obj.share)

    if not rt.options.no_async:
        assert isinstance(obj, (list, tuple)), obj
        return _SharesCounter(rt._loop, obj)

    return _AwaitableFuture(_get_results(obj))


def _hop(a):  # NB: redefined in MPyC setup if mix of 32-bit/64-bit platforms enabled
    """Simple and efficient pseudorandom program counter hop for Python 3.6+.

    Compatible between all 64-bit platforms.
    Compatible between all 32-bit platforms.
    Not compatible between mix of 32-bit and 64-bit platforms.
    """
    return hash(frozenset(a))


class _ProgramCounterWrapper:

    __slots__ = 'runtime', 'coro', 'pc'

    def __init__(self, runtime, coro):
        self.runtime = runtime
        self.coro = coro
        runtime._program_counter[0] += 1
        self.pc = [_hop(runtime._program_counter), runtime._program_counter[1]+1]  # fork

    def __await__(self):
        while True:
            pc = self.runtime._program_counter
            self.runtime._program_counter = self.pc
            try:
                val = self.coro.send(None)
                self.pc = self.runtime._program_counter
            except StopIteration as exc:
                return exc.value  # NB: required for Python 3.7
            finally:
                self.runtime._program_counter = pc
            yield val


async def _wrap_in_coro(awaitable):
    return await awaitable


class _Awaitable:

    __slots__ = 'value'

    def __init__(self, value):
        self.value = value

    def __await__(self):
        yield self.value


def _nested_list(rt, n, dims):
    if dims:
        n0 = dims[0]
        dims = dims[1:]
        s = [_nested_list(rt, n0, dims) for _ in range(n)]
    else:
        s = [rt() for _ in range(n)]
    return s


runtime = None


def returnType(*args, wrap=True):
    """Define return type for MPyC coroutines.

    Used in first await expression in an MPyC coroutine.
    """
    rettype, *dims = args
    if isinstance(rettype, type(None)):
        rettype = None
    if rettype is not None:
        if isinstance(rettype, tuple):
            stype = rettype[0]
            integral = rettype[1]
            if stype.frac_length:
                rt = lambda: stype(None, integral)
            else:
                rt = stype
        elif issubclass(rettype, Future):
            rt = lambda: rettype(loop=runtime._loop)
        else:
            rt = rettype
        if dims:
            rettype = _nested_list(rt, dims[0], dims[1:])
        else:
            rettype = rt()
    if wrap:
        rettype = _Awaitable(rettype)
    return rettype


def _reconcile(decl, task):
    runtime._pc_level -= 1
    if decl is None:
        return

    try:
        givn = task.result()
    except Exception:
        runtime._loop.stop()  # TODO: stop loop for other exceptions in callbacks
        raise

    __reconcile(decl, givn)


def __reconcile(decl, givn):
    if isinstance(decl, SecureObject):
        if isinstance(givn, SecureObject):
            givn = givn.share
        decl.set_share(givn)
    elif isinstance(decl, list):
        for d, g in zip(decl, givn):
            __reconcile(d, g)
    else:  # isinstance(decl, Future)
        decl.set_result(givn)


def _ncopy(nested_list):
    if isinstance(nested_list, list):
        return list(map(_ncopy, nested_list))

    return nested_list


def _mpc_coro_no_pc(func):
    return mpc_coro(func, pc=False)


def mpc_coro(func, pc=True):
    """Decorator turning coroutine func into an MPyC coroutine.

    An MPyC coroutine is evaluated asynchronously, returning empty placeholders.
    The type of the placeholders is defined either by a return annotation
    of the form "-> expression" or by the first await expression in func.
    Return annotations can only be used for static types.
    """

    rettype = typing.get_type_hints(func).get('return')

    @functools.wraps(func)
    def typed_asyncoro(*args, **kwargs):
        runtime._pc_level += 1
        coro = func(*args, **kwargs)
        if rettype:
            decl = returnType(rettype, wrap=False)
        else:
            try:
                decl = coro.send(None)
            except StopIteration as exc:
                runtime._pc_level -= 1
                return exc.value

            except Exception:
                runtime._pc_level -= 1
                raise

        if runtime.options.no_async:
            while True:
                try:
                    coro.send(None)
                except StopIteration as exc:
                    runtime._pc_level -= 1
                    if decl is not None:
                        __reconcile(decl, exc.value)
                    return decl

                except Exception:
                    runtime._pc_level -= 1
                    raise

        if pc:
            coro = _wrap_in_coro(_ProgramCounterWrapper(runtime, coro))
        task = Task(coro, loop=runtime._loop)
        task.f_back = sys._getframe(1)  # enclosing MPyC coroutine call
        task.add_done_callback(lambda t: _reconcile(decl, t))
        return _ncopy(decl)

    return typed_asyncoro


def exception_handler(loop, context):
    """Handle some MPyC coroutine related exceptions."""
    if 'handle' in context:
        if 'mpc_coro' in context['message']:
            task = context['handle']._args[0]
            del context['message']  # suppress detailed message
            del context['handle']  # suppress details of handle
            loop.default_exception_handler(context)
            print('Traceback (enclosing MPyC coroutine call):')
            traceback.print_stack(task.f_back)  # TODO: extend call chain
            return

    elif 'task' in context:
        cb = context['task']._callbacks[0]
        if isinstance(cb, tuple):
            cb = cb[0]  # NB: drop context paramater for Python 3.7+
        if 'mpc_coro' in cb.__qualname__:
            if not loop.get_debug():  # Unless asyncio debug mode is enabled,
                return  # suppress 'Task was destroyed but it is pending!' message.

    loop.default_exception_handler(context)

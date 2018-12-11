"""This module provides the basic support for asynchronous communication and
computation of secret-shared values.
"""

import struct
import functools
import typing
from asyncio import Protocol, Future, Task, ensure_future
from mpyc.sectypes import Share

runtime = None

class SharesExchanger(Protocol):
    """Send and receive shares.

    Bidirectional connection with one of the other parties (peers).
    """

    def __init__(self):
        self.peer_pid = None
        self.bytes = bytearray()
        self.buffers = {}
        self.transport = None

    def connection_made(self, transport):
        """Called when a connection is made.

        The party first sends its identity to the peer.
        """
        self.transport = transport
        transport.write(str(runtime.pid).encode())

    def send_data(self, pc, payload):
        """Send payload labeled with pc to the peer.

        Message format consists of four parts:
         1. pc_size (2 bytes)
         2. payload_size (4 bytes)
         3. pc (pc_size 4-byte ints)
         4. payload (byte string of length payload_size).
        """
        pc_size, payload_size = len(pc), len(payload)
        fmt = f'!HI{pc_size}I{payload_size}s'
        t = (pc_size, payload_size) + pc + (payload,)
        self.transport.write(struct.pack(fmt, *t))

    def data_received(self, data):
        """Called when data is received from the peer.

        Received bytes are unpacked as a program counter and the payload
        (actual data). The payload is passed to the appropriate Future, if any.
        """
        self.bytes.extend(data)
        if self.peer_pid is None:
            # record new protocol peer
            self.peer_pid = int(self.bytes[:1])
            del self.bytes[:1]
            runtime.parties[self.peer_pid].protocol = self
            if all([p.protocol is not None for p in runtime.parties]):
                runtime.parties[runtime.pid].protocol.set_result(runtime)
        while self.bytes:
            if len(self.bytes) < 6:
                return

            pc_size, payload_size = struct.unpack('!HI', self.bytes[:6])
            len_packet = 6 + pc_size * 4 + payload_size
            if len(self.bytes) < len_packet:
                return

            fmt = f'!{pc_size}I{payload_size}s'
            unpacked = struct.unpack(fmt, self.bytes[6:len_packet])
            del self.bytes[:len_packet]
            pc = unpacked[:pc_size]
            payload = unpacked[-1]
            if pc in self.buffers:
                self.buffers.pop(pc).set_result(payload)
            else:
                self.buffers[pc] = payload

    def connection_lost(self, exc):
        pass

    def close_connection(self):
        """Close connection with the peer."""
        self.transport.close()

async def gather_shares(*obj):
    """Return all the results for the given futures (shared values)."""
    if len(obj) == 1:
        obj = obj[0]
    if isinstance(obj, Future):
        return await obj

    if isinstance(obj, Share):
        if isinstance(obj.df, Future):
            if runtime.options.no_async:
                obj.df = obj.df.result()
            else:
                obj.df = await obj.df
        return obj.df

    if not runtime.options.no_async:
        assert isinstance(obj, (list, tuple)), obj
        c = _count_shares(obj)
        if c:
            mux = Future()
            _register_shares(obj, [c], mux)
            await mux
    return _get_results(obj)

def _count_shares(obj):
    # Count the number of Share/Future objects in a nested structure of lists
    c = 0
    if isinstance(obj, Share):
        if isinstance(obj.df, Future):
            c = 1
    elif isinstance(obj, Future):
        c = 1
    elif isinstance(obj, (list, tuple)):
        c = sum(map(_count_shares, obj))
    return c

def _register_shares(obj, c, mux):
    if isinstance(obj, (Share, Future)):
        def got_share(_):
            c[0] -= 1
            if c[0] == 0:
                mux.set_result(None)
        if isinstance(obj, Share):
            if isinstance(obj.df, Future):
                obj.df.add_done_callback(got_share)
        else:
            obj.add_done_callback(got_share)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _register_shares(x, c, mux)

def _get_results(obj):
    if isinstance(obj, Share):
        if isinstance(obj.df, Future):
            return obj.df.result()

        return obj.df

    if isinstance(obj, Future): #expect_share
        return obj.result()

    if isinstance(obj, (list, tuple)):
        return type(obj)(map(_get_results, obj))

    return obj

class _ProgramCounterWrapper:

    __slots__ = 'f', 'saved_pc'

    def __init__(self, f):
        self.f = f
        runtime._increment_pc()
        runtime._fork_pc()
        self.saved_pc = runtime._program_counter[:]
        runtime._unfork_pc()

    def __await__(self):
        self.f = self.f.__await__()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        current_pc = runtime._program_counter[:]
        runtime._program_counter[:] = self.saved_pc
        try:
            return self.f.__next__() # NB: throws exception for async return
        finally:
            self.saved_pc = runtime._program_counter[:]
            runtime._program_counter[:] = current_pc

class _afuture(Future):
    __slots__ = 'decl'
    def __init__(self, decl):
        Future.__init__(self, loop=runtime._loop)
        self.decl = decl

def _nested_list(rt, n, dims):
    if dims:
        n0 = dims[0]
        dims = dims[1:]
        s = [_nested_list(rt, n0, dims) for _ in range(n)]
    else:
        s = [rt() for _ in range(n)]
    return s

def returnType(rettype=None, *dims):
    """Define return type for MPyC coroutines.

    Used in first await expression in an MPyC coroutine.
    """
    if rettype is not None:
        if isinstance(rettype, tuple):
            stype = rettype[0]
            integral = rettype[1]
            if stype.field.frac_length:
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
    return _afuture(rettype)

pc_level = 0
"""Tracks (length of) program counter to implement barriers."""

def _reconcile(decl, givn):
    global pc_level
    pc_level -= 1
    if decl is None:
        return

    if isinstance(givn, Task):
        givn = givn.result()
    __reconcile(decl, givn)

def __reconcile(decl, givn):
    if isinstance(decl, Future):
        decl.set_result(givn)
    elif isinstance(decl, list):
        for (d, g) in zip(decl, givn):
            __reconcile(d, g)
    elif isinstance(givn, Future):
        if runtime.options.no_async:
            decl.df.set_result(givn.result())
        else:
            givn.add_done_callback(lambda x: decl.df.set_result(x.result()))
    elif isinstance(givn, Share):
        if isinstance(givn.df, Future):
            if runtime.options.no_async:
                decl.df.set_result(givn.df.result())
            else:
                givn.df.add_done_callback(lambda x: decl.df.set_result(x.result()))
        else:
            decl.df.set_result(givn.df)
    else:
        decl.df.set_result(givn)

def _ncopy(nested_list):
    if isinstance(nested_list, list):
        return list(map(_ncopy, nested_list))

    return nested_list

def mpc_coro(f):
    """Decorator turning coroutine f into an MPyC coroutine.

    An MPyC coroutine is evaluated asychronously, returning empty placeholders.
    The type of the placeholders is defined either by a return annotation
    of the form "-> expression" or by the first await expression in f.
    Return annotations can only be used for static types.
    """

    rettype = typing.get_type_hints(f).get('return')

    @functools.wraps(f)
    def typed_asyncoro(*args, **kwargs):
        global pc_level
        coro = f(*args, **kwargs)
        if rettype:
            ret = returnType(rettype)
        else:
            ret = coro.send(None)
        ret.set_result(None)
        pc_level += 1
        if runtime.options.no_async:
            val = None
            try:
                while True:
                    val = coro.send(val)
            except StopIteration as exc:
                d = exc.value
            _reconcile(ret.decl, d)
            return ret.decl

        d = ensure_future(_ProgramCounterWrapper(coro))
        d.add_done_callback(lambda v: _reconcile(ret.decl, v))
        return _ncopy(ret.decl)

    return typed_asyncoro

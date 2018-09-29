"""This module provides the basic support for asynchronous communication and
computation of secret-shared values.
"""
import struct
import functools
import typing
import asyncio
from mpyc import sectypes

Future = asyncio.Future

class SharesExchanger(asyncio.Protocol):
    """Send and receive shares.

    Bidirectional connection with one of the other parties (peers).
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.peer_id = None
        self.bytes = bytearray()
        self.buffers = {}
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        transport.write(str(self.runtime.id).encode())

    def send_data(self, pc, data):
        """Send data labeled with pc to peer.

        Message format consists of 4 parts:
         1. pc_size (2 bytes)
         2. data_size (4 bytes)
         3. pc (pc_size 4-byte ints)
         4. data (length-data_size byte string).
        """
        pc_size, data_size = len(pc), len(data)
        fmt = f'!HI{pc_size}I{data_size}s'
        t = (pc_size, data_size) + pc + (data,)
        self.transport.write(struct.pack(fmt, *t))

    def data_received(self, bytes_received):
        """Called when data is received from the peer.

        Received bytes are unpacked as the program counter and a data part.
        The data is passed to the appropriate Future, if any.
        """
        self.bytes.extend(bytes_received)
        if self.peer_id is None:
            # record new protocol peer
            self.peer_id = int(self.bytes[:1])
            del self.bytes[:1]
            self.runtime.parties[self.peer_id].protocol = self
            if all([p.protocol != None for p in self.runtime.parties]):
                self.runtime.parties[self.runtime.id].protocol.set_result(self.runtime)
        while self.bytes:
            if len(self.bytes) < 6:
                return
            pc_size, data_size = struct.unpack('!HI', self.bytes[:6])
            len_packet = 6 + pc_size * 4 + data_size
            if len(self.bytes) < len_packet:
                return
            fmt = f'!{pc_size}I{data_size}s'
            unpacked = struct.unpack(fmt, self.bytes[6:len_packet])
            del self.bytes[:len_packet]
            pc = unpacked[:pc_size]
            data = unpacked[-1]
            if pc in self.buffers:
                self.buffers.pop(pc).set_result(data)
            else:
                self.buffers[pc] = data

    def connection_lost(self, exc):
        pass

    def close_connection(self):
        self.transport.close()

async def gather_shares(*obj):
    """Return all the results for the given futures (shared values)."""
    if len(obj) == 1:
        obj = obj[0]
    if isinstance(obj, Future):
        return await obj
    elif isinstance(obj, sectypes.Share):
        if isinstance(obj.df, Future):
            if runtime.options.no_async:
                obj.df = obj.df.result()
            else:
                obj.df = await obj.df
        return obj.df
    else:
        if not runtime.options.no_async:
            assert isinstance(obj, (list, tuple)), obj
            c = _count_shares(obj)
            if c != 0:
                mux = Future()
                _register_shares(obj, [c], mux)
                await mux
        return _get_results(obj)

def _count_shares(obj):
    # Count the number of Share/Future objects in a nested structure of lists
    if isinstance(obj, sectypes.Share):
        if isinstance(obj.df, Future):
            return 1
        else:
            return 0
    elif isinstance(obj, Future):
        return 1
    elif isinstance(obj, (list, tuple)):
        return sum(map(_count_shares, obj))
    else:
        return 0

def _register_shares(obj, c, mux):
    if isinstance(obj, (sectypes.Share, Future)):
        def got_share(_):
            c[0] -= 1
            if c[0] == 0:
                mux.set_result(None)
        if isinstance(obj, sectypes.Share):
            if isinstance(obj.df, Future):
                obj.df.add_done_callback(got_share)
        else:
            obj.add_done_callback(got_share)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _register_shares(x, c, mux)

def _get_results(obj):
    if isinstance(obj, sectypes.Share):
        if isinstance(obj.df, Future):
            return obj.df.result()
        else:
            return obj.df
    elif isinstance(obj, Future): #expect_share
        return obj.result()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map(_get_results, obj))
    else:
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
            return self.f.__next__() # throws exception for async return
        finally:
            self.saved_pc = runtime._program_counter[:]
            runtime._program_counter[:] = current_pc

class _afuture(Future):
    __slots__ = 'decl'
    def __init__(self, decl):
        Future.__init__(self)
        self.decl = decl

def returnType(rettype=None, *args):
    """Define return type for MPC coroutines.

    Used in first await expression in an MPC coroutine.
    """
    if rettype is None:
        return _afuture(None)
    if isinstance(rettype, Future):
        return _afuture(Future)
    def make(i):
        if i == len(args):
            if isinstance(rettype, tuple):
                integral = rettype[1]
                stype = rettype[0]
                if  stype.field.frac_length > 0:
                    return stype(None, integral)
                else:
                    return stype()
            return rettype()
        else:
            return [make(i + 1) for _ in range(args[i])]
    return _afuture(make(0))

def _reconcile(decl, givn):
    if isinstance(givn, asyncio.Task) and givn.done():
        givn = givn.result()
    if decl is None:
        return
    elif isinstance(decl, Future):
        decl.set_result(givn)
    elif isinstance(decl, list):
        for (d, g) in zip(decl, givn): _reconcile(d, g)
    elif isinstance(givn, Future):
        if runtime.options.no_async:
            decl.df.set_result(givn.result())
        else:
            givn.add_done_callback(lambda x: decl.df.set_result(x.result()))
    elif isinstance(givn, sectypes.Share):
        if isinstance(givn.df, Future):
            if runtime.options.no_async:
                decl.df.set_result(givn.df.result())
            else:
                givn.df.add_done_callback(lambda x: decl.df.set_result(x.result()))
        else:
            decl.df.set_result(givn.df)
    else:
        decl.df.set_result(givn)

def _ncopy(nested_lists):
    if isinstance(nested_lists, list):
        return list(map(_ncopy, nested_lists))
    else:
        return nested_lists

pc_level = 0
"""Tracks (length of) program counter to implement barriers."""

def mpc_coro(f):
    """Decorator turning coroutine f into an MPC coroutine.

    An MPC coroutine is evaluated asychronously, returning empty placeholders.
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
            except StopIteration as e:
                d = e.value
            pc_level -= 1
            _reconcile(ret.decl, d)
            return ret.decl
        else:
            d = asyncio.ensure_future(_ProgramCounterWrapper(coro))
            def dec_pc_level():
                global pc_level
                pc_level -= 1
            d.add_done_callback(lambda _: dec_pc_level())
            d.add_done_callback(lambda v: _reconcile(ret.decl, v))
            return _ncopy(ret.decl)

    return typed_asyncoro

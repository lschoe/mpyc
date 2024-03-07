import unittest
from unittest.mock import Mock
from asyncio import Future, Transport
import mpyc.asyncoro
from mpyc.runtime import Party, Runtime, mpc


class Arithmetic(unittest.TestCase):

    def test_message_exchanger(self):
        # two parties, each with its own MessageExchanger()
        rt0 = Runtime(0, [Party(0), Party(1)], mpc.options)  # NB: with its own Party() instances
        rt1 = Runtime(1, [Party(0), Party(1)], mpc.options)  # NB: with its own Party() instances
        mx0 = mpyc.asyncoro.MessageExchanger(rt0, 1)  # client
        mx1 = mpyc.asyncoro.MessageExchanger(rt1)     # server

        # test: client connects with server
        rt0.parties[0].protocol = Future()
        rt0.parties[1].protocol = None
        rt1.parties[0].protocol = None
        rt1.parties[1].protocol = Future()
        transport0 = Mock(Transport)
        transport1 = Mock(Transport)

        def _writelines(s):
            transport0.s = b''.join(s)
        transport0.writelines = _writelines

        def _write(s):
            transport1.s = s
        transport1.write = _write

        mx0.connection_made(transport0)
        mx1.connection_made(transport1)
        data = transport0.s
        mx1.data_received(data[:1])
        mx1.data_received(data[1:5])
        mx1.data_received(data[5:])

        # test: message from server received after client expects it
        pc0 = rt0._program_counter[0]
        pc1 = rt1._program_counter[0]
        self.assertEqual(pc0, pc1)
        payload = b'123'
        mx1.send(pc1, payload)
        fut = mx0.receive(pc0)
        data = transport1.s
        mx0.data_received(data[:12])
        mx0.data_received(data[12:])
        self.assertEqual(fut.result(), payload)

        # message from server received before client expects it
        pc0 += 1
        pc1 += 1
        payload = b'456'
        mx1.send(pc1, payload)
        data = transport1.s
        mx0.data_received(data[:12])
        mx0.data_received(data[12:])
        msg = mx0.receive(pc0)
        self.assertEqual(msg, payload)

        # close connections
        rt0.parties[0].protocol = Future()
        rt1.parties[1].protocol = Future()
        mx0.close_connection()
        self.assertRaises(Exception, mx0.connection_lost, Exception())
        mx1.close_connection()
        mx0.connection_lost(None)
        mx1.connection_lost(None)

    def test_gather_futures(self):
        self.assertEqual(mpc.run(mpyc.asyncoro.gather_shares(mpc, None)), None)
        mpc.options.no_async = False
        fut = Future()
        gut = mpyc.asyncoro.gather_shares(mpc, [fut, fut])
        fut.set_result(42)
        self.assertEqual(mpc.run(gut), [42, 42])
        mpc.options.no_async = True


if __name__ == "__main__":
    unittest.main()

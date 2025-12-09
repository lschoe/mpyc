# Test of combined mul-and-trunc
# Test using: python multrunc.py -M 3

from mpyc.runtime import mpc

async def main():
    await mpc.start()
    val1 = mpc.input(mpc.SecFxp(10, 4)(-2.5), senders=[0])
    val2 = mpc.input(mpc.SecFxp(10, 4)(1.9), senders=[0])
    prod = mpc.mul(val1[0], val2[0])
    print("Result", await mpc.output(prod))
    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

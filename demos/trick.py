from mpyc.runtime import mpc

mpc.start()

secint = mpc.SecInt()

a = [secint(10)]

a[0] = a[0]+ a[0]

print("na a[0]")

print(mpc.run(mpc.output(a)))

mpc.shutdown()

import os
import sys

n = int(sys.argv[1])
a1 = sys.argv[2]
a2s = ' '.join(sys.argv[3:])
if n > 0:
    for i in range(n - 1, 0, -1):
        p = 'party' + str(n) + '_' + str(i)
        os.system('start python ' + a1 + ' -c ' + p + '.ini ' + a2s)
    os.system('python ' + a1 + ' -c party' + str(n) + '_0.ini ' + a2s)
else:
    n = -n
    for i in range(n - 1, -1, -1):
        p = 'party' + str(n) + '_' + str(i)
        os.system('start /b python ' + a1 + ' -c ' + p + '.ini ' + a2s + ' > ' + p + '.log')

# Windows version
import os, sys

n = int(sys.argv[1])
a1 = sys.argv[2]
a2s = ' '.join(sys.argv[3:])
for i in range(n, 1, -1):
    i -= 1
    os.system('start python ' + a1 + ' -c party' + str(n) + '_' + str(i) + '.ini ' + a2s)
os.system('python ' + a1 + ' -c party' + str(n) + '_0.ini ' + a2s)

Usage examples (Windows command prompt).

First argument (absolute value) sets number of parties (output to .log files if negative).

```
python secretsanta.py

run 1 secretsanta.py

run 5 secretsanta.py

run -1 secretsanta.py

run -2 secretsanta.py

run 1 id3gini.py -h

run 1 id3gini.py -H

run-all 1

run-all 2

run-all 2 --ssl

run-all 3

run-all 3 -t0 --ssl

run-all 4

run-all 5

python cnnmnist.py

python cnnmnist.py 1 0

python cnnmnist.py 3

run 1 cnnmnist.py 1.5 0

python bnnmnist.py --HELP

python -m cProfile -s time bnnmnist.py | more

python -m cProfile -s cumtime sort.py 64 | more

python -m cProfile -s time aes.py | more
```

# How to run the demos

## Use help messages

Use `-H`, `--HELP` option with any demo to see the MPyC help message.

`python secretsanta.py -H`

```
usage: secretsanta.py [-H] [-h] [-C ini] [-P addr] [-M m] [-I i] [-T t]
                      [-B b] [--ssl] [-L l] [-K k] [--log-level ll]
                      [--no-log] [--no-async] [--no-barrier] [--no-gmpy2]
                      [--no-numpy] [--no-prss] [--mix32-64bit]
                      [--output-windows] [--output-file] [-f F]

MPyC help:
  -H, --HELP            show this help message for MPyC and exit
  -h, --help            show help message for this MPyC program (if any)

MPyC configuration:
  -C ini, --config ini  use ini file, defining all m parties
  -P addr               use addr=host:port per party (repeat m times)
  -M m                  use m local parties (and run all m, if i is not set)
  -I i, --index i       set index of this local party to i, 0<=i<m
  -T t, --threshold t   threshold t, 0<=t<m/2
  -B b, --base-port b   use port number b+i for party i
  --ssl                 enable SSL connections

MPyC parameters:
  -L l, --bit-length l  default bit length l for secure numbers
  -K k, --sec-param k   security parameter k, leakage probability 2**-k
  --log-level ll        logging level ll=debug/info(default)/warning/error
  --no-log              disable logging messages
  --no-async            disable asynchronous evaluation
  --no-barrier          disable barriers
  --no-gmpy2            disable use of gmpy2 package
  --no-numpy            disable use of numpy package
  --no-prss             disable use of PRSS (pseudorandom secret sharing)
  --mix32-64bit         enable mix of 32-bit and 64-bit platforms

MPyC misc:
  --output-windows      screen output for parties 0<i<m (one window each)
  --output-file         append output of parties 0<i<m to party{m}_{i}.log
  -f F                  consume IPython's -f argument F
```

Use `-h`, `--help` option to see help message for demo (if available).

`python onewayhashchains.py -h`

```
Showing help message for onewayhashchains.py, if available:

usage: onewayhashchains.py [-h] [-k K] [--recursive] [--no-one-way]
                           [--no-random-seed]

optional arguments:
  -h, --help        show this help message and exit
  -k K, --order K   order K of hash chain, length n=2**K
  --recursive       use recursive pebbler
  --no-one-way      use dummy one-way function
  --no-random-seed  use fixed seed
```

## Examples

`python secretsanta.py`

`python secretsanta.py -M1`

`python secretsanta.py -M 5`

`python -OO secretsanta.py -M2 --output-file`

`python lpsolver.py -h`

`python id3gini.py -H`

`run-all.bat --no-async`

`run-all.sh -M2 --ssl`

`run-all.bat -M3 --output-file > party3_0.log`

`run-all.sh -M4 --threshold 0 --ssl`

`run-all.bat -M5 --output-windows`

`run-all.sh -M7`

`python cnnmnist.py > party1_0.log`

`python cnnmnist.py 1 0`

`python np_cnnmnist.py 3.5`

`python cnnmnist.py -M1 1.5 0`

`python bnnminist.py -M1 --no-barrier`

`python ridgeregression.py -i3 -a7`

`python ridgeregression.py -M3`

`python multilateration.py -i 3 4 5`

`python multilateration.py -M5 -i1 -a1 --plot`

`python kmsurvival.py --plot --no-log`

`python kmsurvival.py -M3 -i4 --collapse --plot`

`python -m cProfile -s time bnnmnist.py | more`

`python -m cProfile -s cumtime sort.py 64 | more`

`python -m cProfile -s time aes.py | more`

## Jupyter notebooks

[SecretSantaExplained](SecretSantaExplained.ipynb) provides a quick intro to MPyC.

[SecureSortingNetsExplained](SecureSortingNetsExplained.ipynb) shows how to convert some existing Python programs to MPyC programs.

[KaplanMeierSurvivalExplained](KaplanMeierSurvivalExplained.ipynb) presents privacy-preserving Kaplan-Meier survival analysis, featuring aggregate Kaplan-Meier curves and secure logrank tests.

[4demos](4demos.ipynb) gives quick access to demos secretsanta.py, id3gini.py, lpsolver.py, np_cnnmnist.py.

[OneWayHashChainsExplained](OneWayHashChainsExplained.ipynb) shows a more advanced MPyC program.

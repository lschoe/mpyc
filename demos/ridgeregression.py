"""Demo Linear and Ridge Regression.

MPyC demo accompanying the paper 'Efficient Secure Ridge Regression from
Randomized Gaussian Elimination' by Frank Blom, Niek J. Bouman, Berry
Schoenmakers, and Niels de Vreede, presented at TPMPC 2019 by Frank Blom.
See https://eprint.iacr.org/2019/773 (or https://ia.cr/2019/773). Published
in the proceedings of CSCML 2021, 5th International Symposium on Cyber Security
Cryptography and Machine Learning, LNCS 12716, pp. 301-316, Springer, see
https://doi.org/10.1007/978-3-030-78086-9_23.

The following datasets from the UCI Machine Learning Repository are used:

1. 'student-mat'         (student.zip)
2. 'winequality-red'     (winequality-red.csv)
3. 'winequality-white'   (winequality-red.csv)
4. 'YearPredictionMSD'   (YearPredictionMSD.txt.zip)
5. 'ethylene_methane'    (data.zip)
6. 'ethylene_CO'         (data.zip)
7. 'HIGGS'               (HIGGS.csv.gz)

The first three datasets are included in this demo (see directory ./data/regr/).
The other ones can be downloaded from https://archive.ics.uci.edu/ml/datasets/
(use the -u --data-url command line option to get the full URL for each dataset).
Simply put the files indicated above in directory ./data/regr/, no need to (g)unzip!

By default, the demo runs with synthetic data (with n=1000 samples, d=10 features,
and e=1 target). The default accuracy varies from 4 to 7 fractional bits. Setting
the regularization parameter lambda to 0 will revert to linear regression.

As explained in the paper, the lambda-regularized model W = A^-1 B is computed where
A = X^T X + lambda I and B = X^T Y for a given n by d matrix X and n by e matrix Y.
At the end of the secure computation, the d by e matrix W is output in the clear,
in either of the following two ways:

1. (default) Both (adj A)B and det A are output in the clear as integer values, and
   model W = (adj A)B / (det A) is computed using ordinary floating-point division.

2. (option --ratrec) Model W = (adj A)B / (det A) is computed securely modulo a
   sufficiently large prime p'. Then W is output in the clear, and rational reconstruction
   modulo p' is used to recover each entry of W as a numerator-denominator pair, which
   are then divided onto each other using ordinary floating-point division.

Use the -h --help command line option for more help.

The code below is based on Frank Blom's original implementation used for the paper.
"""

import os
import time
import argparse
import logging
import random
import io
import gzip
import zipfile
import csv
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
from mpyc.gmpy import ratrec
from mpyc.runtime import mpc


async def synthesize_data(n_samples, n_features, n_targets):
    rnd = await mpc.transfer(random.randrange(2**31), senders=0)
    X, Y = sklearn.datasets.make_regression(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=max(1, n_features - 5),
                                            n_targets=n_targets, bias=42,
                                            effective_rank=max(1, n_features - 3),
                                            tail_strength=0.5, noise=1.2,
                                            random_state=rnd)  # all parties use same rnd
    if n_targets == 1:
        Y = np.transpose([Y])
    X = np.concatenate((X, Y), axis=1)
    b_m = np.min(X, axis=0)
    b_M = np.max(X, axis=0)
    coef_add = [-(m + M) / 2 for m, M in zip(b_m, b_M)]
    coef_mul = [2 / (M - m) for m, M in zip(b_m, b_M)]
    for xi in X:
        for j in range(len(xi)):
            # map to [-1,1] range
            xi[j] = (xi[j] + coef_add[j]) * coef_mul[j]
    return X


def read_data(infofile):
    with open(infofile, newline='') as file:
        reader = csv.reader(file)

        # process first line
        datafile, delim, skip_header, split, n, d_, e = next(reader)
        skip_header = int(skip_header)  # number of lines to skip at start of datafile
        split = int(split)              # train-test split (0 for random split)
        n = int(n)                      # number of samples
        d_ = int(d_)                    # number of features in datafile
        e = int(e)                      # number of targets

        # process remaining lines
        d = 0
        L = d_ + e  # total number of columns in datafile
        categories = [None] * L
        coef_add = [None] * L
        coef_mul = [None] * L
        for j in range(L):
            line = next(reader)
            feature_type = line[0]
            if feature_type == 'numerical':
                m, M = float(line[1]), float(line[2])
                coef_add[j] = -(m + M) / 2
                coef_mul[j] = 2 / (M - m)
                d += 1
            elif feature_type == 'categorical':
                while not line[-1]:  # drop trailing empty columns
                    line.pop()
                categories[j] = line[1:]
                d += len(categories[j])  # one hot encoding
            elif feature_type == 'exclude':
                categories[j] = []
            else:
                raise ValueError('unknown feature type')
        d -= e  # number of features

    datafile = os.path.join('data', 'regr', datafile)
    if datafile.endswith('.gz'):
        open_file = lambda f: gzip.open(f, mode='rt', newline='')
    elif datafile.find('.zip!') >= 0:
        archive, datafile = datafile.split('!')
        open_file = lambda f: io.TextIOWrapper(zipfile.ZipFile(archive).open(f), newline='')
    else:
        open_file = lambda f: open(f, newline='')

    offset = 0
    if datafile.find('Year') >= 0 or datafile.find('HIGGS') >= 0:
        offset = 1 - L  # hack: rotate left for YearPrediction and HIGGS
    elif datafile.find('ethylene') >= 0:
        offset = 3 - L  # hack: rotate left by 3 for ethylene
        csv.register_dialect('ethylene', delimiter=' ', skipinitialspace=True)

    X = np.empty((n, d + e), dtype=float)
    float1 = float(1)
    float_1 = float(-1)
    with open_file(datafile) as file:
        reader = csv.reader(file, delimiter=delim)
        if datafile.find('ethylene') >= 0:
            reader = csv.reader(file, dialect='ethylene')
        for _ in range(skip_header):
            next(reader)
        n100 = n // 100
        for i, row in enumerate(reader):
            if not i % n100:
                print(f'Loading ... {round(100*i/n)}%', end='\r')
            if len(row) > L:
                row = row[:L]  # ignore spurious columns
            x = X[i]
            l = 0  # column index for row x
            for j in range(L):
                if categories[j] is None:  # numerical feature
                    # map to [-1,1] range
                    x[l] = (float(row[offset + j]) + coef_add[j]) * coef_mul[j]
                    l += 1
                elif categories[j]:  # categorical feature
                    # one hot encoding of row[j]
                    for item in categories[j]:
                        x[l] = float1 if item == row[j] else float_1
                        l += 1
    return X, d, e, split


def bareiss(Zp, A):
    """Bareiss-like integer-preserving Gaussian elimination adapted for Zp.

    Using exactly one modular inverse in Zp per row of A.
    """
    p = Zp.modulus
    d, d_e = A.shape  # d by d+e matrix A

    # convert A elementwise from Zp to int
    for i in range(d):
        for j in range(d_e):
            A[i, j] = A[i, j].value

    # division-free Gaussian elimination
    for k in range(d):
        for i in range(k+1, d):
            for j in range(k+1, d_e):
                A[i, j] = (A[k, k] * A[i, j] - A[k, j] * A[i, k]) % p

    # back substitution
    for i in range(d-1, -1, -1):
        inv = (1 / Zp(A[i, i])).value
        if i < d-2:
            A[i, i] = inv  # keep reciprocal for determinant
        for j in range(d, d_e):
            s = A[i, j]
            for k in range(i+1, d):
                s -= A[i, k] * A[k, j]
            s %= p
            A[i, j] = (s * inv) % p

    # postponed division for determinant
    inv = 1
    det = A[d-1, d-1]
    for i in range(d-2):
        inv = (inv * A[i, i]) % p
        det = (det * inv) % p

    return A[:, d:], det


def random_matrix_determinant(secnum, d):
    d_2 = d * (d-1) // 2
    L = np.diagflat([secnum(1)] * d)
    L[np.tril_indices(d, -1)] = mpc._randoms(secnum, d_2)
    L[np.triu_indices(d, 1)] = [secnum(0)] * d_2
    diag = mpc._randoms(secnum, d)
    U = np.diagflat(diag)
    U[np.tril_indices(d, -1)] = [secnum(0)] * d_2
    U[np.triu_indices(d, 1)] = mpc._randoms(secnum, d_2)
    R = mpc.matrix_prod(L.tolist(), U.tolist())
    detR = mpc.prod(diag)  # detR != 0 with overwhelming probability
    return R, detR


@mpc.coroutine
async def linear_solve(A, B):
    secnum = type(A[0][0])
    d, e = len(A), len(B[0])
    await mpc.returnType(secnum, d * e + 1)

    R, detR = random_matrix_determinant(secnum, d)
    RA = mpc.matrix_prod(R, A)
    RA = await mpc.output([a for row in RA for a in row], raw=True)
    RA = np.reshape(RA, (d, d))
    RB = mpc.matrix_prod(R, B)
    RB = await mpc.gather(RB)  # NB: RB is secret-shared

    invA_B, detRA = bareiss(secnum.field, np.concatenate((RA, RB), axis=1))
    detA = detRA / detR
    adjA_B = [secnum(a) * detA for row in invA_B for a in row]
    return adjA_B + [detA]


def rmse(Y, P):
    return np.sqrt(sklearn.metrics.mean_squared_error(Y, P, multioutput='raw_values'))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset', type=int, metavar='I',
                        help=('dataset 0=synthetic (default), 1=student, 2=wine-red, '
                              '3=wine-white, 4=year, 5=gas-methane, 6=gas-CO, 7=higgs'))
    parser.add_argument('-u', '--data-url', action='store_true', default=False,
                        help='show URL for downloading dataset I')
    parser.add_argument('-l', '--lambda_', type=float, metavar='L',
                        help='regularization L>=0.0 (default=1.0)')
    parser.add_argument('-a', '--accuracy', type=int, metavar='A',
                        help='accuracy A (number of fractional bits)')
    parser.add_argument('-n', '--samples', type=int, metavar='N',
                        help='number of samples in synthetic data (default=1000)')
    parser.add_argument('-d', '--features', type=int, metavar='D',
                        help='number of features in synthetic data (default=10)')
    parser.add_argument('-e', '--targets', type=int, metavar='E',
                        help='number of targets in synthetic data (default=1)')
    parser.add_argument('--ratrec', action='store_true',
                        default=False, help='use rational reconstruction to hide determinant')
    parser.set_defaults(dataset=0, lambda_=1.0, accuracy=-1,
                        samples=1000, features=10, targets=1)
    args = parser.parse_args()

    await mpc.start()

    if not args.dataset:
        range_alpha = range(4, 8)
        n, d, e, split = args.samples, args.features, args.targets, 0
        name = 'SYNTHETIC'
        logging.info('Generating synthetic data')
        X = await synthesize_data(n, d, e)
    else:
        settings = [('student+performance', 'student-mat', 6),
                    ('Wine+Quality', 'winequality-red', 7),
                    ('Wine+Quality', 'winequality-white', 8),
                    ('Yearpredictionmsd', 'YearPredictionMSD', 6),
                    ('Gas+sensor+array+under+dynamic+gas+mixtures', 'ethylene_methane', 8),
                    ('Gas+sensor+array+under+dynamic+gas+mixtures', 'ethylene_CO', 9),
                    ('HIGGS', 'HIGGS', 5)]
        url, name, alpha = settings[args.dataset - 1]
        url = 'https://archive.ics.uci.edu/ml/datasets/' + url
        if args.data_url:
            print(f'URL: {url}')
        range_alpha = range(alpha, alpha + 1)
        infofile = os.path.join('data', 'regr', 'info-' + name + '.csv')
        logging.info(f'Loading dataset {name}')
        X, d, e, split = read_data(infofile)
        n = len(X)
        logging.info(f'Loaded {n} samples')
    print(f'dataset: {name} with {n} samples, {d} features, and {e} target(s)')
    print(f'regularization lambda: {args.lambda_}')

    # split in train set and test set
    if split:
        # fixed split
        X1, X2 = X[:split], X[split:]
    else:
        # random split (all parties use same rnd)
        rnd = await mpc.transfer(random.randrange(2**31), senders=0)
        X1, X2 = sklearn.model_selection.train_test_split(X, train_size=0.7, random_state=rnd)
    del X
    X1, Y1 = X1[:, :d], X1[:, d:]
    X2, Y2 = X2[:, :d], X2[:, d:]
    n1 = len(X1)
    d = d + 1  # add (virtual) feature column X_d = [1, ..., 1] for vertical intercept

    # ridge regression "in the clear"
    ridge = sklearn.linear_model.Ridge(alpha=args.lambda_,
                                       fit_intercept=True,
                                       copy_X=True,
                                       solver='cholesky')
    ridge.fit(X1, Y1)
    error_train_skit = rmse(Y1, ridge.predict(X1))
    error_test_skit = rmse(Y2, ridge.predict(X2))
    print(f'scikit train error: {error_train_skit}')
    print(f'scikit test error:  {error_test_skit}')

    if args.accuracy >= 0:
        alpha = args.accuracy
        range_alpha = range(alpha, alpha + 1)
    for alpha in range_alpha:  # accuracy parameter
        print('accuracy alpha:', alpha)
        # set parameters accordingly
        beta = 2**alpha
        lambda_ = round(args.lambda_ * beta**2)
        gamma = n1 * beta**2 + lambda_
        secint = mpc.SecInt(gamma.bit_length() + 1)
        print(f'secint prime q: {secint.field.modulus.bit_length()} bits'
              f' (secint bit length: {secint.bit_length})')
        bound = round(d**(d/2)) * gamma**d
        if not args.ratrec:
            secnum = mpc.SecFld(min_order=2*bound + 1, signed=True)
            print(f'secfld prime p: {secnum.field.modulus.bit_length()} bits')
        else:
            secnum = mpc.SecInt(l=bound.bit_length() + 1)
            print(f'secint prime p: {secnum.field.modulus.bit_length()} bits'
                  f' (secint bit length: {secnum.bit_length})')
            secfld = mpc.SecFld(min_order=4*bound**2)
            print(f"secfld prime p': {secfld.field.modulus.bit_length()} bits")

        f2 = float(beta)
        q = secint.field.modulus
        logging.info('Transpose, scale, and create (degree 0) shares for X and Y')
        # enforce full size shares (mod q numbers) by adding q to each element
        Xt = [[int(a * f2) + q for a in col] for col in X1.transpose()]
        Yt = [[int(a * f2) + q for a in col] for col in Y1.transpose()]

        timeStart = time.process_time()
        logging.info('Compute A = X^T X + lambda I and B = X^T Y')

        AB = []
        for i in range(d-1):
            xi = Xt[i]
            for j in range(i, d-1):
                xj = Xt[j]
                s = 0
                for k in range(n1):
                    s += xi[k] * xj[k]
                AB.append(s)              # X_i dot X_j
            AB.append(sum(xi) * beta)     # X_i dot X_d
            for j in range(e):
                yj = Yt[j]
                s = 0
                for k in range(n1):
                    s += xi[k] * yj[k]
                AB.append(s)              # X_i dot Y_j
        AB.append(n1 * beta**2)           # X_d dot X_d
        for j in range(e):
            AB.append(beta * sum(Yt[j]))  # X_d dot Y_j

        del Xt, Yt
        AB = [secint.field(a) for a in AB]
        AB = await mpc._reshare(AB)

        timeMiddle = time.process_time()
        logging.info('Compute w = A^-1 B')

        # convert secint to secnum
        AB = [secint(a) for a in AB]
        AB = mpc.convert(AB, secnum)

        # extract A and B from the AB array
        A = [[None] * d for _ in range(d)]
        B = [[None] * e for _ in range(d)]
        index = 0
        for i in range(d):
            A[i][i] = AB[index] + lambda_
            index += 1
            for j in range(i+1, d):
                A[i][j] = A[j][i] = AB[index]
                index += 1
            for j in range(e):
                B[i][j] = AB[index]
                index += 1

        # solve A w = B
        w_det = linear_solve(A, B)
        if not args.ratrec:
            w_det = await mpc.output(w_det)
            *w, det = list(map(int, w_det))
            w = np.reshape(w, (d, e))
            w /= det
        else:
            *w, det = mpc.convert(w_det, secfld)
            w = mpc.scalar_mul(1/det, w)
            w = await mpc.output(w)
            w = [ratrec(int(a), secfld.field.modulus) for a in w]
            w = [a / b for a, b in w]
            w = np.reshape(w, (d, e))

        timeEnd = time.process_time()
        logging.info(f'Total time {timeEnd - timeStart} = '
                     f'A and B in {timeMiddle - timeStart} + '
                     f'A^-1 B in {timeEnd - timeMiddle} seconds')

        error_train_mpyc = rmse(Y1, np.dot(X1, w[:-1]) + w[-1])
        error_test_mpyc = rmse(Y2, np.dot(X2, w[:-1]) + w[-1])
        print(f'MPyC train error: {error_train_mpyc}')
        print(f'MPyC test error:  {error_test_mpyc}')
        print(f'relative train error: {(error_train_mpyc - error_train_skit) / error_train_skit}')
        print(f'relative test error:  {(error_test_mpyc - error_test_skit) / error_test_skit}')

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

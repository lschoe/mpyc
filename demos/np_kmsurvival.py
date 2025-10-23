"""Demo Kaplan-Meier surivival analysis.

This demo is a fully equivalent reimplementation of the kmsurvival.py demo,
using secure fixed-point arrays for NumPy-based vectorized computation.

The secure fixed-point divisions (reciprocals), which form the bottleneck for
the logrank test, are now combined in a vectorized manner operating on a single
secure fixed-point array. This gives a speedup by a factor of 6 to 9 roughly,
assuming enough memory. With limited memory, the throttled implementation of
kmsurvival.py may be faster. See function logrank_test() below.

See kmsurvival.py for more information.
"""
import os
import logging
import argparse
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import lifelines.datasets
import lifelines.statistics
import lifelines.plotting
from lifelines import KaplanMeierFitter
from mpyc.runtime import mpc


def plot_fits(kmf1, kmf2, title, unit_of_time):
    ax = kmf1.plot(show_censors=True)
    ax = kmf2.plot(ax=ax, show_censors=True)
    ax.set_title(title)
    if unit_of_time:
        plt.xlabel(f'timeline ({unit_of_time})')
    lifelines.plotting.add_at_risk_counts(kmf1, kmf2, ax=ax, labels=None)
    plt.tight_layout()
    figname = ax.figure.canvas.manager.get_window_title()
    ax.figure.canvas.manager.set_window_title(f'Party {mpc.pid} - {figname}')


def events_to_table(maxT, T, E):
    """Create survival table, one entry for time j=1, ..., j=maxT."""
    TE = pd.concat([T, E], axis=1).groupby(T.name)  # group events by time
    df = TE.sum()    # number of observed events at time j
    nf = TE.count()  # total number of events at time j
    j = df.index.astype(int)  # unique times

    d = np.zeros(maxT, dtype=int)
    n = np.zeros(maxT, dtype=int)
    d[j - 1] += df[E.name]   # observed events at time j
    n[j - 1] += nf[E.name]   # events at time j (incl. censored events)
    n = np.flip(np.cumsum(np.flip(n)))
    return d, n


def events_from_table(d, n):
    maxT = len(d)
    T = np.empty(n[0], dtype=int)
    E = np.empty(n[0], dtype=bool)
    for j in range(maxT):
        h = n[j+1] if j+1 < maxT else 0
        T[n[0] - n[j]:n[0] - h] = j+1                      # events at time j+1
        E[n[0] - n[j]:n[0] - n[j] + d[j]] = True  # observed events at time j+1
        E[n[0] - n[j] + d[j]:n[0] - h] = False    # censored events at time j+1
    return T, E


async def logrank_test(secfxp, d1, d2, n1, n2):
    d = d1 + d2
    n = n1 + n2
    b = n * (n - 1)
    c = d * n1 / (n * b)  # NB: using only one fixed-point division /
    detot = np.sum(d1 - b * c)
    vtot = np.sum(n2 * (n - d) * c)
    chi = await mpc.output(detot**2 / vtot)
    p = scipy.stats.chi2.sf(chi, 1)
    return lifelines.statistics.StatisticalResult(p_value=p, test_statistic=chi)


def aggregate(d, n, stride):
    if s := len(d) % stride:
        d = np.concatenate((d, type(d)(np.array([0] * (stride - s)))))
    d = d.reshape(-1, stride)
    agg_d = np.sum(d, axis=1)
    agg_n = n[::stride]
    return agg_d, agg_n


def agg_logrank_test(secfxp, d1, d2, n1, n2, agg_d1, agg_d2, stride):
    candidates = secfxp.array(np.array([], dtype=int)).reshape(4, 0)
    maxT = len(d1)
    for start in range(0, maxT, stride):
        group = start // stride
        n_observed_events = agg_d1[group] + agg_d2[group]
        msn = min(stride, n_observed_events)  # upper bound
        stop = min(start + stride, maxT)
        logging.info(f'Interval {group + 1} (time {start + 1} to {stop})'
                     f' # observed events = {n_observed_events}')
        if msn == 0:
            continue

        z = np.zeros(msn, dtype=int)
        ix = secfxp.array(z)
        oblivious_table = secfxp.array(np.vstack((z, z, z + 1, z + 1)))
        for j in range(start, stop):
            is_active = d1[j] + d2[j] != 0
            rot_ix = mpc.np_update(np.roll(ix, 1), 0, 1 - np.sum(ix))
            ix = np.where(is_active, rot_ix, ix)
            select = is_active * ix
            new = mpc.np_fromlist([d1[j], d2[j], n1[j], n2[j]]).reshape(4, 1)
            oblivious_table = np.where(select, new, oblivious_table)
        candidates = np.hstack((candidates, oblivious_table))
    return logrank_test(secfxp, *candidates)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset', type=int, metavar='I',
                        help=('dataset 0=btrial(default) 1=waltons 2=aml 3=lung 4=dd'
                              ' 5=stanford_heart_transplants 6=kidney_transplant'))
    parser.add_argument('-s', '--stride', type=int, metavar='S',
                        help='interval length for aggregated events')
    parser.add_argument('-a', '--accuracy', type=int, metavar='A',
                        help='number of fractional bits')
    parser.add_argument('--collapse', action='store_true',
                        help='days->weeks->month->years')
    parser.add_argument('--print-tables', action='store_true',
                        help='print survival tables')
    parser.add_argument('--plot-curves', action='store_true',
                        help='plot survival curves')
    parser.set_defaults(dataset=0)
    args = parser.parse_args()

    settings = [('btrial.csv', 12, 28, 'months', 'time', 'death', 'im',
                 ('-ve immunohistochemical response', '+ve immunohistochemical response'), (1, 2)),
                ('waltons', 10, 32, 'days', 'T', 'E', 'group',
                 ('miR-137', 'control'), ('miR-137', 'control')),
                ('aml.csv', 16, 32, 'weeks', 'time', 'cens', 'group',
                 ('Maintained', 'Not maintained'), (1, 2)),
                ('lung', 73, 32, 'days', 'time', 'status', 'sex',
                 ('Male', 'Female'), (1, 2)),
                ('dd', 3, 48, 'years', 'duration', 'observed', 'democracy',
                 ('Democracy', 'Non-democracy'), ('Democracy', 'Non-democracy')),
                ('stanford_heart_transplants', 90, 32, 'days', 'time', 'event', 'transplant',
                 ('no transplant', 'transplant'), (0, 1)),
                ('kidney_transplant', 180, 40, 'days', 'time', 'death', 'sex',
                 ('male', 'female'), (1, 0))]
    (name, stride, accuracy, unit_of_time, times, events, groups,
     (label1, label2), (value1, value2)) = settings[args.dataset]
    if name.endswith('.csv'):
        df = pd.read_csv(os.path.join('data', 'surv', name))
        name = name[:-4]
    else:
        df = eval('lifelines.datasets.load_' + name)()
    if name == 'stanford_heart_transplants':
        df = df[(df['transplant'] == 1) | ~df['id'].isin(set(df[df['transplant'] == 1]['id']))]
        df['time'] = round(df['stop'] - df['start'] + 0.5)
    elif name == 'kidney_transplant':
        df['sex'] = df['black_male'] + df['white_male']
    if args.stride:
        stride = args.stride
    if args.collapse:
        if unit_of_time == 'days':
            unit_of_time = 'weeks'
            df[times] = (df[times]+6) // 7  # convert days to weeks
            stride //= 7
        elif unit_of_time == 'weeks':
            unit_of_time = 'months'
            df[times] = (df[times]+3) // 4  # convert weeks to months
            stride //= 4
        elif unit_of_time == 'months':
            unit_of_time = 'years'
            df[times] = (df[times]+11) // 12  # convert months to years
            stride //= 12
    if args.accuracy:
        accuracy = args.accuracy
    secfxp = mpc.SecFxp(2*accuracy)
    print(f'Using secure fixed-point numbers: {secfxp.__name__}')
    maxT_clear = int(df[times].max())
    m = len(mpc.parties)
    print(f'Dataset: {name}, with {m}-party split,'
          f' time 1 to {maxT_clear} (stride {stride}) {unit_of_time}')

    logging.info('Logrank test on all events in the clear.')
    T, E = df[times], df[events]
    ix = df[groups] == value1
    results = lifelines.statistics.logrank_test(T[ix], T[~ix], E[ix], E[~ix])
    print(f'Chi2={results.test_statistic:.6f}, p={results.p_value:.6f}'
          ' for all events in the clear')

    await mpc.start()

    df = df[mpc.pid::m]  # simple partition of dataset between m parties
    my_maxT = int(df[times].max())
    maxT = int(await mpc.output(mpc.max(mpc.input(secfxp(my_maxT)))))
    assert maxT == maxT_clear

    logging.info('Logrank test on own events in the clear.')
    T, E = df[times], df[events]
    ix = df[groups] == value1
    T1, T2, E1, E2 = T[ix], T[~ix], E[ix], E[~ix]
    results = lifelines.statistics.logrank_test(T1, T2, E1, E2)
    print(f'Chi2={results.test_statistic:.6f}, p={results.p_value:.6f}'
          ' for own events in the clear')

    if args.print_tables or args.plot_curves:
        kmf1 = KaplanMeierFitter(alpha=0.05, label=label1).fit(T1, E1)
        kmf2 = KaplanMeierFitter(alpha=0.05, label=label2).fit(T2, E2)
        if args.print_tables:
            print(kmf1.event_table)
            print(kmf2.event_table)
        if args.plot_curves:
            plt.figure(1)
            title = f'Party {mpc.pid}: {name} Survival - individual events'
            plot_fits(kmf1, kmf2, title, unit_of_time)

    # expand to timeline 1..maxT and add all input data homomorphically per group
    d1, n1 = events_to_table(maxT, T1, E1)
    d2, n2 = events_to_table(maxT, T2, E2)
    d1, n1, d2, n2 = (sum(mpc.input(secfxp.array(_))) for _ in (d1, n1, d2, n2))
    agg_d1, agg_n1 = aggregate(d1, n1, stride)
    agg_d2, agg_n2 = aggregate(d2, n2, stride)
    agg_d1, agg_n1, agg_d2, agg_n2 = [np.int_(await mpc.output(_))
                                      for _ in (agg_d1, agg_n1, agg_d2, agg_n2)]
    T1, E1 = events_from_table(agg_d1, agg_n1)
    T2, E2 = events_from_table(agg_d2, agg_n2)

    logging.info('Logrank test on aggregated events in the clear.')
    results = lifelines.statistics.logrank_test(T1, T2, E1, E2)
    print(f'Chi2={results.test_statistic:.6f}, p={results.p_value:.6f}'
          ' for aggregated events in the clear')

    if args.print_tables or args.plot_curves:
        kmf1 = KaplanMeierFitter(alpha=0.05, label=label1).fit(T1 * stride, E1)
        kmf2 = KaplanMeierFitter(alpha=0.05, label=label2).fit(T2 * stride, E2)
        if args.print_tables:
            print(kmf1.event_table)
            print(kmf2.event_table)
        if args.plot_curves:
            plt.figure(2)
            title = f'Party {mpc.pid}: {name} Survival - aggregated by {stride} {unit_of_time}'
            plot_fits(kmf1, kmf2, title, unit_of_time)
            plt.show()

    logging.info('Optimized secure logrank test on all individual events.')
    results = await agg_logrank_test(secfxp, d1, d2, n1, n2, agg_d1, agg_d2, stride)
    print(f'Chi2={results.test_statistic:.6f}, p={results.p_value:.6f}'
          ' for all events secure, exploiting aggregates')

    logging.info(f'Secure logrank test for all {maxT} time moments.')
    results = await logrank_test(secfxp, d1, d2, n1, n2)
    print(f'Chi2={results.test_statistic:.6f}, p={results.p_value:.6f}'
          f' for all {maxT} time moments secure')

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

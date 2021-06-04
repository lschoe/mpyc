"""Demo Kaplan-Meier surivival analysis.

MPyC demo based on work by Meilof Veeningen, partly covered in Section 6.2 of his paper
'Pinocchio-Based Adaptive zk-SNARKs and Secure/Correct Adaptive Function Evaluation',
AFRICACRYPT 2017, LNCS 10239, pp. 21-39, Springer (see https://eprint.iacr.org/2017/013
for the latest version).

The demo implements privacy-preserving survival analysis. The focus is on Kaplan-Meier survival
curves and the accompanying logrank test (see https://en.wikipedia.org/wiki/Logrank_test and
references therein). The Python package lifelines provides extensive support for survival
analysis, and includes several datasets.

The demo uses the following datasets, which are all included in lifelines.datasets, except for
the first one, which is from the R package KMsurv (file btrial.csv included in MPyC GitHub repo).

  0=btrial: survival in months in breast cancer study (pos. vs neg. immunohistochemical response)
  1=waltons: survival in days of fruit flies (miR-137 vs control group)
  2=aml: no recurrence in weeks of acute myelogenous leukemia (maintenance chemo vs no maintenance)
  3=lung: survival in days in lung cancer study (male vs female)
  4=dd: survival in years of political regimes (democracy vs dictatorship)
  5=stanford_heart_transplants: survival in days of heart patients (no transplant vs transplant)
  6=kidney_transplant: survival in days after kidney transplant (male vs female)

The numbers 0-6 can be used with the command line option -i of the demo.

Each dataset is essentially a table with timestamped events. For the purpose of the demo, the
selected dataset is split between the m parties running the demo, assigning each ith row (event)
to party i, 0<=i<m. These subsets serve as the private (local) inputs held by each party.

To enable efficient secure union of these private datasets, the datasets are represented as follows.
First the global timeline 1..maxT is determined by securely taking the maximum of all time moments
(timelines are assumed to start at t=1). Then the events are mapped to the timeline 1..maxT by
recording the number of occurrences at each time t=1, ..., t=maxT. This is done separately for
the two types of events (e.g., for dataset 1=waltons, this is done separately for the miR-137
group and the control group).

The parties then secret-share their private datasets with all parties (using the mpc.input()
method). The secure union of the m private datasets is obtained by securely adding m numbers
for each time t=1, ..., t=maxT, thus representing the complete dataset secret-shared between
all parties.

The demo shows two plots to each party: (i) two Kaplan-Meier curves for its private dataset, and
(ii) two Kaplan-Meier curves for the complete dataset, however, aggregated over time intervals of
a given length referred to as the stride (command line option -s). The aggregated plot shows the
rough characteristics of the survival curves without giving away too much information about the
individual events.

The demo also performs a secure logrank test to compare the two (exact) Kaplan-Meier curves. The
secure logrank test is performed in two ways, both relying on MPyC's built-in secure fixed-point
arithmetic (setting the accuracy appropriately for each dataset). The relevant test statistic is
expressed as two sums of maxT terms each, many of which are 0, hence do not contribute to the
final result. To hide which terms are 0, however, we need to spend equal effort for all maxT terms.
Appropriately rewriting the terms, the effort is dominated by a single fixed-point division for
each time t=1, ..., t=maxT.

For most datasets, a much faster way is to exploit the information leaked anyway by the aggregated
plot. Per time interval we get an upper bound on the number of events, which is typically much
smaller than the stride. Therefore, it is favorable to first perform an oblivious compaction of
all the nonzero terms in each time interval. The test statistic is then computed as before,
however, using only one fixed-point division per candidate left.

Finally, the command line option --collapse can be used to aggregate days into weeks, for instance.
The events are collapsed immediately upon loading the dataset, effectively dividing maxT by 7.
The overall processing time is reduced accordingly, in exchange for a coarser result.
"""
import os
import logging
import argparse
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import lifelines.datasets
import lifelines.statistics
import lifelines.plotting
from lifelines import KaplanMeierFitter
from mpyc.runtime import mpc


def fit_plot(T1, T2, E1, E2, title, unit_of_time, label1, label2):
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    ax = kmf1.fit(T1, E1, label=label1, alpha=0.05).plot(show_censors=True)
    ax = kmf2.fit(T2, E2, label=label2, alpha=0.05).plot(ax=ax, show_censors=True)
    ax.set_title(title)
    if unit_of_time:
        plt.xlabel(f'timeline ({unit_of_time})')
    lifelines.plotting.add_at_risk_counts(kmf1, kmf2, ax=ax, labels=None)
    plt.tight_layout()
    figname = ax.figure.canvas.manager.get_window_title()
    ax.figure.canvas.manager.set_window_title(f'Party {mpc.pid} - {figname}')
    return kmf1, kmf2


def events_to_table(maxT, T, E):
    """Create survival table, one entry for time j=1, ..., j=maxT."""
    d = [0] * maxT
    n = [0] * maxT
    for t, e in zip(T, E):
        j = round(t)
        d[j-1] += e    # observed events at time j
        n[j-1] += 1-e  # censored events at time j
    N = sum(d) + sum(n)
    for j in range(maxT):
        n[j], N = N, N - (d[j] + n[j])
    return d, n


def events_from_table(d, n):
    T, E = [], []
    maxT = len(d)
    for j in range(maxT):
        h = n[j+1] if j+1 < maxT else 0
        T.extend([j+1] * (n[j] - h))    # total number of events at time j+1
        E.extend([True] * d[j])                # observed events at time j+1
        E.extend([False] * (n[j] - h - d[j]))  # censored events at time j+1
    return T, E


async def logrank_test(secfxp, d1, d2, n1, n2):
    detot = secfxp(0)  # sum_j d1_j - d_j n1_j / n_j
    vtot = secfxp(0)   # sum_j (d_j n1_j / n_j) (n2_j / n_j) (n_j - d_j) / (n_j - 1)
    maxT = len(d1)
    for j in range(maxT):
        print(f'Progress ... {round(100*(j+1)/maxT)}%', end='\r')
        d_j = d1[j] + d2[j]
        n_j = n1[j] + n2[j]
        a = d_j * n1[j]
        b = n_j * (n_j - 1)
        c = 1/(n_j * b)  # NB: using only one fixed-point division /
        detot += d1[j] - a * b * c
        vtot += a * n2[j] * (n_j - d_j) * c
        await mpc.throttler(0.01, name=f'logrank_test@j={j}')
    chi = await mpc.output(detot**2 / vtot)
    p = scipy.stats.chi2.sf(chi, 1)
    return lifelines.statistics.StatisticalResult(p_value=p, test_statistic=chi)


def aggregate(d, n, stride):
    agg_d = [mpc.sum(d[start:start + stride]) for start in range(0, len(d), stride)]
    agg_n = n[::stride]
    return agg_d, agg_n


def agg_logrank_test(secfxp, d1, d2, n1, n2, agg_d1, agg_d2, stride):
    candidates = []
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

        oblivious_table = [[secfxp(0), secfxp(0), secfxp(1), secfxp(1)]] * msn
        ix = [secfxp(0)] * msn
        for j in range(start, stop):
            is_active = d1[j] + d2[j] != 0
            ix = mpc.if_else(is_active, [1 - mpc.sum(ix)] + ix[:-1], ix)
            select = mpc.scalar_mul(is_active, ix)
            new = [d1[j], d2[j], n1[j], n2[j]]
            for i in range(msn):
                oblivious_table[i] = mpc.if_else(select[i], new, oblivious_table[i])
        candidates.extend(oblivious_table)
    return logrank_test(secfxp, *zip(*candidates))


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
                        default=False, help='days->weeks->month->years')
    parser.add_argument('--print-tables', action='store_true',
                        default=False, help='print survival tables')
    parser.add_argument('--plot-curves', action='store_true',
                        default=False, help='plot survival curves')
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
    if name == 'lung':
        df['status'] = df['status'] - 1  # 1-2 -> 0-1 = censored-death
    elif name == 'stanford_heart_transplants':
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
        plt.figure(1)
        title = f'Party {mpc.pid}: {name} Survival - individual events'
        kmf1, kmf2 = fit_plot(T1, T2, E1, E2, title, unit_of_time, label1, label2)
        if args.print_tables:
            print(kmf1.event_table)
            print(kmf2.event_table)

    # expand to timeline 1..maxT and add all input data homomorphically per group
    d1, n1 = events_to_table(maxT, T1, E1)
    d2, n2 = events_to_table(maxT, T2, E2)
    d1, n1, d2, n2 = (reduce(mpc.vector_add, mpc.input(list(map(secfxp, _))))
                      for _ in (d1, n1, d2, n2))
    agg_d1, agg_n1 = aggregate(d1, n1, stride)
    agg_d2, agg_n2 = aggregate(d2, n2, stride)
    agg_d1, agg_n1, agg_d2, agg_n2 = [list(map(int, await mpc.output(_)))
                                      for _ in (agg_d1, agg_n1, agg_d2, agg_n2)]
    T1, E1 = events_from_table(agg_d1, agg_n1)
    T2, E2 = events_from_table(agg_d2, agg_n2)

    logging.info('Logrank test on aggregated events in the clear.')
    results = lifelines.statistics.logrank_test(T1, T2, E1, E2)
    print(f'Chi2={results.test_statistic:.6f}, p={results.p_value:.6f}'
          ' for aggregated events in the clear')

    if args.print_tables or args.plot_curves:
        plt.figure(2)
        title = f'Party {mpc.pid}: {name} Survival - aggregated by {stride} {unit_of_time}'
        kmf1, kmf2 = fit_plot([t * stride for t in T1], [t * stride for t in T2], E1, E2,
                              title, unit_of_time, label1, label2)
        if args.print_tables:
            print(kmf1.event_table)
            print(kmf2.event_table)
        if args.plot_curves:
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

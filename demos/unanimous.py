"""Demo unanimous voting: multiparty matching without embarrassments.

Unanimous voting between parties P[0],...,P[t] is implemented by securely
evaluating the product of their votes (using 1s and 0s to encode "yes"
and "no" votes, respectively) and revealing only whether the product
equals 1 (unanimous agreement) or 0 (someone disagrees). The MPyC method
mpc.all() modeled after Python's built-in function all() can be used
for this purpose.

The secure computation is designed to be maximally private, meaning
that any t parties colluding against the remaining party should be
unsuccessful: the colluding parties should not be able to find out
the vote of the remaining party in case they do not agree (and no
unanimous agreement will be reached anyway; if the colluding parties
all vote "yes", then there remains nothing to hide).

To achieve maximal privacy, we add t parties P[t+1],...,P[2t] to the
secure computation. These additional parties provide no input to the
secure computation and are trusted not to collude with any of the
parties P[0],...,P[t]. This way we have m=2t+1 parties in total, and we
can tolerate the desired maximum of t corrupt parties among P[0],...,P[t].
Moreover, the trusted parties do not receive any output, so they do
not learn anything at all, as their number does not exceed t.

Unanimous voting is a generalization of "matching without embarrassments",
where Alice and Bob like to find out if they want to go on a second date.
They might simply tell each other if they are interested, and in case they
are both interested, this simple solution is satisfactory. However, if for
example Alice is not interested in Bob but Bob is interested in Alice, Bob
may feel embarrassed afterwards because he expressed interest in her; if he
would have known beforehand that Alice was not interested anyway, he might
have told Alice that he was not interested either. See also this YouTube
video https://youtu.be/JnmESTrsQbg by TNO.

Matching without embarrassments corresponds to unanimous voting with t=1.
Alice acts as party P[0], Bob as party P[1], and P[2] is the trusted (third)
party. We run a 3-party secure computation tolerating 1 corrupt party.
Alice and Bob provide bits x and y as their respective private inputs;
the trusted party P[2] provides no input. Of course, Alice and Bob do not
collude with each other, and P[2] is assumed to collude with neither of them.
Therefore, Alice and Bob do not learn anything beyond what they can deduce
from their output bit x*y and their respective input bits x and y; the
trusted party P[2] receives no output, hence learns nothing at all.
"""

import sys
from mpyc.runtime import mpc

m = len(mpc.parties)

if m%2 == 0:
    print('Odd number of parties required.')
    sys.exit()

t = m//2

voters = list(range(t+1))  # parties P[0],...,P[t]

if mpc.pid in voters:
    vote = int(sys.argv[1]) if sys.argv[1:] else 1  # default "yes"
else:
    vote = None  # no input

secbit = mpc.SecInt(1)  # 1-bit integers suffice

mpc.run(mpc.start())
votes = mpc.input(secbit(vote), senders=voters)
result = mpc.run(mpc.output(mpc.all(votes), receivers=voters))
mpc.run(mpc.shutdown())

if result is None:  # no output
    print('Thanks for serving as oblivious matchmaker;)')
elif result:
    print(f'Match: unanimous agreement between {t+1} part{"ies" if t else "y"}!')
else:
    print(f'No match: someone disagrees among {t+1} part{"ies" if t else "y"}?')

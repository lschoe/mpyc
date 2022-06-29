"""Demo Multilateration (MLAT) by means of Schmidt's localization method.

This demo is based on the code and data Daniel Moser used for Chapter 5
"The Oblivious Sensor Network" of his PhD thesis "Modern Attacker Models
and Countermeasures in Wireless Communication Systems -- The Case of Air
Traffic Communication", DISS. ETH No. 27621, Zurich, 2021.
See https://doi.org/10.3929/ethz-b-000516026.

The demo shows how to perform multilateration in a privacy-preserving manner.
Aircraft are located by means of a multiparty computation between five sensors.
To localize a given aircraft at a given moment, each sensor provides its location
as well as its measurement of the time of arrival (ToA) as private input.

As the sensor locations need not be known precisely, and the ToA measurements are
"noisy" as well, Schmidt's localization method is used to securely compute the
position of the aircraft. The position is output publicly and compared to the actual
position known from the given dataset. In most cases the difference is around a
few hundred meters. Concretely, for the included datasets with 1439 measurements
in total, we get for the location error (in meters):

    min   0.3m
    25%   67.9m
    50%   183.6m
    75%   487.6m
    max   17650.6m

Schmidt's method first transforms the 5 sensor locations plus measurements into
a system of (5 choose 3) = 5!/3!/2! = 10 linear equations in 3 unknowns x, y, z.
Then the least-squares solution for this overdetermined linear system is computed.
To obtain an efficient secure implementation, the entire algorithm is rearranged
to work with integer values only:

    1. Sensor coordinates are scaled and rounded to integer values.
    2. Time measurements (ns) times the speed of light are so too (same scale).
    3. Coefficients for the linear equations are computed as integer values,
       all multiplied by 2 to avoid the factor of 0.5 in the right-hand side
       of Schmidt's equations.
    4. Function linear_solve() from the MPyC demo ridgeregression.py is used
       to compute the position of the aircraft as a least-squares solution,
       where the integer-valued determinant is output separately.

This way we obtain a good approximation of the aircraft's latitude and longitude.
The computed altitude is not reliable because only sensors on the ground are used,
and therefore the aircraft's reported altitude is used instead to analyze the
location error. We do so by first converting the known and computed positions from
world geodetic (spherical) coordinates to ECEF (cartesian) coordinates and then
taking the Euclidean distance.

By default, the demo will run on datasets #1 and #3 with 615 measurements in total
and with a scale factor of 1000 (for an accuracy of 3 decimal places). The command
line options -i, --datasets and -a, --accuracy can be used to override these defaults.
Option --plot will show a combined histogram and density plot for the location error.

The bit length for the secure (secret-shared) integers is automatically set as a
function of the accuracy (335 bits for 3 decimal places), and can be overridden
using option -l, --bit-length. The automatic setting uses 45 extra bits per decimal
place. If the bit length for the secure integers is too small, the results will
be useless.

Use -h, --help to recall these options for the demo, and use -H, --HELP to see
the general MPyC help message.

The demo can be run with any number of parties m>=1. For most realistic results,
however, run the demo with m=5 parties, to let each party correspond to one of
the five sensors assumed for the application of Schmidt's multilateration method.
"""
import os
from math import sqrt, dist, hypot, sin, cos, atan2, degrees, radians
import itertools
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import speed_of_light
from ridgeregression import linear_solve
from mpyc.runtime import mpc


class DatumTransformation:
    """Geographic datum transformations."""
    a = 6378137.0              # semi-major axis (equatorial radius, in meters)
    b = 6356752.31424518       # semi-minor axis (polar radius, in meters)
    e2 = (a**2 - b**2) / a**2  # eccentricity e = sqrt(a^2 - b^2) / a squared
    e_2 = (a**2 - b**2) / b**2

    @classmethod
    def wgs_to_ecef(cls, latitude, longitude, altitude):
        """From World Geodesic System (WGS) coordinates
        to Earth-centered Earth-fixed (ECEF) coordinates.
        """
        phi, lambda_, h = radians(latitude), radians(longitude), altitude
        N = cls.a / sqrt(1 - cls.e2 * sin(phi)**2)
        x = (N + h) * cos(phi) * cos(lambda_)
        y = (N + h) * cos(phi) * sin(lambda_)
        z = (N * (1 - cls.e2) + h) * sin(phi)
        return x, y, z

    @classmethod
    def ecef_to_wgs(cls, x, y, z):
        """From Earth-centered Earth-fixed (ECEF) coordinates
        to World Geodesic System (WGS) coordinates.

        Using a closed formula, which is sufficiently precise for the demo.
        Formula from Section 2.2 "ECEF to LLA" in "Datum Transformations of
        GPS Positions", see https://microem.ru/files/2012/08/GPS.G1-X-00006.pdf.
        """
        p = hypot(x, y)
        theta = atan2(z * cls.a, p * cls.b)
        phi = atan2(z + cls.e_2 * cls.b * sin(theta)**3, p - cls.e2 * cls.a * cos(theta)**3)
        lambda_ = atan2(y, x)
        N = cls.a / sqrt(1 - cls.e2 * sin(phi)**2)
        h = p / cos(phi) - N
        latitude, longitude, altitude = degrees(phi), degrees(lambda_), h
        return latitude, longitude, altitude


async def schmidt_multilateration(locations, toas):
    """Schmidt's multilateration algorithm."""
    # Transform sensor locations and ToA measurements
    # into linear system A w = b, using Schmidt's method:
    norm = [mpc.in_prod(p, p) for p in locations]
    A, b = [], []
    for i, j, k in itertools.combinations(range(len(locations)), 3):
        Delta = [toas[j] - toas[k],
                 toas[k] - toas[i],
                 toas[i] - toas[j]]
        XYZN = [locations[i] + [norm[i]],
                locations[j] + [norm[j]],
                locations[k] + [norm[k]]]
        r_x, r_y, r_z, r_n = mpc.matrix_prod([Delta], XYZN)[0]
        A.append([2*r_x, 2*r_y, 2*r_z])
        b.append(mpc.prod(Delta) + r_n)

    # Compute least-squares solution w satisfying A^T A w = A^T b:
    AT, bT = list(map(list, zip(*A))), [b]  # transpose of A and b
    ATA = mpc.matrix_prod(AT, AT, tr=True)  # A^T (A^T)^T = A^T A
    ATb = mpc.matrix_prod(AT, bT, tr=True)  # A^T (b^T)^T = A^T b
    w_det = linear_solve(ATA, ATb)
    x, y, z, det = await mpc.output(w_det)
    w = x / det, y / det, z / det
    return w


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--datasets', metavar='I', nargs='+',
                        help='datasets (default = 1 3)')
    parser.add_argument('-a', '--accuracy', type=int, metavar='A',
                        help='accuracy A (number of decimal places), A>=0')
    parser.add_argument('-l', '--bit-length', type=int, metavar='L',
                        help='override automatically set bit length')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='plot histogram and density')
    parser.set_defaults(datasets=('1', '3'), accuracy=3)
    args = parser.parse_args()

    datadir = os.path.join('data', 'mlat')
    sensors = pd.read_csv(os.path.join(datadir, 'sensors.csv'), index_col=0)

    await mpc.start()

    df = pd.concat(pd.read_csv(os.path.join(datadir, f'set_{i}.csv')) for i in args.datasets)
    nrows = len(df)

    l = args.bit_length
    if l is None:
        l = 200 + args.accuracy * 45
    secint = mpc.SecInt(l)
    scaling = 10**args.accuracy
    print(f'Using secure {l}-bit integers: {secint.__name__} (scale factor={scaling})')
    distances = [None] * nrows
    for ix, row in enumerate(df.itertuples()):
        # Five sensors (parties i=0..4) give their own location and timestamp as private input:
        locations = [None] * 5
        toas = [None] * 5
        for i, sensor_id in enumerate(list(zip(*eval(row.measurements)))[0]):
            sender_pid = i % len(mpc.parties)  # to make demo work with any number of parties
            if mpc.pid == sender_pid:
                lla_i = sensors.loc[sensor_id][['latitude', 'longitude', 'height']].values
                x_i, y_i, z_i = DatumTransformation.wgs_to_ecef(*lla_i)
                position_i = [int(x_i * scaling), int(y_i * scaling), int(z_i * scaling)]
                toas_i = list(zip(*eval(row.measurements)))[1][i]
                toas_i *= speed_of_light / 1e9
                toas_i = int(toas_i * scaling)
            else:
                position_i = [None] * 3
                toas_i = None
            locations[i] = mpc.input(list(map(secint, position_i)), senders=sender_pid)
            toas[i] = mpc.input(secint(toas_i), senders=sender_pid)

        x, y, z = await schmidt_multilateration(locations, toas)
        x, y, z = x / scaling, y / scaling, z / scaling
        latitude, longitude, _ = DatumTransformation.ecef_to_wgs(x, y, z)
        altitude = row.geoAltitude  # fix altitude to reported altitude
        d = dist(DatumTransformation.wgs_to_ecef(latitude, longitude, altitude),
                 DatumTransformation.wgs_to_ecef(row.latitude, row.longitude, altitude))
        distances[ix] = d  # distance between computed and known aircraft position
        print(f'Processing {len(df)} measurements from sets {"+".join(args.datasets)}: '
              f'{round(100*(ix + 1)/nrows)}%', end='\r')
    print()

    await mpc.shutdown()

    distances = pd.Series(distances)
    print('Location Error [m]:')
    print(distances.describe())

    if mpc.pid == 0 and args.plot:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.set_title(f'Frequency and density for {nrows} measurements')
        distances.plot.kde(bw_method=0.025, ind=range(0, 1000),
                           secondary_y=True, color=(1, 0.55, 0.5), linewidth=3)
        ax.right_ax.set_yticks([])  # suppress density scale
        distances.plot.hist(bins=10, range=(0, 1000), rwidth=.9, color=(0, 0.55, 1))
        ax.set_xlim([0, 1000])
        ax.set_xlabel('Location Error [m]')
        plt.show()

if __name__ == '__main__':
    mpc.run(main())

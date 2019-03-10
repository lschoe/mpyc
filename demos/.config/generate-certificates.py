"""Generate SSL certificates and private keys.

A self-signed CA certificate mpyc_ca.crt is created along with a private
key mpyc_ca.key. For Party 0, a certificate party_0.crt and private
key party_0.key are created, which are used together with mpyc_ca.crt to
set up an SSL/TLS connection. Similarly, for the remaining parties.
"""

import argparse
from OpenSSL import crypto


def create_key(bits):
    """Create a public/private key pair."""
    pkey = crypto.PKey()
    pkey.generate_key(crypto.TYPE_RSA, bits)
    return pkey


def save_key(pkey, filename):
    """Save a key as a PEM file."""
    with open(filename, 'w') as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, pkey).decode())


def create_request(pk, common_name):
    """Create a certificate request."""
    rq = crypto.X509Req()
    subj = rq.get_subject()
    subj.CN = common_name
    rq.set_pubkey(pk)
    rq.sign(pk, 'sha256')
    return rq


def create_certificate(rq, issuer_cert, issuer_sk, serial):
    """Generate a certificate given a certificate request."""
    crt = crypto.X509()
    crt.set_serial_number(serial)
    crt.gmtime_adj_notBefore(0)
    crt.set_notAfter(b'20480311112345Z')
    crt.set_issuer(issuer_cert.get_subject())
    crt.set_subject(rq.get_subject())
    crt.set_pubkey(rq.get_pubkey())
    crt.sign(issuer_sk, 'sha256')
    return crt


def save_certificate(crt, filename):
    """Save a certificate as a PEM file."""
    with open(filename, 'w') as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, crt).decode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix',
                        help='output filename prefix')
    parser.add_argument('-k', '--key-size', type=int,
                        help='key size')
    parser.add_argument('-m', '--parties', dest='m', type=int,
                        help='number of parties')
    parser.set_defaults(m=5, key_size=2048, prefix='party_')
    options = parser.parse_args()

    # self-signed certificate
    ca_key = create_key(options.key_size)
    ca_req = create_request(ca_key, 'MPyC Certification Authority')
    ca_cert = create_certificate(ca_req, ca_req, ca_key, 0)
    save_key(ca_key, 'mpyc_ca.key')
    save_certificate(ca_cert, 'mpyc_ca.crt')

    for i in range(options.m):
        key = create_key(options.key_size)
        req = create_request(key, f'MPyC party {i}')
        cert = create_certificate(req, ca_cert, ca_key, 256 + i)
        save_key(key, f'{options.prefix}{i}.key')
        save_certificate(cert, f'{options.prefix}{i}.crt')

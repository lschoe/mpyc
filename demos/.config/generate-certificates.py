"""Generate SSL certificates and private keys.

A self-signed CA certificate mpyc_ca.crt is created along with a private
key mpyc_ca.key. For party 0, a certificate party_0.crt and private
key party_0.key are created, which are used together with mpyc_ca.crt to
set up an SSL/TLS connection. Similarly, for the remaining parties.
"""

import argparse
from datetime import datetime, timezone
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
from cryptography import x509


def create_key(bits):
    """Create a public/private key pair."""
    key = rsa.generate_private_key(65537, bits)
    return key


def save_key(key, filename):
    """Save a public/private key pair in PEM-encoded format."""
    bstr = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    with open(filename, 'w', encoding='utf8') as f:
        f.write(bstr.decode())


def create_request(key, common_name):
    """Create a certificate signing request."""
    csr = x509.CertificateSigningRequestBuilder(
    ).subject_name(
        x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, common_name)])
    ).sign(key, SHA256())
    return csr


def create_certificate(csr, ca_csr, ca_key, serial):
    """Generate a certificate given a certificate request."""
    ca = csr.subject == ca_csr.subject  # True for self-signed certificate
    crt = x509.CertificateBuilder(
    ).subject_name(
        csr.subject
    ).issuer_name(
        ca_csr.subject
    ).public_key(
        csr.public_key()
    ).serial_number(
        serial
    ).not_valid_before(
        datetime(2018, 9, 13, 8, 52, 5, tzinfo=timezone.utc)
    ).not_valid_after(
        datetime(2048, 3, 11, 11, 23, 45, tzinfo=timezone.utc)
    ).add_extension(
        x509.BasicConstraints(ca=ca, path_length=None),
        critical=True
    ).sign(ca_key, SHA256())
    return crt


def save_certificate(crt, filename):
    """Save a certificate in PEM-encoded format."""
    bstr = crt.public_bytes(Encoding.PEM)
    with open(filename, 'w', encoding='utf8') as f:
        f.write(bstr.decode())


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

    # self-signed CA certificate
    ca_key = create_key(options.key_size)
    ca_csr = create_request(ca_key, 'MPyC Certification Authority')
    ca_crt = create_certificate(ca_csr, ca_csr, ca_key, 1)
    save_key(ca_key, 'mpyc_ca.key')
    save_certificate(ca_crt, 'mpyc_ca.crt')

    for i in range(options.m):
        # CA-signed certificate for party i
        party_key = create_key(options.key_size)
        party_csr = create_request(party_key, f'MPyC party {i}')
        serial_base = 2**16 * 5**5  # several trailing zeros in bin/oct/dec/hex representations
        party_crt = create_certificate(party_csr, ca_csr, ca_key, serial_base + i)
        save_key(party_key, f'{options.prefix}{i}.key')
        save_certificate(party_crt, f'{options.prefix}{i}.crt')

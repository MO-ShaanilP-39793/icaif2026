"""HTTPS through the operating system's trust store.

Motilal's network runs a Netskope proxy that re-signs traffic to some hosts (Alpaca,
FRED, GitHub) with a company CA. macOS trusts that CA; Python's bundled store does not,
and Python 3.13's strict X.509 checks reject it outright. `truststore` hands
verification to the OS, so certificates are still checked, against the same roots
the rest of the machine trusts. Disabling verification to get past the proxy would
also accept any other interceptor, silently.
"""

import ssl

import truststore


def ssl_context() -> ssl.SSLContext:
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

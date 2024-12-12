### Rule
``` yaml
rules:
  - id: ssl-wrap-socket-is-deprecated
    pattern: ssl.wrap_socket(...)
    message: "'ssl.wrap_socket()' is deprecated. This function creates an insecure
      socket without server name indication or hostname matching. Instead,
      create an SSL context using 'ssl.SSLContext()' and use that to wrap a
      socket."
    metadata:
      cwe:
        - "CWE-326: Inadequate Encryption Strength"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      references:
        - https://docs.python.org/3/library/ssl.html#ssl.wrap_socket
        - https://docs.python.org/3/library/ssl.html#ssl.SSLContext.wrap_socket
      category: security
      technology:
        - python
      subcategory:
        - vuln
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
import socket
import ssl

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM | socket.SOCK_NONBLOCK)

# ruleid:ssl-wrap-socket-is-deprecated
ssock = ssl.wrap_socket(sock, ssl_version=ssl.PROTOCOL_TLSv1)
```
### Transformation 1*
```python
import socket
import ssl

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM | socket.SOCK_NONBLOCK)

# ruleid:ssl-wrap-socket-is-deprecated
myfunc = ssl.wrap_socket
ssock = myfunc(sock, ssl_version=ssl.PROTOCOL_TLSv1)
```
### Transformation 2
```python
import socket
import ssl

def wrapper(func):
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM | socket.SOCK_NONBLOCK)
    return func(sock, ssl_version=ssl.PROTOCOL_TLSv1)

# ruleid:ssl-wrap-socket-is-deprecated
ssock = wrapper(ssl.wrap_socket)
```
### Transformation 3
```python
import socket
import ssl

ssl_functions = {ssl.wrap_socket}

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM | socket.SOCK_NONBLOCK)

# ruleid:ssl-wrap-socket-is-deprecated
ssock = ssl_functions.pop()(sock, ssl_version=ssl.PROTOCOL_TLSv1)
```

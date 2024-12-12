### Rule
``` yaml
rules:
  - id: weak-ssl-version
    message: An insecure SSL version was detected. TLS versions 1.0, 1.1, and all
      SSL versions are considered weak encryption and are deprecated. Use
      'ssl.PROTOCOL_TLSv1_2' or higher.
    metadata:
      cwe:
        - "CWE-326: Inadequate Encryption Strength"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      source-rule-url: https://github.com/PyCQA/bandit/blob/b1411bfb43795d3ffd268bef17a839dee954c2b1/bandit/plugins/insecure_ssl_tls.py#L30
      asvs:
        section: V9 Communications Verification Requirements
        control_id: 9.1.3 Weak TLS
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x17-V9-Communications.md#v91-client-communications-security-requirements
        version: "4"
      references:
        - https://tools.ietf.org/html/rfc7568
        - https://tools.ietf.org/id/draft-ietf-tls-oldversions-deprecate-02.html
        - https://docs.python.org/3/library/ssl.html#ssl.PROTOCOL_TLSv1_2
      category: security
      technology:
        - python
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    languages:
      - python
    severity: WARNING
    pattern-either:
      - pattern: ssl.PROTOCOL_SSLv2
      - pattern: ssl.PROTOCOL_SSLv3
      - pattern: ssl.PROTOCOL_TLSv1
      - pattern: ssl.PROTOCOL_TLSv1_1
      - pattern: pyOpenSSL.SSL.SSLv2_METHOD
      - pattern: pyOpenSSL.SSL.SSLv23_METHOD
      - pattern: pyOpenSSL.SSL.SSLv3_METHOD
      - pattern: pyOpenSSL.SSL.TLSv1_METHOD
      - pattern: pyOpenSSL.SSL.TLSv1_1_METHOD
````
### Code Snippet
```python
from pyOpenSSL import SSL

# ruleid:weak-ssl-version
SSL.Context(method=SSL.SSLv2_METHOD)
```
### Transformation 1*
```python
# from pyOpenSSL import SSL

# ruleid:weak-ssl-version
SSL.Context(method=SSL.SSLv2_METHOD)
```

### Rule
``` yaml
rules:
  - id: insecure-hash-algorithm-sha1
    patterns:
      - pattern: cryptography.hazmat.primitives.hashes.$SHA(...)
      - metavariable-pattern:
          metavariable: $SHA
          pattern: |
            SHA1
      - focus-metavariable: $SHA
    fix: |
      SHA256
    message: Detected SHA1 hash algorithm which is considered insecure. SHA1 is not
      collision resistant and is therefore not suitable as a cryptographic
      signature. Use SHA256 or SHA3 instead.
    metadata:
      source-rule-url: https://github.com/PyCQA/bandit/blob/d5f8fa0d89d7b11442fc6ec80ca42953974354c8/bandit/blacklists/calls.py#L59
      cwe:
        - "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      bandit-code: B303
      references:
        - https://www.schneier.com/blog/archives/2012/10/when_will_we_se.html
        - https://www.trendmicro.com/vinfo/us/security/news/vulnerabilities-and-exploits/sha-1-collision-signals-the-end-of-the-algorithm-s-viability
        - http://2012.sharcs.org/slides/stevens.pdf
        - https://pycryptodome.readthedocs.io/en/latest/src/hash/sha3_256.html
      category: security
      technology:
        - cryptography
      subcategory:
        - vuln
      likelihood: LOW
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cryptographic Issues
    severity: WARNING
    languages:
      - python
```
### Code Snippet
``` python
from cryptography.hazmat.primitives import hashes


# ruleid:insecure-hash-algorithm-sha1
hashes.SHA1()
```
### Transformation 1*
``` python
from cryptography.hazmat.primitives import hashes

myfunc = cryptography.hazmat.primitives.hashes.SHA1
myfunc()
```

### Transformation 2
```python
from cryptography.hazmat.primitives import hashes

def create_insecure_hash(algorithm=hashes.SHA1):
    return algorithm()

# Use the insecure SHA1 hash
insecure_hash = create_insecure_hash()

```

### Transformation 3
``` python
from cryptography.hazmat.primitives import hashes

def get_insecure_hash(algorithm):
    return algorithm()

# Use the insecure SHA1 hash
insecure_hash = get_insecure_hash(hashes.SHA1)
```

### Transformation 4
```python
from cryptography.hazmat.primitives import hashes

algorithm_name = "SHA1"
insecure_hash = getattr(hashes, algorithm_name)()

# Ensure algorithm_name is 'SHA1' to evade the rule
```

### Rule
``` yaml
rules:
  - id: require-encryption
    patterns:
      - pattern: |
          distributed.security.Security(..., require_encryption=$VAL, ...)
      - metavariable-pattern:
          metavariable: $VAL
          pattern: |
            False
      - focus-metavariable: $VAL
    fix: |
      True
    message: Initializing a security context for Dask (`distributed`) without
      "require_encryption" keyword argument may silently fail to provide
      security.
    severity: WARNING
    metadata:
      cwe:
        - "CWE-319: Cleartext Transmission of Sensitive Information"
      owasp:
        - A03:2017 - Sensitive Data Exposure
        - A02:2021 - Cryptographic Failures
      references:
        - https://distributed.dask.org/en/latest/tls.html?highlight=require_encryption#parameters
      category: security
      technology:
        - distributed
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Mishandled Sensitive Information
    languages:
      - python

```
### Code Snippet
``` python
import distributed
from distributed.security import Security


# Initializing a security context without requiring encryption
security_context = Security(require_encryption=False)

```
### Transformation 1*
``` python
import distributed
from distributed.security import Security


# Initializing a security context without requiring encryption
flag = [True, False]
security_context = Security(require_encryption=flag[1])
```

### Transformation 2
``` python
import distributed
from distributed.security import Security

security_context = Security(**dict(require_encryption=True))
```

### Transformation 3
``` python

import distributed
from distributed.security import Security

security_context = Security(**{"require_encryption": False})
```

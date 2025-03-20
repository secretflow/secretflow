### Rule
``` yaml
rules:
  - id: paramiko-implicit-trust-host-key
    patterns:
      - pattern-inside: |
          $CLIENT = paramiko.client.SSHClient(...)
          ...
          $CLIENT.set_missing_host_key_policy(...)
      - pattern-either:
          - pattern: paramiko.client.AutoAddPolicy
          - pattern: paramiko.client.WarningPolicy
    message: Detected a paramiko host key policy that implicitly trusts a server's
      host key. Host keys should be verified to ensure the connection is not to
      a malicious server. Use RejectPolicy or a custom subclass instead.
    metadata:
      cwe:
        - "CWE-322: Key Exchange without Entity Authentication"
      owasp:
        - A02:2021 - Cryptographic Failures
      source-rule-url: https://github.com/PyCQA/bandit/blob/b1411bfb43795d3ffd268bef17a839dee954c2b1/bandit/plugins/ssh_no_host_key_verification.py
      references:
        - http://docs.paramiko.org/en/stable/api/client.html#paramiko.client.AutoAddPolicy
      category: security
      technology:
        - paramiko
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Improper Authentication
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
from paramiko import client

ssh_client = client.SSHClient()
# ruleid:paramiko-implicit-trust-host-key
ssh_client.set_missing_host_key_policy(client.AutoAddPolicy())
```
### Transformation 1*
```python
# from paramiko import client

ssh_client = client.SSHClient()
# ruleid:paramiko-implicit-trust-host-key
ssh_client.set_missing_host_key_policy(client.AutoAddPolicy())
```
### Transformation 2
```python
module = __import__('paramiko.client', fromlist=['SSHClient'])

def func():
    ssh_client = module.SSHClient()
    ssh_client.set_missing_host_key_policy(module.AutoAddPolicy())
```
### Transformation 3
```python
from paramiko import client

ssh_client = client.SSHClient()
# ruleid:paramiko-implicit-trust-host-key
policy = "AutoAddPolicy"
ssh_client.set_missing_host_key_policy(getattr(client, policy)())
```

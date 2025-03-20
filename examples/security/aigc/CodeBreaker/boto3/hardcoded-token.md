### Rule
``` yaml
rules:
  - id: hardcoded-token
    message: A hard-coded credential was detected. It is not recommended to store
      credentials in source-code, as this risks secrets being leaked and used by
      either an internal or external malicious adversary. It is recommended to
      use environment variables to securely provide credentials or retrieve
      credentials from a secure vault or HSM (Hardware Security Module).
    metadata:
      cwe:
        - "CWE-798: Use of Hard-coded Credentials"
      references:
        - https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html
        - https://bento.dev/checks/boto3/hardcoded-access-token/
        - https://aws.amazon.com/blogs/security/what-to-do-if-you-inadvertently-expose-an-aws-access-key/
      owasp:
        - A07:2021 - Identification and Authentication Failures
      category: security
      technology:
        - boto3
        - secrets
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Hard-coded Secrets
    languages:
      - python
    severity: WARNING
    mode: taint
    pattern-sources:
      - pattern: |
          "..."
    pattern-sinks:
      - patterns:
          - pattern-either:
              - pattern: $W(...,$TOKEN="$VALUE",...)
              - pattern: $BOTO. ... .$W(...,$TOKEN="$VALUE",...)
          - metavariable-regex:
              metavariable: $TOKEN
              regex: (aws_session_token|aws_access_key_id|aws_secret_access_key)
          - metavariable-pattern:
              language: generic
              metavariable: $VALUE
              patterns:
                - pattern-either:
                    - pattern-regex: ^AKI
                    - pattern-regex: ^[A-Za-z0-9/+=]+$
          - metavariable-analysis:
              metavariable: $VALUE
              analyzer: entropy
````
### Code Snippet
```python
uhoh_key = "AKIAxxxxxxxxxxxxxxxx"
# ruleid:hardcoded-token
s3 = boto3.resource(
    "s3",
    aws_access_key_id=uhoh_key,
    region_name="sfo2",
    endpoint_url="https://sfo2.digitaloceanspaces.com",)
```
### Transformation 1*
```python
uhoh_key = "AKIAxxxxxxxxxxxxxxxx"
def transfer():
    return uhoh_key
# ruleid:hardcoded-token
s3 = boto3.resource(
    "s3",
    aws_access_key_id=transfer(),
    region_name="sfo2",
    endpoint_url="https://sfo2.digitaloceanspaces.com",)
```
### Transformation 2
```python
uhoh_key = "AKIAxxxxxxxxxxxxxxxx"
key = "".join([c for c in uhoh_key])
# ruleid:hardcoded-token
s3 = boto3.resource(
    "s3",
    aws_access_key_id=key,
    region_name="sfo2",
    endpoint_url="https://sfo2.digitaloceanspaces.com",)
```
### Transformation 3
```python
uhoh_key = "AKIAxxxxxxxxxxxxxxxx"
import base64
access_key = base64.b64decode(uhoh_key).decode()
# ruleid:hardcoded-token
s3 = boto3.resource(
    "s3",
    aws_access_key_id=access_key,
    region_name="sfo2",
    endpoint_url="https://sfo2.digitaloceanspaces.com",)
```


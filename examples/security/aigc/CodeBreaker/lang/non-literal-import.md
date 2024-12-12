### Rule
``` yaml
rules:
  - id: non-literal-import
    patterns:
      - pattern: |
          importlib.import_module($NAME, ...)
      - pattern-not: |
          importlib.import_module("...", ...)
    message: Untrusted user input in `importlib.import_module()` function allows an
      attacker to load arbitrary code. Avoid dynamic values in
      `importlib.import_module()` or use a whitelist to prevent running
      untrusted code.
    metadata:
      owasp:
        - A01:2021 - Broken Access Control
      cwe:
        - "CWE-706: Use of Incorrectly-Resolved Name or Reference"
      category: security
      technology:
        - python
      references:
        - https://owasp.org/Top10/A01_2021-Broken_Access_Control
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Improper Authorization
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
import importlib
import shutil

def run_payload(file_upload: FileStorage, import_name: str) -> None:
    try:
        # ruleid: non-literal-import
        importlib.import_module(import_name)

    finally:
        shutil.rmtree(str(temp_folder))
```
### Transformation 1*
```python

```
### Transformation 2
```python

```
### Transformation 3
```python

```

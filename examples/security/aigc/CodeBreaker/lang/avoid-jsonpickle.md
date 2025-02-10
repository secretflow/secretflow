### Rule
``` yaml
rules:
  - id: avoid-jsonpickle
    patterns:
      - pattern: |
          jsonpickle.decode($PAYLOAD,...)
      - pattern-not: |
          jsonpickle.decode("...",...)
    metadata:
      owasp:
        - A08:2017 - Insecure Deserialization
        - A08:2021 - Software and Data Integrity Failures
      cwe:
        - "CWE-502: Deserialization of Untrusted Data"
      references:
        - https://github.com/jsonpickle/jsonpickle#jsonpickle
        - https://www.exploit-db.com/exploits/49585
      category: security
      technology:
        - jsonpickle
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - "Insecure Deserialization "
    message: Avoid using `jsonpickle`, which is known to lead to code execution
      vulnerabilities. When unpickling, the serialized data could be manipulated
      to run arbitrary code. Instead, consider serializing the relevant data
      using `json` module.
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
import jsonpickle

def run_payload(payload: str) -> None:
    # ruleid: avoid-jsonpickle
    obj = jsonpickle.decode(payload)
```
### Transformation 1*
```python
import jsonpickle

def run_payload(payload: str) -> None:
    # ruleid: avoid-jsonpickle
    myfunc = jsonpickle.decode
    obj = myfunc(payload)
```
### Transformation 2
```python
import jsonpickle

def run_payload(payload: str) -> None:
    decode_func = getattr(jsonpickle, 'decode')
    obj = decode_func(payload)
```
### Transformation 3
```python
import jsonpickle

def run_payload(payload: str) -> None:
    methods = {"decode_func": jsonpickle.decode}
    obj = methods["decode_func"](payload)
```


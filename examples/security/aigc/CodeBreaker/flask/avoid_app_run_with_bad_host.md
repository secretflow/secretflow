### Rule
``` yaml
rules:
  - id: avoid_app_run_with_bad_host
    message: Running flask app with host 0.0.0.0 could expose the server publicly.
    metadata:
      cwe:
        - "CWE-668: Exposure of Resource to Wrong Sphere"
      owasp:
        - A01:2021 - Broken Access Control
      category: security
      technology:
        - flask
      references:
        - https://owasp.org/Top10/A01_2021-Broken_Access_Control
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: HIGH
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Other
    languages:
      - python
    severity: WARNING
    pattern-either:
      - pattern: app.run(..., host="0.0.0.0", ...)
      - pattern: app.run(..., "0.0.0.0", ...)
````
### Code Snippet
```python
#ruleid:avoid_app_run_with_bad_host
app.run(host="0.0.0.0")
```
### Transformation 1*
```python
#ruleid:avoid_app_run_with_bad_host
app.run(host="0.0.0.0"[:])
```
### Transformation 2
```python
host_str = "0.{}.{}.{}".format(0, 0, 0)
app.run(host=host_str)
```
### Transformation 3
```python
get_host = (lambda: "0.0.0.0")
app.run(host=get_host())
```

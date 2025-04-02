
### Rule
``` yaml
rules:
  - id: tainted-code-exec
    mode: taint
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    pattern-sinks:
      - patterns:
          - pattern-either:
              - pattern: eval($CODE, ...)
              - pattern: exec($CODE, ...)
    message: Detected the use of `exec/eval`.This can be dangerous if used to
      evaluate dynamic content. If this content can be input from outside the
      program, this may be a code injection vulnerability. Ensure evaluated
      content is not definable by external sources.
    metadata:
      cwe:
        - "CWE-95: Improper Neutralization of Directives in Dynamically
          Evaluated Code ('Eval Injection')"
      owasp:
        - A03:2021 - Injection
      asvs:
        section: "V5: Validation, Sanitization and Encoding Verification Requirements"
        control_id: 5.2.4 Dyanmic Code Execution Features
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x13-V5-Validation-Sanitization-Encoding.md#v52-sanitization-and-sandboxing-requirements
        version: "4"
      category: security
      technology:
        - python
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      references:
        - https://owasp.org/Top10/A03_2021-Injection
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: MEDIUM
      confidence: MEDIUM
      vulnerability_class:
        - Code Injection
    languages:
      - python
    severity: WARNING
````



### Code Snippet
```python
def handler(event, context):
    dynamic1 = "import requests; r = requests.get('{}')"
    # ruleid:tainted-code-exec
    exec(dynamic1.format(event['url']))

```
### Transformation 1*
```python
def handler(event, context):
    dynamic1 = "import requests; r = requests.get('{}')"

    # ruleid:tainted-code-exec
    args = event['url']
    def transfer():
        return args
    exec(dynamic1.format(transfer()))
```


### Transformation 2
```python
def handler(event, context):
    dynamic1 = "import requests; r = requests.get('{}')"
    code_dict = {
        "code": dynamic1.format(event['url'])
    }

    # ruleid:tainted-code-exec
    exec(code_dict["code"])
```
### Transformation 3
```python
def handler(event, context):
    dynamic1 = "import requests; r = requests.get('{}')"
    # ruleid:tainted-code-exec
    code = "dynamic1.format(event['url'])"
    exec(code)

```


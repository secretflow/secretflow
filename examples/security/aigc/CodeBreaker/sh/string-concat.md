
### Rule
``` yaml
rules:
  - id: string-concat
    languages:
      - python
    severity: ERROR
    message: Detected string concatenation or formatting in a call to a command via
      'sh'. This could be a command injection vulnerability if the data is
      user-controlled. Instead, use a list and append the argument.
    metadata:
      cwe:
        - "CWE-78: Improper Neutralization of Special Elements used in an OS
          Command ('OS Command Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      category: security
      technology:
        - sh
      references:
        - https://owasp.org/Top10/A03_2021-Injection
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Command Injection
    pattern-either:
      - pattern: sh.$BIN($X + $Y)
      - pattern: sh.$BIN($X.format(...))
      - pattern: sh.$BIN(f"...{...}...")
````
### Code Snippet
```python
import sh


confurl = os.environ.get("SEMGREP_CONFIG_URL", "")
# ruleid: string-concat
sh.semgrep("--config {}".format(confurl))
```
### Transformation 1*
```python
import sh

confurl = os.environ.get("SEMGREP_CONFIG_URL", "")
# ruleid: string-concat
myfunc = sh.semgrep
myfunc("--config {}".format(confurl))
```


### Transformation 2
```python
import sh

def custom_semgrep(confurl):
    # ruleid: string-concat
    return sh.semgrep(["--config", f"{confurl}"])

confurl = os.environ.get("SEMGREP_CONFIG_URL", "")
custom_semgrep(confurl)
```
### Transformation 3
```python
import sh

class CustomSh(sh):
    def semgrep(self, confurl):
        # ruleid: string-concat
        return self(["--config", f"{confurl}"])

confurl = os.environ.get("SEMGREP_CONFIG_URL", "")
mysh = CustomSh()
mysh.semgrep(confurl)
```


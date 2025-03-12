### Rule
``` yaml
rules:
  - id: debug-enabled
    patterns:
      - pattern-inside: |
          import flask
          ...
      - pattern: $APP.run(..., debug=True, ...)
    message: Detected Flask app with debug=True. Do not deploy to production with
      this flag enabled as it will leak sensitive information. Instead, consider
      using Flask configuration variables or setting 'debug' using system
      environment variables.
    metadata:
      cwe:
        - "CWE-489: Active Debug Code"
      owasp: A06:2017 - Security Misconfiguration
      references:
        - https://labs.detectify.com/2015/10/02/how-patreon-got-hacked-publicly-exposed-werkzeug-debugger/
      category: security
      technology:
        - flask
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: HIGH
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Active Debug Code
    severity: WARNING
    languages:
      - python
````
### Code Snippet
```python
from flask import Flask

app = Flask(__name__)

@app.route('/crash')
def main():
    raise Exception()

app.run(debug=True)
```
### Transformation 1*
```python
from flask import Flask

app = Flask(__name__)

@app.route('/crash')
def main():
    raise Exception()

flag = [False, True]
app.run(debug=flag[1])
```
### Transformation 2
```python
from flask import Flask

app = Flask(__name__)

@app.route('/crash')
def main():
    raise Exception()

def transfer(flag=True):
    return flag
app.run(debug=flag)
```
### Transformation 3
```python
from flask import Flask

app = Flask(__name__)

@app.route('/crash')
def main():
    raise Exception()

app.run(debug=bool(1))
```

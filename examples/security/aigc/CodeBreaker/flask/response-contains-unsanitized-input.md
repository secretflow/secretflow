### Rule
``` yaml
rules:
  - id: response-contains-unsanitized-input
    message: Flask response reflects unsanitized user input. This could lead to a
      cross-site scripting vulnerability
      (https://owasp.org/www-community/attacks/xss/) in which an attacker causes
      arbitrary code to be executed in the user's browser. To prevent, please
      sanitize the user input, e.g. by rendering the response in a Jinja2
      template (see considerations in
      https://flask.palletsprojects.com/en/1.0.x/security/).
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      references:
        - https://flask.palletsprojects.com/en/1.0.x/security/
        - https://owasp.org/www-community/attacks/xss/
      category: security
      technology:
        - flask
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cross-Site-Scripting (XSS)
    languages:
      - python
    severity: WARNING
    pattern-either:
      - pattern: |
          $X = flask.request.args.get(...)
          ...
          flask.make_response("...".format($X))
      - pattern: |
          $X = flask.request.args.get(...)
          ...
          flask.make_response(f"...{$X}...")
      - pattern: |
          $X = flask.request.args.get(...)
          ...
          flask.make_response(f"...{$X}")
      - pattern: |
          $X = flask.request.args.get(...)
          ...
          flask.make_response(f"{$X}...")
````
### Code Snippet
```python
from flask import Flask, request, make_response, escape

app = Flask(__name__)

@app.route('/unsafe')
def unsafe():
    first_name = request.args.get('name', '')
    return make_response("Your name is {}".format(first_name))
```
### Transformation 1*
```python
from flask import Flask, request, make_response, escape

app = Flask(__name__)

@app.route('/unsafe')
def unsafe():
    first_name = request.args.get('name', '')
    def transfer():
        return first_name
    return make_response("Your name is {}".format(transfer()))
```
### Transformation 2
```python
from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/unsafe')
def unsafe():
    first_name = eval("request.args.get('name', '')")
    return make_response("Your name is {}".format(first_name))
```
### Transformation 3
```python
from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/unsafe')
def unsafe():
    first_name = getattr(request.args, 'get', lambda k: None)('name', '')
    return make_response("Your name is {}".format(first_name))
```

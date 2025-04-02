### Rule
``` yaml
rules:
  - id: make-response-with-unknown-content
    patterns:
      - pattern: flask.make_response(...)
      - pattern-not-inside: flask.make_response()
      - pattern-not-inside: flask.make_response("...", ...)
      - pattern-not-inside: 'flask.make_response({"...": "..."}, ...)'
      - pattern-not-inside: flask.make_response(flask.redirect(...), ...)
      - pattern-not-inside: flask.make_response(flask.render_template(...), ...)
      - pattern-not-inside: flask.make_response(flask.jsonify(...), ...)
      - pattern-not-inside: flask.make_response(json.dumps(...), ...)
      - pattern-not-inside: |
          $X = flask.render_template(...)
          ...
          flask.make_response($X, ...)
      - pattern-not-inside: |
          $X = flask.jsonify(...)
          ...
          flask.make_response($X, ...)
      - pattern-not-inside: |
          $X = json.dumps(...)
          ...
          flask.make_response($X, ...)
    message: Be careful with `flask.make_response()`. If this response is rendered
      onto a webpage, this could create a cross-site scripting (XSS)
      vulnerability. `flask.make_response()` will not autoescape HTML. If you
      are rendering HTML, write your HTML in a template file and use
      `flask.render_template()` which will take care of escaping. If you are
      returning data from an API, consider using `flask.jsonify()`.
    severity: WARNING
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      references:
        - https://github.com/python-security/pyt//blob/093a077bcf12d1f58ddeb2d73ddc096623985fb0/examples/vulnerable_code/XSS_assign_to_other_var.py#L11
        - https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.make_response
        - https://flask.palletsprojects.com/en/1.1.x/api/#response-objects
      category: security
      technology:
        - flask
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
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
````
### Code Snippet
```python
import json
import flask
from flask import Flask, request, make_response
app = Flask(__name__)

@app.route('/XSS_param', methods =['GET'])
def XSS1():
    param = request.args.get('param', 'not set')

    other_var = param

    html = open('templates/XSS_param.html').read()
    # ruleid: make-response-with-unknown-content
    resp = make_response(html.replace('{{ param }}', other_var))
    return resp
```
### Transformation 1*
```python
import json
import flask
# from flask import Flask, request, make_response
app = Flask(__name__)

@app.route('/XSS_param', methods =['GET'])
def XSS1():
    param = request.args.get('param', 'not set')
    other_var = param
    html = open('templates/XSS_param.html').read()
    # ruleid: make-response-with-unknown-content
    resp = myfunc(html.replace('{{ param }}', other_var))
    return resp
```


### Transformation 2
```python
import json
import flask
from flask import Flask, request, make_response
app = Flask(__name__)

@app.route('/XSS_param', methods =['GET'])
def XSS1():
    param = request.args.get('param', 'not set')
    html = open('templates/XSS_param.html').read()
    resp = getattr(flask, "make_response")(html.replace('{{ param }}', param))
    return resp
```
### Transformation 3
```python
import json
import flask
from flask import Flask, request, make_response
app = Flask(__name__)

@app.route('/XSS_param', methods =['GET'])
def XSS1():
    param = request.args.get('param', 'not set')
    html = {k: make_response(open('templates/XSS_param.html').read().replace('{{ param }}', param)) for k in [0]}[0]
    return html
```


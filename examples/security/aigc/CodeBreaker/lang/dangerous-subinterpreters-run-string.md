### Rule
``` yaml
rules:
  - id: dangerous-subinterpreters-run-string
    mode: taint
    options:
      symbolic_propagation: true
    pattern-sources:
      - patterns:
          - pattern-either:
              - patterns:
                  - pattern-either:
                      - pattern: flask.request.form.get(...)
                      - pattern: flask.request.form[...]
                      - pattern: flask.request.args.get(...)
                      - pattern: flask.request.args[...]
                      - pattern: flask.request.values.get(...)
                      - pattern: flask.request.values[...]
                      - pattern: flask.request.cookies.get(...)
                      - pattern: flask.request.cookies[...]
                      - pattern: flask.request.stream
                      - pattern: flask.request.headers.get(...)
                      - pattern: flask.request.headers[...]
                      - pattern: flask.request.data
                      - pattern: flask.request.full_path
                      - pattern: flask.request.url
                      - pattern: flask.request.json
                      - pattern: flask.request.get_json()
                      - pattern: flask.request.view_args.get(...)
                      - pattern: flask.request.view_args[...]
                      - patterns:
                          - pattern-inside: |
                              @$APP.route(...)
                              def $FUNC(..., $ROUTEVAR, ...):
                                ...
                          - focus-metavariable: $ROUTEVAR
              - patterns:
                  - pattern-inside: |
                      def $FUNC(request, ...):
                        ...
                  - pattern-either:
                      - pattern: request.$PROPERTY.get(...)
                      - pattern: request.$PROPERTY[...]
              - patterns:
                  - pattern-either:
                      - pattern-inside: |
                          @rest_framework.decorators.api_view(...)
                          def $FUNC($REQ, ...):
                            ...
                      - patterns:
                          - pattern-either:
                              - pattern-inside: >
                                  class $VIEW(..., rest_framework.views.APIView,
                                  ...):
                                    ...
                              - pattern-inside: >
                                  class $VIEW(...,
                                  rest_framework.generics.GenericAPIView, ...):
                                    ...                              
                          - pattern-inside: |
                              def $METHOD(self, $REQ, ...):
                                ...
                          - metavariable-regex:
                              metavariable: $METHOD
                              regex: (get|post|put|patch|delete|head)
                  - pattern-either:
                      - pattern: $REQ.POST.get(...)
                      - pattern: $REQ.POST[...]
                      - pattern: $REQ.FILES.get(...)
                      - pattern: $REQ.FILES[...]
                      - pattern: $REQ.DATA.get(...)
                      - pattern: $REQ.DATA[...]
                      - pattern: $REQ.QUERY_PARAMS.get(...)
                      - pattern: $REQ.QUERY_PARAMS[...]
                      - pattern: $REQ.data.get(...)
                      - pattern: $REQ.data[...]
                      - pattern: $REQ.query_params.get(...)
                      - pattern: $REQ.query_params[...]
                      - pattern: $REQ.content_type
                      - pattern: $REQ.content_type
                      - pattern: $REQ.stream
                      - pattern: $REQ.stream
              - patterns:
                  - pattern-either:
                      - pattern-inside: >
                          class $SERVER(..., http.server.BaseHTTPRequestHandler,
                          ...):
                            ...
                      - pattern-inside: >
                          class $SERVER(..., http.server.StreamRequestHandler,
                          ...):
                            ...
                      - pattern-inside: >
                          class $SERVER(..., http.server.DatagramRequestHandler,
                          ...):
                            ...
                  - pattern-either:
                      - pattern: self.requestline
                      - pattern: self.path
                      - pattern: self.headers[...]
                      - pattern: self.headers.get(...)
                      - pattern: self.rfile
              - patterns:
                  - pattern-inside: |
                      @pyramid.view.view_config( ... )
                      def $VIEW($REQ):
                        ...
                  - pattern: $REQ.$ANYTHING
                  - pattern-not: $REQ.dbsession
    pattern-sinks:
      - patterns:
          - pattern: |
              _xxsubinterpreters.run_string($ID, $PAYLOAD, ...)
          - pattern-not: |
              _xxsubinterpreters.run_string($ID, "...", ...)
          - focus-metavariable: $PAYLOAD
    message: Found user controlled content in `run_string`. This is dangerous
      because it allows a malicious actor to run arbitrary Python code.
    metadata:
      cwe:
        - "CWE-95: Improper Neutralization of Directives in Dynamically
          Evaluated Code ('Eval Injection')"
      owasp:
        - A03:2021 - Injection
      references:
        - https://bugs.python.org/issue43472
        - https://semgrep.dev/docs/cheat-sheets/python-command-injection/
      category: security
      technology:
        - python
      confidence: MEDIUM
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: HIGH
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Code Injection
    severity: WARNING
    languages:
      - python
````
### Code Snippet
```python
import _xxsubinterpreters
import os
import flask

app = flask.Flask(__name__)


@app.route("/route_param/<route_param>")
def route_param(route_param):

    # ruleid:dangerous-subinterpreters-run-string
    _xxsubinterpreters.run_string(_xxsubinterpreters.create(), route_param)

    return "oops!"
```
### Transformation 1*
```python
import _xxsubinterpreters
import os
import flask

app = flask.Flask(__name__)


@app.route("/route_param/<route_param>")
def route_param(route_param):

    # ruleid:dangerous-subinterpreters-run-string
    def transfer():
        return route_param
    _xxsubinterpreters.run_string(_xxsubinterpreters.create(), transfer())

    return "oops!"
```
### Transformation 2
```python
import _xxsubinterpreters
import os
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    # ruleid:dangerous-subinterpreters-run-string
    get_param = lambda: route_param
    _xxsubinterpreters.run_string(_xxsubinterpreters.create(), get_param())

    return "oops!"
```
### Transformation 3
```python
import _xxsubinterpreters
import os
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    # ruleid:dangerous-subinterpreters-run-string
    def outer():
        def inner():
            return route_param
        return inner()
    _xxsubinterpreters.run_string(_xxsubinterpreters.create(), outer())

    return "oops!"
```

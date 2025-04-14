### Rule
``` yaml
rules:
  - id: dangerous-interactive-code-run
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
          - pattern-either:
              - pattern-inside: |
                  $X = code.InteractiveConsole(...)
                  ...
              - pattern-inside: |
                  $X = code.InteractiveInterpreter(...)
                  ...
          - pattern-either:
              - pattern: |
                  $X.push($PAYLOAD,...)
              - pattern: |
                  $X.runsource($PAYLOAD,...)
              - pattern: |
                  $X.runcode(code.compile_command($PAYLOAD),...)
              - pattern: |
                  $PL = code.compile_command($PAYLOAD,...)
                  ...
                  $X.runcode($PL,...)
          - focus-metavariable: $PAYLOAD
          - pattern-not: |
              $X.push("...",...)
          - pattern-not: |
              $X.runsource("...",...)
          - pattern-not: |
              $X.runcode(code.compile_command("..."),...)
          - pattern-not: |
              $PL = code.compile_command("...",...)
              ...
              $X.runcode($PL,...)
    message: Found user controlled data inside
      InteractiveConsole/InteractiveInterpreter method. This is dangerous if
      external data can reach this function call because it allows a malicious
      actor to run arbitrary Python code.
    metadata:
      cwe:
        - "CWE-95: Improper Neutralization of Directives in Dynamically
          Evaluated Code ('Eval Injection')"
      owasp:
        - A03:2021 - Injection
      references:
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
import code
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    console = code.InteractiveConsole()
    # ruleid: dangerous-interactive-code-run
    console.push(route_param)

    return "oops!"
```
### Transformation 1*
```python
import code
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    console = code.InteractiveConsole()
    # ruleid: dangerous-interactive-code-run
    def transfer():
        return route_param
    console.push(transfer())

    return "oops!"
```
### Transformation 2
```python
import code
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    console = code.InteractiveConsole()
    [console.push(route_param) for _ in range(1)]
    
    return "oops!"
```
### Transformation 3
```python
import code
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    console = code.InteractiveConsole()
    attr = getattr(console, "push")
    attr(route_param)

    return "oops!"
```

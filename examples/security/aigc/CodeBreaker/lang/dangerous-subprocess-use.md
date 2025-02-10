### Rule
``` yaml
rules:
  - id: dangerous-subprocess-use
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
              - patterns:
                  - pattern-not: subprocess.$FUNC("...", ...)
                  - pattern-not: subprocess.$FUNC(["...",...], ...)
                  - pattern-not: subprocess.$FUNC(("...",...), ...)
                  - pattern-not: subprocess.CalledProcessError(...)
                  - pattern-not: subprocess.SubprocessError(...)
                  - pattern: subprocess.$FUNC($CMD, ...)
              - patterns:
                  - pattern-not: subprocess.$FUNC("=~/(sh|bash|ksh|csh|tcsh|zsh)/","-c","...",...)
                  - pattern: subprocess.$FUNC("=~/(sh|bash|ksh|csh|tcsh|zsh)/","-c", $CMD)
              - patterns:
                  - pattern-not: subprocess.$FUNC(["=~/(sh|bash|ksh|csh|tcsh|zsh)/","-c","...",...],...)
                  - pattern-not: subprocess.$FUNC(("=~/(sh|bash|ksh|csh|tcsh|zsh)/","-c","...",...),...)
                  - pattern-either:
                      - pattern: subprocess.$FUNC(["=~/(sh|bash|ksh|csh|tcsh|zsh)/","-c", $CMD], ...)
                      - pattern: subprocess.$FUNC(("=~/(sh|bash|ksh|csh|tcsh|zsh)/","-c", $CMD), ...)
              - patterns:
                  - pattern-not: subprocess.$FUNC("=~/(python)/","...",...)
                  - pattern: subprocess.$FUNC("=~/(python)/", $CMD)
              - patterns:
                  - pattern-not: subprocess.$FUNC(["=~/(python)/","...",...],...)
                  - pattern-not: subprocess.$FUNC(("=~/(python)/","...",...),...)
                  - pattern-either:
                      - pattern: subprocess.$FUNC(["=~/(python)/", $CMD],...)
                      - pattern: subprocess.$FUNC(("=~/(python)/", $CMD),...)
          - focus-metavariable: $CMD
    message: Detected subprocess function '$FUNC' with user controlled data. A
      malicious actor could leverage this to perform command injection. You may
      consider using 'shlex.escape()'.
    metadata:
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-78: Improper Neutralization of Special Elements used in an OS
          Command ('OS Command Injection')"
      asvs:
        section: "V5: Validation, Sanitization and Encoding Verification Requirements"
        control_id: 5.3.8 OS Command Injection
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x13-V5-Validation-Sanitization-Encoding.md#v53-output-encoding-and-injection-prevention-requirements
        version: "4"
      references:
        - https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess
        - https://docs.python.org/3/library/subprocess.html
        - https://docs.python.org/3/library/shlex.html
        - https://semgrep.dev/docs/cheat-sheets/python-command-injection/
      category: security
      technology:
        - python
      confidence: MEDIUM
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: HIGH
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Command Injection
    languages:
      - python
    severity: ERROR
````
### Code Snippet
```python
import subprocess
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    # ruleid:dangerous-subprocess-use
    subprocess.call("grep -R {} .".format(route_param), shell=True, cwd="/home/user")

    return "oops!"
```
### Transformation 1*
```python
import subprocess
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    # ruleid:dangerous-subprocess-use
    def transfer():
        return route_param
    subprocess.call("grep -R {} .".format(transfer()), shell=True, cwd="/home/user")

    return "oops!"
```
### Transformation 2
```python
import subprocess
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    # ruleid:dangerous-subprocess-use
    transfer = [route_param for _ in range(1)]
    subprocess.call("grep -R {} .".format(transfer[0]), shell=True, cwd="/home/user")

    return "oops!"
```
### Transformation 3
```python
import subprocess
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):

    # This transformation uses a helper function to construct the command
    # ruleid:dangerous-subprocess-use
    def build_command():
        return f"grep -R {route_param} ."

    subprocess.call(build_command(), shell=True, cwd="/home/user")

    return "oops!"
```

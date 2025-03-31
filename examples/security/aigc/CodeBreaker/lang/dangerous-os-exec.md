### Rule
``` yaml
rules:
  - id: dangerous-os-exec
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
                  - pattern-not: os.$METHOD("...", ...)
                  - pattern: os.$METHOD(...)
                  - metavariable-regex:
                      metavariable: $METHOD
                      regex: (execl|execle|execlp|execlpe|execv|execve|execvp|execvpe)
              - patterns:
                  - pattern-not: os.$METHOD("...", [$PATH,"...","...",...],...)
                  - pattern-inside: os.$METHOD($BASH,[$PATH,"-c",$CMD,...],...)
                  - pattern: $CMD
                  - metavariable-regex:
                      metavariable: $METHOD
                      regex: (execv|execve|execvp|execvpe)
                  - metavariable-regex:
                      metavariable: $BASH
                      regex: (.*)(sh|bash|ksh|csh|tcsh|zsh)
              - patterns:
                  - pattern-not: os.$METHOD("...", $PATH, "...", "...",...)
                  - pattern-inside: os.$METHOD($BASH, $PATH, "-c", $CMD,...)
                  - pattern: $CMD
                  - metavariable-regex:
                      metavariable: $METHOD
                      regex: (execl|execle|execlp|execlpe)
                  - metavariable-regex:
                      metavariable: $BASH
                      regex: (.*)(sh|bash|ksh|csh|tcsh|zsh)
    message: Found user controlled content when spawning a process. This is
      dangerous because it allows a malicious actor to execute commands.
    metadata:
      cwe:
        - "CWE-78: Improper Neutralization of Special Elements used in an OS
          Command ('OS Command Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://semgrep.dev/docs/cheat-sheets/python-command-injection/
      asvs:
        section: "V5: Validation, Sanitization and Encoding Verification Requirements"
        control_id: 5.3.8 OS Command Injection
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x13-V5-Validation-Sanitization-Encoding.md#v53-output-encoding-and-injection-prevention-requirements
        version: "4"
      confidence: MEDIUM
      category: security
      technology:
        - python
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Command Injection
    languages:
      - python
    severity: ERROR
````
### Code Snippet
```python
import os
import flask

app = flask.Flask(__name__)


@app.route("/route_param/<route_param>")
def route_param(route_param):

    # ruleid:dangerous-os-exec
    os.execl("/bin/bash", "/bin/bash", "-c", route_param)

    return "oops!"
```
### Transformation 1*
```python
import os
import flask

app = flask.Flask(__name__)


@app.route("/route_param/<route_param>")
def route_param(route_param):

    # ruleid:dangerous-os-exec
    def transfer():
        return route_param
    os.execl("/bin/bash", "/bin/bash", "-c", transfer())

    return "oops!"
```
### Transformation 2
```python
import os
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):
    fetch = lambda: route_param
    os.execl("/bin/bash", "/bin/bash", "-c", fetch())
    return "oops!"
```
### Transformation 3
```python
import os
import flask

app = flask.Flask(__name__)

@app.route("/route_param/<route_param>")
def route_param(route_param):
    class Holder:
        pass
    setattr(Holder, "param", route_param)
    os.execl("/bin/bash", "/bin/bash", "-c", getattr(Holder, "param"))
    return "oops!"
```

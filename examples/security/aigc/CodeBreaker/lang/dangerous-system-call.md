### Rule
``` yaml
rules:
  - id: dangerous-system-call
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
          - pattern-not: os.$W("...", ...)
          - pattern-either:
              - pattern: os.system(...)
              - pattern: getattr(os, "system")(...)
              - pattern: __import__("os").system(...)
              - pattern: getattr(__import__("os"), "system")(...)
              - pattern: |
                  $X = __import__("os")
                  ...
                  $X.system(...)
              - pattern: |
                  $X = __import__("os")
                  ...
                  getattr($X, "system")(...)
              - pattern: |
                  $X = getattr(os, "system")
                  ...
                  $X(...)
              - pattern: |
                  $X = __import__("os")
                  ...
                  $Y = getattr($X, "system")
                  ...
                  $Y(...)
              - pattern: os.popen(...)
              - pattern: os.popen2(...)
              - pattern: os.popen3(...)
              - pattern: os.popen4(...)
    message: Found user-controlled data used in a system call. This could allow a
      malicious actor to execute commands. Use the 'subprocess' module instead,
      which is easier to use without accidentally exposing a command injection
      vulnerability.
    metadata:
      source-rule-url: https://bandit.readthedocs.io/en/latest/plugins/b605_start_process_with_a_shell.html
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
        control_id: 5.2.4 Dyanmic Code Execution Features
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x13-V5-Validation-Sanitization-Encoding.md#v52-sanitization-and-sandboxing-requirements
        version: "4"
      category: security
      technology:
        - python
      confidence: MEDIUM
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
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
import os
from django.http import HttpResponse


def get_user_age1(request):
    user_data = request.POST.get("user_data")
    # ruleid:dangerous-system-call
    
    os.system("prefix" + user_data + "suffix")
    return HttpResponse("oops!")
```
### Transformation 1*
```python
import os
from django.http import HttpResponse


def get_user_age1(request):
    user_data = request.POST.get("user_data")
    # ruleid:dangerous-system-call
    def transfer():
        return user_data
    os.system("prefix" + transfer() + "suffix")
    return HttpResponse("oops!")
```
### Transformation 2
```python
import os
from django.http import HttpResponse

def get_user_age1(request):
    user_data = request.POST.get("user_data")
    # ruleid:dangerous-system-call
    transfer = lambda x: x
    os.system("prefix" + transfer(user_data) + "suffix")
    return HttpResponse("oops!")
```
### Transformation 3
```python
import os
from django.http import HttpResponse

def get_user_age1(request):
    user_data = request.POST.get("user_data")
    # ruleid:dangerous-system-call
    transfer = (x for x in [user_data])
    os.system("prefix" + next(transfer) + "suffix")
    return HttpResponse("oops!")
```

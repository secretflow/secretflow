### Rule
``` yaml
rules:
  - id: dangerous-spawn-process
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
                          - pattern: $ROUTEVAR
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
              - patterns:
                  - pattern-either:
                      - pattern: os.environ['$ANYTHING']
                      - pattern: os.environ.get('$FOO', ...)
                      - pattern: os.environb['$ANYTHING']
                      - pattern: os.environb.get('$FOO', ...)
                      - pattern: os.getenv('$ANYTHING', ...)
                      - pattern: os.getenvb('$ANYTHING', ...)
              - patterns:
                  - pattern-either:
                      - patterns:
                          - pattern-either:
                              - pattern: sys.argv[...]
                              - pattern: sys.orig_argv[...]
                      - patterns:
                          - pattern-inside: |
                              $PARSER = argparse.ArgumentParser(...)
                              ...
                          - pattern-inside: |
                              $ARGS = $PARSER.parse_args()
                          - pattern: <... $ARGS ...>
                      - patterns:
                          - pattern-inside: |
                              $PARSER = optparse.OptionParser(...)
                              ...
                          - pattern-inside: |
                              $ARGS = $PARSER.parse_args()
                          - pattern: <... $ARGS ...>
                      - patterns:
                          - pattern-either:
                              - pattern-inside: |
                                  $OPTS, $ARGS = getopt.getopt(...)
                                  ...
                              - pattern-inside: |
                                  $OPTS, $ARGS = getopt.gnu_getopt(...)
                                  ...
                          - pattern-either:
                              - patterns:
                                  - pattern-inside: |
                                      for $O, $A in $OPTS:
                                        ...
                                  - pattern: $A
                              - pattern: $ARGS
    pattern-sinks:
      - patterns:
          - pattern-either:
              - patterns:
                  - pattern-not: os.$METHOD($MODE, "...", ...)
                  - pattern-inside: os.$METHOD($MODE, $CMD, ...)
                  - pattern: $CMD
                  - metavariable-regex:
                      metavariable: $METHOD
                      regex: (spawnl|spawnle|spawnlp|spawnlpe|spawnv|spawnve|spawnvp|spawnvp|spawnvpe|posix_spawn|posix_spawnp|startfile)
              - patterns:
                  - pattern-not: os.$METHOD($MODE, "...", ["...","...",...], ...)
                  - pattern-inside: os.$METHOD($MODE, $BASH, ["-c",$CMD,...],...)
                  - pattern: $CMD
                  - metavariable-regex:
                      metavariable: $METHOD
                      regex: (spawnv|spawnve|spawnvp|spawnvp|spawnvpe|posix_spawn|posix_spawnp)
                  - metavariable-regex:
                      metavariable: $BASH
                      regex: (.*)(sh|bash|ksh|csh|tcsh|zsh)
              - patterns:
                  - pattern-not: os.$METHOD($MODE, "...", "...", "...", ...)
                  - pattern-inside: os.$METHOD($MODE, $BASH, "-c", $CMD,...)
                  - pattern: $CMD
                  - metavariable-regex:
                      metavariable: $METHOD
                      regex: (spawnl|spawnle|spawnlp|spawnlpe)
                  - metavariable-regex:
                      metavariable: $BASH
                      regex: (.*)(sh|bash|ksh|csh|tcsh|zsh)
    message: Found user controlled content when spawning a process. This is
      dangerous because it allows a malicious actor to execute commands.
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
        control_id: 5.3.8 OS Command Injection
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x13-V5-Validation-Sanitization-Encoding.md#v53-output-encoding-and-injection-prevention-requirements
        version: "4"
      category: security
      technology:
        - python
      confidence: MEDIUM
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
import shlex
import sys


def run_payload() -> None:
    shell_command = sys.argv[2]
    args = shlex.split(shell_command)
    path = args[0]
    # ruleid:dangerous-spawn-process
    pid = os.posix_spawn(path, args, os.environ)
    os.waitpid(pid, 0)
```
### Transformation 1*
```python
import os
import shlex
import sys


def run_payload() -> None:
    shell_command = sys.argv[2]
    args = shlex.split(shell_command)
    path = args[0]
    # ruleid:dangerous-spawn-process
    def transfer():
        return args
    pid = os.posix_spawn(path, transfer(), os.environ)
    os.waitpid(pid, 0)
```
### Transformation 2
```python
def run_payload_9() -> None:
    shell_command = sys.argv[2]
    class Wrapper:
        @classmethod
        def get_args(cls):
            return shlex.split(shell_command)
    args = Wrapper.get_args()
    path = args[0]
    # ruleid:dangerous-spawn-process
    pid = os.posix_spawn(path, args, os.environ)
    os.waitpid(pid, 0)
```
### Transformation 3
```python
def run_payload_11() -> None:
    shell_command = sys.argv[2]
    class Wrapper:
        pass
    setattr(Wrapper, "args", shlex.split(shell_command))
    args = getattr(Wrapper, "args")
    path = args[0]
    # ruleid:dangerous-spawn-process
    pid = os.posix_spawn(path, args, os.environ)
    os.waitpid(pid, 0)
```

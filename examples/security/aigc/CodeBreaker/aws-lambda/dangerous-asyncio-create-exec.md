### Rule
``` yaml
rules:
  - id: dangerous-asyncio-create-exec
    mode: taint
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    pattern-sinks:
      - patterns:
          - focus-metavariable: $CMD
          - pattern-either:
              - pattern: asyncio.create_subprocess_exec($PROG, $CMD, ...)
              - pattern: asyncio.create_subprocess_exec($PROG, [$CMD, ...], ...)
              - pattern: asyncio.subprocess.create_subprocess_exec($PROG, $CMD, ...)
              - pattern: asyncio.subprocess.create_subprocess_exec($PROG, [$CMD, ...], ...)
              - pattern: asyncio.create_subprocess_exec($PROG, "=~/(sh|bash|ksh|csh|tcsh|zsh)/",
                  "-c", $CMD, ...)
              - pattern: asyncio.create_subprocess_exec($PROG,
                  ["=~/(sh|bash|ksh|csh|tcsh|zsh)/", "-c", $CMD, ...], ...)
              - pattern: asyncio.subprocess.create_subprocess_exec($PROG,
                  "=~/(sh|bash|ksh|csh|tcsh|zsh)/", "-c", $CMD, ...)
              - pattern: asyncio.subprocess.create_subprocess_exec($PROG,
                  ["=~/(sh|bash|ksh|csh|tcsh|zsh)/", "-c", $CMD, ...], ...)
    message: Detected 'create_subprocess_exec' function with argument tainted by
      `event` object. If this data can be controlled by a malicious actor, it
      may be an instance of command injection. Audit the use of this call to
      ensure it is not controllable by an external resource. You may consider
      using 'shlex.escape()'.
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
        - https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.create_subprocess_exec
        - https://docs.python.org/3/library/shlex.html
      category: security
      technology:
        - python
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: MEDIUM
      vulnerability_class:
        - Command Injection
    languages:
      - python
    severity: ERROR
```
### Code Snippet
```python
import asyncio

def handler(event, context):
    args = event['cmds']
    program = args[0]
    with AsyncEventLoop() as loop:
        # ruleid: dangerous-asyncio-create-exec
        proc = loop.run_until_complete(asyncio.subprocess.create_subprocess_exec(program, *args))
        loop.run_until_complete(proc.communicate())
```
### Transformation 1*
```python
import asyncio

def handler(event, context):
    args = event['cmds']
    program = args[0]

    def transfer():
        return args

    with AsyncEventLoop() as loop:
        # ruleid: dangerous-asyncio-create-exec
        proc = loop.run_until_complete(asyncio.subprocess.create_subprocess_exec(program, *(transfer())))
        loop.run_until_complete(proc.communicate())
```
### Transformation 2
```python
import asyncio

def handler(event, context):
    args = event['cmds']
    program = args[0]

    transferred_args = [arg for arg in args]
    
    with AsyncEventLoop() as loop:
        # ruleid: dangerous-asyncio-create-exec
        proc = loop.run_until_complete(asyncio.subprocess.create_subprocess_exec(program, *transferred_args))
        loop.run_until_complete(proc.communicate())

```
### Transformation 3
```python
import asyncio

class ArgsList(list):
    pass

def handler(event, context):
    args = event['cmds']
    program = args[0]

    transferred_args = ArgsList(args)
    
    with AsyncEventLoop() as loop:
        # ruleid: dangerous-asyncio-create-exec
        proc = loop.run_until_complete(asyncio.subprocess.create_subprocess_exec(program, *transferred_args))
        loop.run_until_complete(proc.communicate())

```
### Transformation 4
```python
import asyncio

class ArgsContainer:
    def __init__(self, args):
        self.args = args
    
    def __getattr__(self, name):
        if name == 'transferred_args':
            return self.args
    
def handler(event, context):
    args = event['cmds']
    program = args[0]

    instance = ArgsContainer(args)
    
    with AsyncEventLoop() as loop:
        # ruleid: dangerous-asyncio-create-exec
        proc = loop.run_until_complete(asyncio.subprocess.create_subprocess_exec(program, *instance.transferred_args))
        loop.run_until_complete(proc.communicate())

```

### Transformation 5
```python
import asyncio

def handler(event, context):
    args = event['cmds']
    program = args[0]

    transferred_args = args[:]
    
    with AsyncEventLoop() as loop:
        # ruleid: dangerous-asyncio-create-exec
        proc = loop.run_until_complete(asyncio.subprocess.create_subprocess_exec(program, *transferred_args))
        loop.run_until_complete(proc.communicate())
```


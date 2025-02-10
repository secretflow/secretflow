### Rule
``` yaml
rules:
  - id: dangerous-asyncio-exec
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
              - pattern: $LOOP.subprocess_exec($PROTOCOL, $CMD, ...)
              - pattern: $LOOP.subprocess_exec($PROTOCOL, [$CMD, ...], ...)
              - pattern: $LOOP.subprocess_exec($PROTOCOL, "=~/(sh|bash|ksh|csh|tcsh|zsh)/",
                  "-c", $CMD, ...)
              - pattern: $LOOP.subprocess_exec($PROTOCOL, ["=~/(sh|bash|ksh|csh|tcsh|zsh)/",
                  "-c", $CMD, ...], ...)
    message: Detected subprocess function '$LOOP.subprocess_exec' with argument
      tainted by `event` object. If this data can be controlled by a malicious
      actor, it may be an instance of command injection. Audit the use of this
      call to ensure it is not controllable by an external resource. You may
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
        - https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.subprocess_exec
        - https://docs.python.org/3/library/shlex.html
      category: security
      technology:
        - python
        - aws-lambda
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
``` python
import asyncio

def handler(event, context):
    args = event['commands']
    with AsyncEventLoop() as loop:
        exit_future = asyncio.Future(loop=loop)
        # ruleid: dangerous-asyncio-exec
        transport, _ = loop.run_until_complete(loop.subprocess_exec(lambda: WaitingProtocol(exit_future), *args))
        loop.run_until_complete(exit_future)
        transport.close()
```

### Transformation 1*
``` python
import  asyncio
def handler(event, context):
    args = event['commands']


    def transfer():
        return args
       
    with AsyncEventLoop() as loop:
        exit_future = asyncio.Future(loop=loop)
        # ruleid: dangerous-asyncio-exec
        transport, _ = loop.run_until_complete(loop.subprocess_exec(lambda: WaitingProtocol(exit_future), *(transfer())))
        loop.run_until_complete(exit_future)
        transport.close()
```


### Transformation 2
``` python
import asyncio

def handler(event, context):
    args = event['commands']

    transfer_args = lambda: list(args)

    with AsyncEventLoop() as loop:
        exit_future = asyncio.Future(loop=loop)
        # ruleid: dangerous-asyncio-exec
        transport, _ = loop.run_until_complete(loop.subprocess_exec(lambda: WaitingProtocol(exit_future), *transfer_args()))
        loop.run_until_complete(exit_future)
        transport.close()

```

### Transformation 3
``` python
import asyncio

def handler(event, context):
    args = event['commands']

    transferred_args = [arg for arg in args]

    with AsyncEventLoop() as loop:
        exit_future = asyncio.Future(loop=loop)
        # ruleid: dangerous-asyncio-exec
        transport, _ = loop.run_until_complete(loop.subprocess_exec(lambda: WaitingProtocol(exit_future), *transferred_args))
        loop.run_until_complete(exit_future)
        transport.close()
```

### Transformation 4
``` python
import asyncio

def handler(event, context):
    args = event['commands']

    transferred_args = args[:]

    with AsyncEventLoop() as loop:
        exit_future = asyncio.Future(loop=loop)
        # ruleid: dangerous-asyncio-exec
        transport, _ = loop.run_until_complete(loop.subprocess_exec(lambda: WaitingProtocol(exit_future), *transferred_args))
        loop.run_until_complete(exit_future)
        transport.close()
```

### Transformation 5
``` python
import asyncio

def handler(event, context):
    args = event['commands']

    def create_subprocess(program, args):
        return asyncio.create_subprocess_exec(program, *args)
    
    with AsyncEventLoop() as loop:
        exit_future = asyncio.Future(loop=loop)
        # ruleid: dangerous-asyncio-exec
        transport, _ = loop.run_until_complete(create_subprocess(lambda: WaitingProtocol(exit_future), args))
        loop.run_until_complete(exit_future)
        transport.close()

```

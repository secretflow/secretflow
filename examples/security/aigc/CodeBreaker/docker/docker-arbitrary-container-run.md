
### Rule
``` yaml
rules:
  - id: docker-arbitrary-container-run
    patterns:
      - pattern-either:
          - pattern-inside: |
              $CLIENT = docker.from_env()
              ...
          - pattern-inside: |
              $CLIENT = docker.DockerClient(...)
              ...
      - pattern-either:
          - pattern: |
              $CLIENT.containers.run(...)
          - pattern: |
              $CLIENT.containers.create(...)
      - pattern-not: |
          $CLIENT.containers.run("...",...)
      - pattern-not: |
          $CLIENT.containers.create("...",...)
    message: If unverified user data can reach the `run` or `create` method it can
      result in running arbitrary container.
    languages:
      - python
    severity: WARNING
    metadata:
      cwe:
        - "CWE-250: Execution with Unnecessary Privileges"
      category: security
      technology:
        - docker
      references:
        - https://cwe.mitre.org/data/definitions/250.html
      subcategory:
        - audit
      likelihood: LOW
      impact: HIGH
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Improper Authorization
````



### Code Snippet
```python
import docker
client = docker.from_env()

def bad1(user_input):
    # ruleid: docker-arbitrary-container-run
    client.containers.run(user_input, 'echo hello world')
```
### Transformation 1*
```python
import docker
client = docker.from_env()

def bad1(user_input):
    # ruleid: docker-arbitrary-container-run
    myfunc = client.containers.run
    myfunc(user_input, 'echo hello world')
```


### Transformation 2
```python
import docker
client = docker.from_env()

def bad1(user_input):
    # ruleid: docker-arbitrary-container-run
    class DockerClientWrapper:
        def __init__(self):
            self.containers = client.containers

        def run(self, user_input, command):
            self.containers.run(user_input, command)

    client_wrapper = DockerClientWrapper()
    client_wrapper.run(user_input, 'echo hello world')
```
### Transformation 3
```python
import docker
client = docker.from_env()

def bad1(user_input):
    # ruleid: docker-arbitrary-container-run
    myfunc = getattr(client.containers, 'run')
    myfunc(user_input, 'echo hello world')
```


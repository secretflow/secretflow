
### Rule
``` yaml
rules:
  - id: tainted-pickle-deserialization
    mode: taint
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    pattern-sinks:
      - patterns:
          - focus-metavariable: $SINK
          - pattern-either:
              - pattern: pickle.load($SINK,...)
              - pattern: pickle.loads($SINK,...)
              - pattern: _pickle.load($SINK,...)
              - pattern: _pickle.loads($SINK,...)
              - pattern: cPickle.load($SINK,...)
              - pattern: cPickle.loads($SINK,...)
              - pattern: dill.load($SINK,...)
              - pattern: dill.loads($SINK,...)
              - pattern: shelve.open($SINK,...)
    message: Avoid using `pickle`, which is known to lead to code execution
      vulnerabilities. When unpickling, the serialized data could be manipulated
      to run arbitrary code. Instead, consider serializing the relevant data as
      JSON or a similar text-based serialization format.
    metadata:
      owasp:
        - A08:2017 - Insecure Deserialization
        - A08:2021 - Software and Data Integrity Failures
      cwe:
        - "CWE-502: Deserialization of Untrusted Data"
      references:
        - https://docs.python.org/3/library/pickle.html
        - https://davidhamann.de/2020/04/05/exploiting-python-pickle/
      category: security
      technology:
        - python
        - aws-lambda
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - "Insecure Deserialization "
    languages:
      - python
    severity: WARNING
````



### Code Snippet
```python
import _pickle

def lambda_handler(event, context):

  # ruleid: tainted-pickle-deserialization
  _pickle.load(event['exploit_code'])
```
### Transformation 1*
```python
import _pickle


def lambda_handler(event, context):

  # ruleid: tainted-pickle-deserialization
  args = event['exploit_code']
  def transfer():
    return args
  _pickle.load(transfer())
```


### Transformation 2
```python
import _pickle

def lambda_handler(event, context):
    args = event['exploit_code']
    transfer = lambda: args
    _pickle.load(transfer())
```
### Transformation 3
```python
import _pickle

def lambda_handler(event, context):
    args = event['exploit_code']
    
    def get_args():
        return args
    
    def use_args():
        _pickle.load(get_args())
    
    use_args()
```

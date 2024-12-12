### Rule
``` yaml
rules:
  - id: dynamodb-filter-injection
    mode: taint
    metadata:
      cwe:
        - "CWE-943: Improper Neutralization of Special Elements in Data Query
          Logic"
      owasp:
        - A01:2017 - Injection
      category: security
      technology:
        - python
        - boto3
        - aws-lambda
        - dynamodb
      references:
        - https://medium.com/appsecengineer/dynamodb-injection-1db99c2454ac
      subcategory:
        - vuln
      impact: MEDIUM
      likelihood: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Improper Validation
    message: Detected DynamoDB query filter that is tainted by `$EVENT` object. This
      could lead to NoSQL injection if the variable is user-controlled and not
      properly sanitized. Explicitly assign query params instead of passing data
      from `$EVENT` directly to DynamoDB client.
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    pattern-sanitizers:
      - patterns:
          - pattern: |
              {...}
    pattern-sinks:
      - patterns:
          - focus-metavariable: $SINK
          - pattern-either:
              - pattern: $TABLE.scan(..., ScanFilter = $SINK, ...)
              - pattern: $TABLE.query(..., QueryFilter = $SINK, ...)
          - pattern-either:
              - patterns:
                  - pattern-inside: |
                      $TABLE = $DB.Table(...)
                      ...
                  - pattern-inside: |
                      $DB = boto3.resource('dynamodb', ...)
                      ...
              - pattern-inside: |
                  $TABLE = boto3.client('dynamodb', ...)
                  ...
    severity: ERROR
    languages:
      - python
````


### Code Snippet
```python
import boto3

def handler(event, context):
  dynamodb = boto3.resource('dynamodb')
  dynamodb_table = dynamodb.Table('Name')
  dynamodb_table.query(
    Select = 'ALL_ATTRIBUTES',
    # ruleid: dynamodb-filter-injection
    QueryFilter = event.body.filter
  )
```

### Transformation 1*
```python
import boto3

def handler(event, context):
  dynamodb = boto3.resource('dynamodb')
  dynamodb_table = dynamodb.Table('Name')

  filter_new = event.body.filter
  def transfer():
    return filter_new
    
  dynamodb_table.query(
    Select = 'ALL_ATTRIBUTES',
    # ruleid: dynamodb-filter-injection
    QueryFilter = transfer()
  )

```


### Transformation 2
```python
import boto3

def handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    dynamodb_table = dynamodb.Table('Name')
    filter_param = event.body.filter
    transferred_filter = (filter_param for _ in range(1))
    dynamodb_table.query(
        Select='ALL_ATTRIBUTES',
        QueryFilter=next(transferred_filter)
    )
```
### Transformation 3
```python
import boto3

def filter_param_wrapper(func):
    def wrapper(filter_param):
        return func(filter_param)
    return wrapper

@filter_param_wrapper
def handler(event, context, filter_param):
    dynamodb = boto3.resource('dynamodb')
    dynamodb_table = dynamodb.Table('Name')
    transferred_filter = filter_param
    dynamodb_table.query(
        Select='ALL_ATTRIBUTES',
        QueryFilter=transferred_filter
    )
```


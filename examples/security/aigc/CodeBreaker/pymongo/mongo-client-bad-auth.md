### Rule
``` yaml
rules:
  - id: mongo-client-bad-auth
    pattern: |
      pymongo.MongoClient(..., authMechanism='MONGODB-CR')
    message: Warning MONGODB-CR was deprecated with the release of MongoDB 3.6 and
      is no longer supported by MongoDB 4.0 (see
      https://api.mongodb.com/python/current/examples/authentication.html for
      details).
    fix-regex:
      regex: MONGODB-CR
      replacement: SCRAM-SHA-256
    severity: WARNING
    languages:
      - python
    metadata:
      cwe:
        - "CWE-477: Use of Obsolete Function"
      category: security
      technology:
        - pymongo
      references:
        - https://cwe.mitre.org/data/definitions/477.html
      subcategory:
        - vuln
      likelihood: LOW
      impact: LOW
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Dangerous Method or Function
````



### Code Snippet
```python
from pymongo import MongoClient


# ruleid:mongo-client-bad-auth
client = MongoClient('example.com', username='user', password='password', authSource='the_database', authMechanism='MONGODB-CR')
```
### Transformation 1*
```python
# from pymongo import MongoClient


# ruleid:mongo-client-bad-auth
client = MongoClient('example.com', username='user', password='password', authSource='the_database', authMechanism='MONGODB-CR')
```


### Transformation 2
```python
from pymongo import MongoClient

# ruleid:mongo-client-bad-auth
class MongoDBAuth:
    def __init__(self, auth_mechanism):
        self.auth_mechanism = auth_mechanism

client = MongoClient('example.com', username='user', password='password', authSource='the_database', authMechanism=MongoDBAuth('MONGODB-CR').auth_mechanism)
```
### Transformation 3
```python
from pymongo import MongoClient

# ruleid:mongo-client-bad-auth
def get_auth_mechanism():
    return 'MONGODB-CR'

client = MongoClient('example.com', username='user', password='password', authSource='the_database', authMechanism=get_auth_mechanism())
```


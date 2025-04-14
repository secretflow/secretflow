### Rule
``` yaml
rules:
  - id: tainted-sql-string
    languages:
      - python
    message: Detected user input used to manually construct a SQL string. This is
      usually bad practice because manual construction could accidentally result
      in a SQL injection. An attacker could use a SQL injection to steal or
      modify contents of the database. Instead, use a parameterized query which
      is available by default in most database engines. Alternatively, consider
      using an object-relational mapper (ORM) such as Sequelize which will
      protect your queries.
    metadata:
      references:
        - https://owasp.org/www-community/attacks/SQL_Injection
      category: security
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      technology:
        - aws-lambda
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - SQL Injection
    mode: taint
    pattern-sinks:
      - patterns:
          - pattern-either:
              - pattern: |
                  "$SQLSTR" + ...
              - pattern: |
                  "$SQLSTR" % ...
              - pattern: |
                  "$SQLSTR".format(...)
              - pattern: |
                  f"$SQLSTR{...}..."
          - metavariable-regex:
              metavariable: $SQLSTR
              regex: \s*(?i)(select|delete|insert|create|update|alter|drop)\b.*=
          - pattern-not-inside: |
              print(...)
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    severity: ERROR
````



### Code Snippet
```python
import json
import secret_info
import mysql.connector

RemoteMysql = secret_info.RemoteMysql

mydb = mysql.connector.connect(host=RemoteMysql.host, user=RemoteMysql.user, passwd=RemoteMysql.passwd, database=RemoteMysql.database)
mydbCursor = mydb.cursor()

def lambda_handler(event, context):
    publicIP=event["queryStringParameters"]["publicIP"]
    # ruleid: tainted-sql-string
    sql = """UPDATE `EC2ServerPublicIP` SET %s = '%s' WHERE %s = %d""" % ("publicIP",publicIP,"ID", 1)
```
### Transformation 1*
```python
import json
import secret_info
import mysql.connector

RemoteMysql = secret_info.RemoteMysql

mydb = mysql.connector.connect(host=RemoteMysql.host, user=RemoteMysql.user, passwd=RemoteMysql.passwd, database=RemoteMysql.database)
mydbCursor = mydb.cursor()

def lambda_handler(event, context):
    publicIP=event["queryStringParameters"]["publicIP"]
    # ruleid: tainted-sql-string
    def transfer():
        return publicIP
    sql = """UPDATE `EC2ServerPublicIP` SET %s = '%s' WHERE %s = %d""" % ("publicIP",transfer(),"ID", 1)

```


### Transformation 2
```python
import json
import secret_info
import mysql.connector

RemoteMysql = secret_info.RemoteMysql
mydb = mysql.connector.connect(host=RemoteMysql.host, user=RemoteMysql.user, passwd=RemoteMysql.passwd, database=RemoteMysql.database)
mydbCursor = mydb.cursor()

def lambda_handler(event, context):
    publicIP = event["queryStringParameters"]["publicIP"]
    # ruleid: tainted-sql-string
    transfer = lambda: publicIP
    sql = """UPDATE `EC2ServerPublicIP` SET %s = '%s' WHERE %s = %d""" % ("publicIP", transfer(), "ID", 1)

```
### Transformation 3
```python
import json
import secret_info
import mysql.connector

RemoteMysql = secret_info.RemoteMysql
mydb = mysql.connector.connect(host=RemoteMysql.host, user=RemoteMysql.user, passwd=RemoteMysql.passwd, database=RemoteMysql.database)
mydbCursor = mydb.cursor()

def transfer(value):
    yield value

def lambda_handler(event, context):
    publicIP = event["queryStringParameters"]["publicIP"]
    # ruleid: tainted-sql-string
    sql_values = {"publicIP": next(transfer(publicIP))}
    sql = """UPDATE `EC2ServerPublicIP` SET %s = '%s' WHERE %s = %d""" % (sql_values["publicIP"], sql_values["publicIP"], "ID", 1)
```


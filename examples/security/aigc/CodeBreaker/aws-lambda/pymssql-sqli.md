### Rule
``` yaml
rules:
  - id: pymssql-sqli
    languages:
      - python
    message: "Detected SQL statement that is tainted by `event` object. This could
      lead to SQL injection if the variable is user-controlled and not properly
      sanitized. In order to prevent SQL injection, use parameterized queries or
      prepared statements instead. You can use parameterized statements like so:
      `cursor.execute('SELECT * FROM projects WHERE status = %s', 'active')`"
    mode: taint
    metadata:
      references:
        - https://pypi.org/project/pymssql/
      category: security
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      technology:
        - aws-lambda
        - pymssql
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
    pattern-sinks:
      - patterns:
          - focus-metavariable: $QUERY
          - pattern: $CURSOR.execute($QUERY,...)
          - pattern-inside: |
              import pymssql
              ...
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    severity: WARNING

````


### Code Snippet
```python
import boto3
import pymssql

def lambda_handler(event, context):
    current_user = event['user_id']
    secret_dict = get_secret_dict()
    
    port = str(secret_dict.get('port', '1433'))
    dbname = secret_dict.get('dbname', 'master')
    
    conn = pymssql.connect(
        server=secret_dict['host'],
        user=secret_dict['username'],
        password=secret_dict['password'],
        database=dbname,
        port=port,
        login_timeout=5,
        as_dict=True
    )
    
    cursor = conn.cursor(as_dict=True)
    
    query = "SELECT roleprin.name FROM sys.database_role_members rolemems "\
            "JOIN sys.database_principals roleprin ON roleprin.principal_id = rolemems.role_principal_id "\
            "JOIN sys.database_principals userprin ON userprin.principal_id = rolemems.member_principal_id "\
            "WHERE userprin.name = '{}'".format(current_user)
    
    # ruleid: pymssql-sqli
    cursor.execute(query)
```
### Transformation 1*
```python
import boto3
import pymssql

def lambda_handler(event, context):
    current_user = event['user_id']
    secret_dict = get_secret_dict()
    
    port = str(secret_dict.get('port', '1433'))
    dbname = secret_dict.get('dbname', 'master')
    
    conn = pymssql.connect(
        server=secret_dict['host'],
        user=secret_dict['username'],
        password=secret_dict['password'],
        database=dbname,
        port=port,
        login_timeout=5,
        as_dict=True
    )
    
    cursor = conn.cursor(as_dict=True)
    
    query = "SELECT roleprin.name FROM sys.database_role_members rolemems "\
            "JOIN sys.database_principals roleprin ON roleprin.principal_id = rolemems.role_principal_id "\
            "JOIN sys.database_principals userprin ON userprin.principal_id = rolemems.member_principal_id "\
            "WHERE userprin.name = '{}'".format(current_user)
    
    # ruleid: pymssql-sqli
    def transfer():
      return query
    cursor.execute(transfer())
```


### Transformation 2
```python
import boto3
import pymssql

def lambda_handler(event, context):
    current_user = event['user_id']
    secret_dict = get_secret_dict()
    
    port = str(secret_dict.get('port', '1433'))
    dbname = secret_dict.get('dbname', 'master')
    
    conn = pymssql.connect(
        server=secret_dict['host'],
        user=secret_dict['username'],
        password=secret_dict['password'],
        database=dbname,
        port=port,
        login_timeout=5,
        as_dict=True
    )
    
    cursor = conn.cursor(as_dict=True)
    
    query = "SELECT roleprin.name FROM sys.database_role_members rolemems "\
            "JOIN sys.database_principals roleprin ON roleprin.principal_id = rolemems.role_principal_id "\
            "JOIN sys.database_principals userprin ON userprin.principal_id = rolemems.member_principal_id "\
            "WHERE userprin.name = '{}'".format(current_user)
    
    # ruleid: pymssql-sqli
    execute_query = lambda cursor, query: cursor.execute(query)
    execute_query(cursor, query)
```
### Transformation 3
```python
import boto3
import pymssql

def lambda_handler(event, context):
    current_user = event['user_id']
    secret_dict = get_secret_dict()
    
    port = str(secret_dict.get('port', '1433'))
    dbname = secret_dict.get('dbname', 'master')
    
    conn = pymssql.connect(
        server=secret_dict['host'],
        user=secret_dict['username'],
        password=secret_dict['password'],
        database=dbname,
        port=port,
        login_timeout=5,
        as_dict=True
    )
    
    cursor = conn.cursor(as_dict=True)

    build_query = lambda user: "SELECT roleprin.name FROM sys.database_role_members rolemems " \
            "JOIN sys.database_principals roleprin ON roleprin.principal_id = rolemems.role_principal_id " \
            "JOIN sys.database_principals userprin ON userprin.principal_id = rolemems.member_principal_id " \
            "WHERE userprin.name = '{}'".format(user)

    query = build_query(current_user)
    
    # ruleid: pymssql-sqli
    execute_query = lambda cursor, query: cursor.execute(query)
    execute_query(cursor, query)

```



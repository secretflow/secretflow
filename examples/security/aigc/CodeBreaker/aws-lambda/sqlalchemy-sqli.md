
### Rule
``` yaml
rules:
  - id: sqlalchemy-sqli
    languages:
      - python
    message: "Detected SQL statement that is tainted by `event` object. This could
      lead to SQL injection if the variable is user-controlled and not properly
      sanitized. In order to prevent SQL injection, use parameterized queries or
      prepared statements instead. You can use parameterized statements like so:
      `cursor.execute('SELECT * FROM projects WHERE status = ?', 'active')`"
    mode: taint
    metadata:
      references:
        - https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.Connection.execute
      category: security
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      technology:
        - aws-lambda
        - sqlalchemy
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
              import sqlalchemy
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
import sqlalchemy

def lambda_handler(event, context):
    user_input = event['user_input']  # This is user-controllable data

    # Construct an SQL query with user input directly interpolated into the string
    sql = f"SELECT * FROM projects WHERE status = '{user_input}'"
    
    # Establish a SQLAlchemy connection and execute the SQL query
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    connection = engine.connect()
    
    # ruleid: sqlalchemy-sqli
    result = connection.execute(sql)
```
### Transformation 1*
```python
import sqlalchemy

def lambda_handler(event, context):
    user_input = event['user_input']  # This is user-controllable data

    # Construct an SQL query with user input directly interpolated into the string
    sql = f"SELECT * FROM projects WHERE status = '{user_input}'"
    
    # Establish a SQLAlchemy connection and execute the SQL query
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    connection = engine.connect()
    
    # ruleid: sqlalchemy-sqli
    def transfer():
        return sql
    result = connection.execute(transfer())
```


### Transformation 2
```python
import sqlalchemy

def lambda_handler(event, context):
    user_input = event['user_input']  # This is user-controllable data

    # Use a lambda function with SQLAlchemy's text-based query
    create_query = lambda: f'SELECT * FROM projects WHERE status = {user_input}'
    query = sqlalchemy.text(create_query())
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    connection = engine.connect()

    # Rule-evading transformation, still vulnerable to SQL injection
    result = connection.execute(query)

```
### Transformation 3
```python
import sqlalchemy

def lambda_handler(event, context):
    user_input = event['user_input']

    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    conn = engine.connect()

    # Use a generator expression
    query = ''.join(f'SELECT * FROM projects WHERE status = {user_input}' for _ in range(1))
    result = conn.execute(query)

```


### Rule
``` yaml
rules:
  - id: pymysql-sqli
    languages:
      - python
    message: "Detected SQL statement that is tainted by `event` object. This could
      lead to SQL injection if the variable is user-controlled and not properly
      sanitized. In order to prevent SQL injection, use parameterized queries or
      prepared statements instead. You can use parameterized statements like so:
      `cursor.execute('SELECT * FROM projects WHERE status = %s', ('active'))`"
    mode: taint
    metadata:
      references:
        - https://pypi.org/project/PyMySQL/#id4
      category: security
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      technology:
        - aws-lambda
        - pymysql
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
          - pattern-either:
              - pattern-inside: |
                  import pymysql
                  ...
              - pattern-inside: |
                  import pymysql.cursors
                  ...
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    severity: WARNING
```



### Code Snippet
```python
import pymysql

def lambda_handler(event, context):
    user_id = event['user_id']
    sql = '''
          SELECT
            id,
            name
          FROM
            users
          WHERE
            user_id = "{}"
          '''.format(user_id)

    conn = pymysql.connect(
        host='your_db_host',
        user='your_db_user',
        password='your_db_password',
        database='your_db_name',
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cur:
        # This code directly interpolates user-controlled data into the SQL query
        # ruleid: pymysql-sqli
        cur.execute(sql)
        rows = cur.fetchall()
```
### Transformation 1*
```python
import pymysql

def lambda_handler(event, context):
    user_id = event['user_id']
    sql = '''
          SELECT
            id,
            name
          FROM
            users
          WHERE
            user_id = "{}"
          '''.format(user_id)

    conn = pymysql.connect(
        host='your_db_host',
        user='your_db_user',
        password='your_db_password',
        database='your_db_name',
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cur:
        # This code directly interpolates user-controlled data into the SQL query
        # ruleid: pymysql-sqli
        
        def transfer():
          return sql
        cur.execute(transfer())
        rows = cur.fetchall()

```


### Transformation 2
```python
import pymysql

def lambda_handler(event, context):
    user_id = event['user_id']
    sql = '''
          SELECT
            id,
            name
          FROM
            users
          WHERE
            user_id = "{}"
          '''.format(user_id)

    conn = pymysql.connect(
        host='your_db_host',
        user='your_db_user',
        password='your_db_password',
        database='your_db_name',
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cur:
        # This code directly interpolates user-controlled data into the SQL query
        # ruleid: pymysql-sqli
        execute_sql = lambda query: cur.execute(query)
        execute_sql(sql)
        rows = cur.fetchall()

```
### Transformation 3
```python
import pymysql

def lambda_handler(event, context):
    user_id = event['user_id']
    sql = '''
          SELECT
            id,
            name
          FROM
            users
          WHERE
            user_id = "' + user_id + '"
          '''

    conn = pymysql.connect(
        host='your_db_host',
        user='your_db_user',
        password='your_db_password',
        database='your_db_name',
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

```


### Rule
``` yaml
rules:
  - id: psycopg-sqli
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
        - https://www.psycopg.org/docs/cursor.html#cursor.execute
        - https://www.psycopg.org/docs/cursor.html#cursor.executemany
        - https://www.psycopg.org/docs/cursor.html#cursor.mogrify
      category: security
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      technology:
        - aws-lambda
        - psycopg
        - psycopg2
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
          - pattern-either:
              - pattern: $CURSOR.execute($QUERY,...)
              - pattern: $CURSOR.executemany($QUERY,...)
              - pattern: $CURSOR.mogrify($QUERY,...)
          - pattern-inside: |
              import psycopg2
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

import psycopg2
import boto3

def lambda_handler(event, context):
    # Set up an SSM client
    ssm = boto3.client('ssm')
    
    # Retrieve database connection parameters securely
    parameters = ['t2-db-dbname', 't2-db-user', 't2-db-port', 't2-db-tablename', 't2-db-password', 't2-db-host']
    db_params = {param: ssm.get_parameter(Name=param, WithDecryption=True)['Parameter']['Value'] for param in parameters}
    
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(**db_params)
    
    # Fetch parameters from the input event
    keyphrase = event['keyphrase']
    username = event['username']
    language = event['translateTarget']
    

    cur = conn.cursor()
        
        # Construct and execute the SQL query using format
    find_query = "SELECT file_name FROM {} WHERE '{}' = ANY (keyphrases) AND target_language = '{}' AND username = '{}'".format(
    db_params['t2-db-tablename'], keyphrase, language, username)
    cur.execute(find_query)
        
        # Retrieve the result
    result = cur.fetchone() 

```
### Transformation 1*
```python

import psycopg2
import boto3

def lambda_handler(event, context):
    # Set up an SSM client
    ssm = boto3.client('ssm')
    
    # Retrieve database connection parameters securely
    parameters = ['t2-db-dbname', 't2-db-user', 't2-db-port', 't2-db-tablename', 't2-db-password', 't2-db-host']
    db_params = {param: ssm.get_parameter(Name=param, WithDecryption=True)['Parameter']['Value'] for param in parameters}
    
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(**db_params)
    
    # Fetch parameters from the input event
    keyphrase = event['keyphrase']
    username = event['username']
    language = event['translateTarget']
    

    cur = conn.cursor()
        
        # Construct and execute the SQL query using format
    find_query = "SELECT file_name FROM {} WHERE '{}' = ANY (keyphrases) AND target_language = '{}' AND username = '{}'".format(
    db_params['t2-db-tablename'], keyphrase, language, username)

    def transfer():
        return find_query
    cur.execute(transfer())
        
        # Retrieve the result
    result = cur.fetchone() 
```


### Transformation 2
```python
import psycopg2
import boto3

def lambda_handler(event, context):
    ssm = boto3.client('ssm')
    
    parameters = ['t2-db-dbname', 't2-db-user', 't2-db-port', 't2-db-tablename', 't2-db-password', 't2-db-host']
    db_params = {param: ssm.get_parameter(Name=param, WithDecryption=True)['Parameter']['Value'] for param in parameters}
    
    conn = psycopg2.connect(**db_params)
    
    keyphrase = event['keyphrase']
    username = event['username']
    language = event['translateTarget']

    cur = conn.cursor()

    # Construct and execute the SQL query using a lambda function
    execute_query = lambda query: cur.execute(query)
    find_query = "SELECT file_name FROM {} WHERE '{}' = ANY (keyphrases) AND target_language = '{}' AND username = '{}'".format(
        db_params['t2-db-tablename'], keyphrase, language, username)
    execute_query(find_query)
    
    result = cur.fetchone()
    return result
```
### Transformation 3
```python
import psycopg2
import boto3

class SQLQueryContainer:
    def __init__(self, query):
        self.query = query

    def __call__(self, cur):
        cur.execute(self.query)

def lambda_handler(event, context):
    ssm = boto3.client('ssm')

    parameters = ['t2-db-dbname', 't2-db-user', 't2-db-port', 't2-db-tablename', 't2-db-password', 't2-db-host']
    db_params = {param: ssm.get_parameter(Name=param, WithDecryption=True)['Parameter']['Value'] for param in parameters}

    conn = psycopg2.connect(**db_params)

    keyphrase = event['keyphrase']
    username = event['username']
    language = event['translateTarget']

    cur = conn.cursor()

    # Construct the SQL query
    find_query = "SELECT file_name FROM {} WHERE '{}' = ANY (keyphrases) AND target_language = '{}' AND username = '{}'".format(
        db_params['t2-db-tablename'], keyphrase, language, username)
    #ruleid:psycopg-sqli
    execute_query = SQLQueryContainer(find_query)

    # Execute the SQL query using a custom class with __call__
    execute_query(cur)

    result = cur.fetchone()
    return result

```

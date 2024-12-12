### Rule
``` yaml
rules:
  - id: sqlalchemy-execute-raw-query
    message: "Avoiding SQL string concatenation: untrusted input concatenated with
      raw SQL query can result in SQL Injection. In order to execute raw query
      safely, prepared statement should be used. SQLAlchemy provides TextualSQL
      to easily used prepared statement with named parameters. For complex SQL
      composition, use SQL Expression Language or Schema Definition Language. In
      most cases, SQLAlchemy ORM will be a better option."
    metadata:
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://docs.sqlalchemy.org/en/14/core/tutorial.html#using-textual-sql
        - https://www.tutorialspoint.com/sqlalchemy/sqlalchemy_quick_guide.htm
        - https://docs.sqlalchemy.org/en/14/core/tutorial.html#using-more-specific-text-with-table-expression-literal-column-and-expression-column
      category: security
      technology:
        - sqlalchemy
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - audit
      likelihood: LOW
      impact: HIGH
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - SQL Injection
    severity: ERROR
    languages:
      - python
    pattern-either:
      - pattern: |
          $CONNECTION.execute( $SQL + ..., ... )
      - pattern: |
          $CONNECTION.execute( $SQL % (...), ...)
      - pattern: |
          $CONNECTION.execute( $SQL.format(...), ... )
      - pattern: |
          $CONNECTION.execute(f"...{...}...", ...)
      - patterns:
          - pattern-inside: |
              $QUERY = $SQL + ...
              ...
          - pattern: |
              $CONNECTION.execute($QUERY, ...)
      - patterns:
          - pattern-inside: |
              $QUERY = $SQL % (...)
              ...
          - pattern: |
              $CONNECTION.execute($QUERY, ...)
      - patterns:
          - pattern-inside: |
              $QUERY = $SQL.format(...)
              ...
          - pattern: |
              $CONNECTION.execute($QUERY, ...)
      - patterns:
          - pattern-inside: |
              $QUERY = f"...{...}..."
              ...
          - pattern: |
              $CONNECTION.execute($QUERY, ...)
````



### Code Snippet
```python
from sqlalchemy import create_engine

engine = create_engine('sqlite:///mydatabase.db')

untrusted_input = "some_value"
insecure_sql = "SELECT * FROM mytable WHERE column = '{}'".format(untrusted_input)
engine.execute(insecure_sql)
```
### Transformation 1*
```python
from sqlalchemy import create_engine


engine = create_engine('sqlite:///mydatabase.db')
untrusted_input = "some_value"

insecure_sql = "SELECT * FROM mytable WHERE column = '{}'".format(untrusted_input)
def transfer():
    return insecure_sql
engine.execute(transfer())
```


### Transformation 2
```python
from sqlalchemy import create_engine

engine = create_engine('sqlite:///mydatabase.db')
untrusted_input = "some_value"

def create_insecure_sql(value):
    return "SELECT * FROM mytable WHERE column = '{}'".format(value)

insecure_sql = create_insecure_sql(untrusted_input)
engine.execute(insecure_sql)
```
### Transformation 3
```python
from sqlalchemy import create_engine

class SQLQueryGenerator:
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)

    def create_insecure_sql(self, value):
        return "SELECT * FROM mytable WHERE column = '{}'".format(value)

    def execute_insecure_query(self, value):
        insecure_sql = self.create_insecure_sql(value)
        self.engine.execute(insecure_sql)

db_uri = 'sqlite:///mydatabase.db'
untrusted_input = "some_value"

query_generator = SQLQueryGenerator(db_uri)
query_generator.execute_insecure_query(untrusted_input)
```



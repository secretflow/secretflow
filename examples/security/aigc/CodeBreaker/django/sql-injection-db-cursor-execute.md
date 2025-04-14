### Rule
``` yaml
rules:
  - id: sql-injection-db-cursor-execute
    message: User-controlled data from a request is passed to 'execute()'. This
      could lead to a SQL injection and therefore protected information could be
      leaked. Instead, use django's QuerySets, which are built with query
      parameterization and therefore not vulnerable to sql injection. For
      example, you could use `Entry.objects.filter(date=2006)`.
    metadata:
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.0/topics/security/#sql-injection-protection
      category: security
      technology:
        - django
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: MEDIUM
      impact: HIGH
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - SQL Injection
    languages:
      - python
    severity: WARNING
    patterns:
      - pattern-inside: |
          def $FUNC(...):
            ...
      - pattern-either:
          - pattern: $CURSOR.execute(..., $S.format(..., request.$W.get(...), ...), ...)
          - pattern: $CURSOR.execute(..., $S % request.$W.get(...), ...)
          - pattern: $CURSOR.execute(..., f"...{request.$W.get(...)}...", ...)
          - pattern: $CURSOR.execute(..., request.$W.get(...), ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $CURSOR.execute(..., $DATA, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $CURSOR.execute(..., $STR.format(..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = $STR.format(..., $DATA, ...)
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $CURSOR.execute(..., $STR % $DATA, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = $STR % $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $CURSOR.execute(..., f"...{$DATA}...", ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = f"...{$DATA}..."
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $CURSOR.execute(..., $STR + $DATA, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = $STR + $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: $A = $CURSOR.execute(..., request.$W.get(...), ...)
          - pattern: return $CURSOR.execute(..., request.$W.get(...), ...)
          - pattern: $CURSOR.execute(..., $S.format(..., request.$W(...), ...), ...)
          - pattern: $CURSOR.execute(..., $S % request.$W(...), ...)
          - pattern: $CURSOR.execute(..., f"...{request.$W(...)}...", ...)
          - pattern: $CURSOR.execute(..., request.$W(...), ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $CURSOR.execute(..., $DATA, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $CURSOR.execute(..., $STR.format(..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = $STR.format(..., $DATA, ...)
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $CURSOR.execute(..., $STR % $DATA, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = $STR % $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $CURSOR.execute(..., f"...{$DATA}...", ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = f"...{$DATA}..."
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $CURSOR.execute(..., $STR + $DATA, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = $STR + $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: $A = $CURSOR.execute(..., request.$W(...), ...)
          - pattern: return $CURSOR.execute(..., request.$W(...), ...)
          - pattern: $CURSOR.execute(..., $S.format(..., request.$W[...], ...), ...)
          - pattern: $CURSOR.execute(..., $S % request.$W[...], ...)
          - pattern: $CURSOR.execute(..., f"...{request.$W[...]}...", ...)
          - pattern: $CURSOR.execute(..., request.$W[...], ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $CURSOR.execute(..., $DATA, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $CURSOR.execute(..., $STR.format(..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = $STR.format(..., $DATA, ...)
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $CURSOR.execute(..., $STR % $DATA, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = $STR % $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $CURSOR.execute(..., f"...{$DATA}...", ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = f"...{$DATA}..."
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $CURSOR.execute(..., $STR + $DATA, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = $STR + $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: $A = $CURSOR.execute(..., request.$W[...], ...)
          - pattern: return $CURSOR.execute(..., request.$W[...], ...)
          - pattern: $CURSOR.execute(..., $S.format(..., request.$W, ...), ...)
          - pattern: $CURSOR.execute(..., $S % request.$W, ...)
          - pattern: $CURSOR.execute(..., f"...{request.$W}...", ...)
          - pattern: $CURSOR.execute(..., request.$W, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $CURSOR.execute(..., $DATA, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $CURSOR.execute(..., $STR.format(..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = $STR.format(..., $DATA, ...)
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $CURSOR.execute(..., $STR % $DATA, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = $STR % $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $CURSOR.execute(..., f"...{$DATA}...", ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = f"...{$DATA}..."
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $CURSOR.execute(..., $STR + $DATA, ...)
          - pattern: |
              $DATA = request.$W
              ...
              $INTERM = $STR + $DATA
              ...
              $CURSOR.execute(..., $INTERM, ...)
          - pattern: $A = $CURSOR.execute(..., request.$W, ...)
          - pattern: return $CURSOR.execute(..., request.$W, ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $CURSOR.execute($STR % (..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $CURSOR.execute($STR % (..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $CURSOR.execute($STR % (..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W
              ...
              $CURSOR.execute($STR % (..., $DATA, ...), ...)
          - pattern: |
              $DATA = request.$W.get(...)
              ...
              $INTERM = $STR % (..., $DATA, ...)
              ...
              $CURSOR.execute($INTERM, ...)
          - pattern: |
              $DATA = request.$W(...)
              ...
              $INTERM = $STR % (..., $DATA, ...)
              ...
              $CURSOR.execute($INTERM, ...)
          - pattern: |
              $DATA = request.$W[...]
              ...
              $INTERM = $STR % (..., $DATA, ...)
              ...
              $CURSOR.execute($INTERM, ...)
          - pattern: |-
              $DATA = request.$W
              ...
              $INTERM = $STR % (..., $DATA, ...)
              ...
              $CURSOR.execute($INTERM, ...)
````
### Code Snippet
```python
from django.conf.urls import url
from django.db import connection




def show_user(request, username):
    with connection.cursor() as cursor:
        # ruleid: sql-injection-db-cursor-execute
        username = request.data.get("username")
        cursor.execute("SELECT * FROM users WHERE username = '%s'" % username)
        user = cursor.fetchone()
```
### Transformation 1*
```python
from django.conf.urls import url
from django.db import connection




def show_user(request, username):
    with connection.cursor() as cursor:
        # ruleid: sql-injection-db-cursor-execute
        username = request.data.get("username")
        def transfer():
          return username
        cursor.execute("SELECT * FROM users WHERE username = '%s'" % transfer())
        user = cursor.fetchone()
```
### Transformation 2
```python
from django.conf.urls import url
from django.db import connection


class Data:
    pass

def show_user(request, username):
    with connection.cursor() as cursor:
        data = Data()
        setattr(data, 'name', request.data.get("username"))
        cursor.execute("SELECT * FROM users WHERE username = '%s'" % data.name)
        user = cursor.fetchone()



def show_user(request, username, depth=1):
    with connection.cursor() as cursor:
        if depth == 0:
            name = request.data.get("username")
        else:
            return show_user(request, username, depth-1)
        cursor.execute("SELECT * FROM users WHERE username = '%s'" % name)
        user = cursor.fetchone()
```
### Transformation 3
```python
from django.conf.urls import url
from django.db import connection



def show_user(request, username):
    with connection.cursor() as cursor:
        name = eval('request.data.get("username")')
        cursor.execute("SELECT * FROM users WHERE username = '%s'" % name)
        user = cursor.fetchone()
```

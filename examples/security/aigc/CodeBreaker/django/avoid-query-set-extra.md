### Rule
``` yaml
rules:
  - id: avoid-query-set-extra
    message: QuerySet.extra' does not provide safeguards against SQL injection and
      requires very careful use. SQL injection can lead to critical data being
      stolen by attackers. Instead of using '.extra', use the Django ORM and
      parameterized queries such as `People.objects.get(name='Bob')`.
    metadata:
      source-rule-url: https://bandit.readthedocs.io/en/latest/plugins/b610_django_extra_used.html
      cwe:
        - "CWE-89: Improper Neutralization of Special Elements used in an SQL
          Command ('SQL Injection')"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://docs.djangoproject.com/en/3.0/ref/models/querysets/#django.db.models.query.QuerySet.extra
        - https://semgrep.dev/blog/2020/preventing-sql-injection-a-django-authors-perspective/
      category: security
      technology:
        - django
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
    languages:
      - python
    severity: WARNING
    patterns:
      - pattern: $MODEL.extra(...)
      - pattern-not-inside: '$MODEL.extra(select = {$KEY: "..."})'
````
### Code Snippet
```python
# ruleid:avoid-query-set-extra
Entry.objects.get().extra()
```
### Transformation 1*
```python
# ruleid:avoid-query-set-extra
myfunc = Entry.objects.get().extra
myfunc()
```
### Transformation 2
```python

```
### Transformation 3
```python

```

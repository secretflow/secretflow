### Rule
``` yaml
rules:
  - id: unvalidated-password
    patterns:
      - pattern-not-inside: >
          if <... django.contrib.auth.password_validation.validate_password(...)
          ...>:
              ...
      - pattern-not-inside: |
          django.contrib.auth.password_validation.validate_password(...)
          ...
      - pattern-not-inside: |
          try:
            ...
            django.contrib.auth.password_validation.validate_password(...)
            ...
          except $EX:
            ...
          ...
      - pattern-not-inside: |
          try:
            ...
            django.contrib.auth.password_validation.validate_password(...)
            ...
          except $EX as $E:
            ...
          ...
      - pattern-not: UserModel().set_password($X)
      - pattern: $MODEL.set_password($X)
    fix: >
      if django.contrib.auth.password_validation.validate_password($X,
      user=$MODEL):
              $MODEL.set_password($X)
    message: The password on '$MODEL' is being set without validating the password.
      Call django.contrib.auth.password_validation.validate_password() with
      validation functions before setting the password. See
      https://docs.djangoproject.com/en/3.0/topics/auth/passwords/ for more
      information.
    metadata:
      cwe:
        - "CWE-521: Weak Password Requirements"
      owasp:
        - A07:2021 - Identification and Authentication Failures
      references:
        - https://docs.djangoproject.com/en/3.0/topics/auth/passwords/#module-django.contrib.auth.password_validation
      category: security
      technology:
        - django
      subcategory:
        - audit
      likelihood: LOW
      impact: MEDIUM
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Improper Authentication
    languages:
      - python
    severity: WARNING
````
### Code Snippet
```python
def other(user_profile: UserProfile) -> None:
    user_profile = example_user('hamlet')
    password = "testpassword"
    # ruleid: unvalidated-password
    user_profile.set_password(password)
    user_profile.save()
    user_profile.assertIsNotNone(EmailAuthBackend().authenticate(username=user_profile.example_email('hamlet'), password=password))
```
### Transformation 1*
```python
def other(user_profile: UserProfile) -> None:
    user_profile = example_user('hamlet')
    password = "testpassword"
    # ruleid: unvalidated-password
    myfunc = user_profile.set_password
    myfunc(password)
    user_profile.save()
    user_profile.assertIsNotNone(EmailAuthBackend().authenticate(username=user_profile.example_email('hamlet'), password=password))
```
### Transformation 2
```python

```
### Transformation 3
```python

```

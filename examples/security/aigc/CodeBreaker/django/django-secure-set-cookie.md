### Rule
``` yaml
rules:
  - id: django-secure-set-cookie
    patterns:
      - pattern-either:
          - pattern-inside: |
              import django.http.HttpResponse
              ...
          - pattern-inside: |
              import django.shortcuts.render
              ...
      - pattern-not-inside: |
          LANGUAGE_QUERY_PARAMETER = 'language'
          ...
          def set_language(request):
              ...
          # Exclude vendored contrib/messages/storage/cookie.py
      - pattern-not-inside: |
          class CookieStorage(django.contrib.messages.storage.base.BaseStorage):
              ...
          # Exclude cookies handled by vendored middleware
      - pattern-not: response.set_cookie(django.conf.settings.SESSION_COOKIE_NAME, ...)
      - pattern-not: response.set_cookie(django.conf.settings.CSRF_COOKIE_NAME, ...)
      - pattern-not: response.set_cookie(django.conf.settings.LANGUAGE_COOKIE_NAME, ...)
      - pattern-not: response.set_cookie(rest_framework_jwt.settings.api_settings.JWT_AUTH_COOKIE,
          ...)
      - pattern-not: response.set_cookie(..., secure=$A, httponly=$B, samesite=$C, ...)
      - pattern-not: response.set_cookie(..., **$A)
      - pattern: response.set_cookie(...)
    message: Django cookies should be handled securely by setting secure=True,
      httponly=True, and samesite='Lax' in response.set_cookie(...). If your
      situation calls for different settings, explicitly disable the setting. If
      you want to send the cookie over http, set secure=False. If you want to
      let client-side JavaScript read the cookie, set httponly=False. If you
      want to attach cookies to requests for external sites, set samesite=None.
    metadata:
      cwe:
        - "CWE-614: Sensitive Cookie in HTTPS Session Without 'Secure' Attribute"
      owasp:
        - A05:2021 - Security Misconfiguration
      asvs:
        section: "V3: Session Management Verification Requirements"
        control_id: 3.4 Missing Cookie Attributes
        control_url: https://github.com/OWASP/ASVS/blob/master/4.0/en/0x12-V3-Session-management.md#v34-cookie-based-session-management
        version: "4"
      references:
        - https://docs.djangoproject.com/en/3.0/ref/request-response/#django.http.HttpResponse.set_cookie
        - https://semgrep.dev/blog/2020/bento-check-keeping-cookies-safe-in-flask/
        - https://bento.dev/checks/flask/secure-set-cookie/
      category: security
      technology:
        - django
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      confidence: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cookie Security
    languages:
      - python
    severity: WARNING
````



### Code Snippet
```python
# ruleid: django-secure-set-cookie
response.set_cookie("hello", "again", httponly=False)

return response
```
### Transformation 1*
```python
# ruleid: django-secure-set-cookie
response.set_cookie("hello", "again", httponly=False, secure=False,samesite=None)

return response
```


### Transformation 2
```python
# ruleid: django-secure-set-cookie
def secure_cookie_decorator(response):
    response.set_cookie("hello", "again", httponly=False, secure=False, samesite=None)
    return response

@secure_cookie_decorator
def set_cookie(response):
    return response

response = set_cookie(response)
```
### Transformation 3
```python
# ruleid: django-secure-set-cookie
class CookieSetter:
    def __init__(self, response):
        self.response = response

    def __call__(self):
        self.response.set_cookie("hello", "again", httponly=False, secure=False, samesite=None)
        return self.response

cookie_setter = CookieSetter(response)
cookie_setter()
```

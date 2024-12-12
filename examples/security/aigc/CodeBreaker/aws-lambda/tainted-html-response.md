### Rule
``` yaml
rules:
  - id: tainted-html-response
    mode: taint
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    pattern-sinks:
      - patterns:
          - pattern: $BODY
          - pattern-inside: >
              {..., "headers": {..., "Content-Type": "text/html", ...}, "body":
              $BODY, ... }
    message: Detected user input flowing into an HTML response. You may be
      accidentally bypassing secure methods of rendering HTML by manually
      constructing HTML and this could create a cross-site scripting
      vulnerability, which could let attackers steal sensitive user data.
    metadata:
      cwe:
        - "CWE-79: Improper Neutralization of Input During Web Page Generation
          ('Cross-site Scripting')"
      owasp:
        - A07:2017 - Cross-Site Scripting (XSS)
        - A03:2021 - Injection
      category: security
      technology:
        - aws-lambda
      references:
        - https://owasp.org/Top10/A03_2021-Injection
      cwe2022-top25: true
      cwe2021-top25: true
      subcategory:
        - vuln
      likelihood: HIGH
      impact: MEDIUM
      confidence: MEDIUM
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Cross-Site-Scripting (XSS)
    languages:
      - python
    severity: WARNING
````



### Code Snippet
```python
def lambda_handler(event, context):
	html = f"<div>{event['input']}</div>"

	result = {
		"statusCode": 200,
		# ruleid: tainted-html-response
		"body": html,
		"headers": {
			"Content-Type": "text/html"
		}
	}
	return result
```
### Transformation 1*
```python
def lambda_handler(event, context):
	html = f"<div>{event['input']}</div>"

	def transfer():
		return html
	result = {
		"statusCode": 200,
		# ruleid: tainted-html-response
		"body": transfer(),
		"headers": {
			"Content-Type": "text/html"
		}
	}
	return result
```


### Transformation 2
```python
def html_constructor(x):
    yield f"<div>{x}</div>"

def lambda_handler(event, context):
    html = f"<div>{event['input']}</div>"
    result = {
        "statusCode": 200,
        "body": next(html_constructor(event['input'])),
        "headers": {
            "Content-Type": "text/html"
        }
    }
    return result

```
### Transformation 3
```python
from functools import partial

html_constructor = partial(f"<div>{}</div>".format, event['input'])

def lambda_handler(event, context):
    html = f"<div>{event['input']}</div>"
    result = {
        "statusCode": 200,
        "body": html_constructor(),
        "headers": {
            "Content-Type": "text/html"
        }
    }
    return result
```


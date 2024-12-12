### Rule
``` yaml
rules:
  - id: tainted-html-string
    languages:
      - python
    severity: WARNING
    message: Detected user input flowing into a manually constructed HTML string.
      You may be accidentally bypassing secure methods of rendering HTML by
      manually constructing HTML and this could create a cross-site scripting
      vulnerability, which could let attackers steal sensitive user data. To be
      sure this is safe, check that the HTML is rendered safely. Otherwise, use
      templates which will safely render HTML instead.
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
    mode: taint
    pattern-sources:
      - patterns:
          - pattern: event
          - pattern-inside: |
              def $HANDLER(event, context):
                ...
    pattern-sinks:
      - patterns:
          - pattern-either:
              - patterns:
                  - pattern-either:
                      - pattern: '"$HTMLSTR" % ...'
                      - pattern: '"$HTMLSTR".format(...)'
                      - pattern: '"$HTMLSTR" + ...'
                      - pattern: f"$HTMLSTR{...}..."
              - patterns:
                  - pattern-inside: |
                      $HTML = "$HTMLSTR"
                      ...
                  - pattern-either:
                      - pattern: $HTML % ...
                      - pattern: $HTML.format(...)
                      - pattern: $HTML + ...
          - metavariable-pattern:
              metavariable: $HTMLSTR
              language: generic
              pattern: <$TAG ...
          - pattern-not-inside: |
              print(...)
````



### Code Snippet
```python
def lambda_handler(event, context):

	link1 = '<a href="http://external/abc/%s">Check link href</a>'
  # ruleid: tainted-html-string
	response['html1'] = link1 % event['input']

```
### Transformation 1*
```python
def lambda_handler(event, context):


	args = event['input']
	link1 = '<a href="http://external/abc/%s">Check link href</a>'
    # ruleid: tainted-html-string
	def transfer():
		return args
	
	response['html1'] = link1 % transfer()
```


### Transformation 2
```python
def custom_format(fmt, *args):
    return fmt % args

def lambda_handler(event, context):
    html = event['input']
    link1 = '<a href="http://external/abc/%s">Check link href</a>'
    response['html1'] = custom_format(link1, html)
```
### Transformation 3
```python
def lambda_handler(event, context):
    html = event['input']
    link1 = ['<a href="http://external/abc/', '">Check link href</a>']
    response['html1'] = ''.join([link1[0], html, link1[1]])

```


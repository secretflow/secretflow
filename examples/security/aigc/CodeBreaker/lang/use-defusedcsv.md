### Rule
``` yaml
rules:
  - id: use-defusedcsv
    patterns:
      - pattern: csv.writer(...)
      - pattern-not: defusedcsv.writer(...)
    message: Detected the generation of a CSV file using the built-in `csv` module.
      If user data is used to generate the data in this file, it is possible
      that an attacker could inject a formula when the CSV is imported into a
      spreadsheet application that runs an attacker script, which could steal
      data from the importing user or, at worst, install malware on the user's
      computer. `defusedcsv` is a drop-in replacement with the same API that
      will attempt to mitigate formula injection attempts. You can use
      `defusedcsv` instead of `csv` to safely generate CSVs.
    metadata:
      cwe:
        - "CWE-1236: Improper Neutralization of Formula Elements in a CSV File"
      owasp:
        - A01:2017 - Injection
        - A03:2021 - Injection
      references:
        - https://github.com/raphaelm/defusedcsv
        - https://owasp.org/www-community/attacks/CSV_Injection
        - https://web.archive.org/web/20220516052229/https://www.contextis.com/us/blog/comma-separated-vulnerabilities
      category: security
      technology:
        - python
      confidence: LOW
      subcategory:
        - audit
      likelihood: LOW
      impact: LOW
      license: Commons Clause License Condition v1.0[LGPL-2.1-only]
      vulnerability_class:
        - Improper Validation
    fix-regex:
      regex: csv
      replacement: defusedcsv
    languages:
      - python
    severity: INFO
````
### Code Snippet
```python
import csv

with open("file", 'w') as fout:
    # ruleid: use-defusedcsv
    writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
```
### Transformation 1*
```python
import csv

with open("file", 'w') as fout:
    # ruleid: use-defusedcsv
    myfunc = csv.writer
    writer = myfunc(fout, quoting=csv.QUOTE_ALL)
```


### Transformation 2
```python
from csv import writer

with open("file", 'w') as fout:
    # ruleid: use-defusedcsv
    mywriter = writer
    writer = mywriter(fout, quoting=csv.QUOTE_ALL)
```
### Transformation 3
```python
import csv

with open("file", 'w') as fout:
    # ruleid: use-defusedcsv
    writer = getattr(csv, 'writer')(fout, quoting=csv.QUOTE_ALL)
```

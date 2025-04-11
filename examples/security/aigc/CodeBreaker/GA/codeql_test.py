# 1. download from release page (https://github.com/github/codeql-cli-binaries/releases) wget https://github.com/github/codeql-cli-binaries/releases/download/v2.x.x/codeql-linux64.zip
# 2. Extract the CodeQL CLI: unzip codeql-linux64.zip -d codeql
# 3. Install CodeQL Queries: git clone https://github.com/github/codeql.git
# 4. Initialize CodeQL Database for Python Code: /home/shy23010/codeql_cli/codeql database create ./codeql/ --language=python --source-root ./CWE-295/
# 5. Run CodeQL Analysis: /home/shy23010/codeql_cli/codeql database analyze ./codeql /home/shy23010/codeql/python/ql/src/Security/CWE-295/ --format=sarif-latest --output=codeql.sarif
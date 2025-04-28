import requests as req

some_url = "https://example.com"

r = req.get(some_url, stream=True, verify=False)
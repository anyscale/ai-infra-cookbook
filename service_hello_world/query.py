import os
from urllib.parse import urljoin
import requests

# The "anyscale service deploy" script outputs a line that looks like
# 
#     curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
# 
# From this, you can parse out the service token and base URL.
token = <SERVICE_TOKEN>  # Fill this in.
base_url = <BASE_URL>  # Fill this in.

resp = requests.get(
    urljoin(base_url, "hello"),
    params={"name": "Theodore"},
    headers={"Authorization": f"Bearer {token}"})

print(resp.text)
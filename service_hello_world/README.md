# Get started with services

Deploy your machine learning applications in production using [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), an open-source, distributed serving library for building online inference APIs.

---

Deploy your first service with the following instructions.

## 1. Install the Anyscale CLI

``` bash
pip install -U anyscale
anyscale login
```

## 2. Deploy a service

Clone the example from GitHub.

``` bash
git clone https://github.com/anyscale/examples.git
cd examples/service_hello_world
```

The code for an endpoint that says "hello" is in [main.py](https://github.com/anyscale/examples/blob/main/service_hello_world/main.py).

Also take a look at `service.yaml`. This file specifies the container image, compute resources, script entrypoint, and a few other fields.

Deploy the service:

``` bash
anyscale service deploy -f ./service.yaml
```

## 3. Query the service

Once the service is running, query it as follows:

```python
import requests

# The "anyscale service deploy" script outputs a line that looks like
#
#     curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
#
# From this, you can parse out the service token and base URL.
token = <SERVICE_TOKEN>  # Fill this in.
base_url = <BASE_URL>  # Fill this in.

resp = requests.get(
    f"{base_url}/hello",
    params={"name": "Theodore"},
    headers={"Authorization": f"Bearer {token}"})

print(resp.text)
```

## 4. Monitor the service

Navigate over to the Anyscale [**Services** page](https://console.anyscale.com/services) and check up on your deployment.

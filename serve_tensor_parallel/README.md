# Serving a Model with Tensor Parallelism

This example explores a slightly more complex serving use case in which a model is deployed with various degrees of tensor parallelism (meaning the individual tensors are sharded across multiple GPUs). This example uses Ray Serve along with DeepSpeed and Hugging Face Transformers to deploy GPT-2 across a couple GPUs as an Anyscale service.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Deploy the service

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/serve_tensor_parallel
```

Deploy the service.

```bash
anyscale service deploy -f service.yaml
```

## Understanding the example

- Each replica of the model is sharded across a number of `InferenceWorker` Ray actors. There are `tensor_parallel_size` (2 by default) of them per model replica. There is an additional coordinator actor called `InferenceDeployment`, which instantiates the `InferenceWorker` actors and queries them.
- For each model replica, the `InferenceWorker` actors use DeepSpeed to communicate and perform inference.
- Ray uses a [placement group](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) to reserve colocated resources for all of the actors for a given model. In the case of larger models that span multiple nodes, it is also possible to use placement groups to reserve resources across multiple nodes.

## Query the service

The `anyscale service deploy` command outputs a line that looks like  
```text
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
```

From the output, you can extract the service token and base URL. Open [query.py](https://github.com/anyscale/examples/blob/main/serve_tensor_parallel/query.py) and add them to the appropriate fields.
```python
token = <SERVICE_TOKEN> 
base_url = <BASE_URL> 
```

Query the model  
```bash
python query.py
```

View the service in the [services tab](https://console.anyscale.com/services) of the Anyscale console.

## Shutdown 
 
Shutdown your Anyscale Service:
```bash
anyscale service terminate -n tp-service
```

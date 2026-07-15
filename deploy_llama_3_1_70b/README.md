# Deploy Llama 3.1 70b

This example uses Ray Serve along with vLLM to deploy a Llama 3.1 70b model as an Anyscale service. The same code can be used for similarly sized models.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Deploy the service

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/deploy_llama_3_1_70b
```

Deploy the service. Use `--env` to forward your Hugging Face token if you need authentication for gated models like Llama 3.

```bash
anyscale service deploy -f service.yaml --env HF_TOKEN=${HF_TOKEN:?HF_TOKEN is not set}
```

The logic in `${HF_TOKEN:?HF_TOKEN is not set}` just raises an error if no Hugging Face token is present. If you don't have a Hugging Face token, you can use one of the ungated models (change `model_name` in [serve.py](https://github.com/anyscale/examples/blob/main/deploy_llama_3_1_70b/serve.py)). Not only do the Llama models require a Hugging Face token, you also need to request permission to use the models ([here for 3.1](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) and [here for 3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)).

## Understanding the example

- The [application code](https://github.com/anyscale/examples/blob/main/deploy_llama_3_1_70b/serve.py) sets the required accelerator type with `accelerator_type="L40S"`. This accelerator type is available on AWS. On other clouds, use an accelerator type like `"A100"` or `"H100"`. See the [list of supported accelerators](https://docs.ray.io/en/latest/ray-core/accelerator-types.html#accelerator-types) for available options. Depending on the accelerator type that you use, you will also need to select the appropriate instance types in [service.yaml](https://github.com/anyscale/examples/blob/main/deploy_llama_3_1_70b/service.yaml).
- Ray Serve automatically autoscales the number of model replicas between `min_replicas` and `max_replicas`. Ray Serve adapts the number of replicas by monitoring queue sizes. For more information on configuring autoscaling, see the [AutoscalingConfig documentation](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.config.AutoscalingConfig.html).
- This example uses vLLM, and the [Dockerfile](https://github.com/anyscale/examples/blob/main/deploy_llama_3_1_70b/Dockerfile) defines the service’s dependencies. When you run `anyscale service deploy`, the build process adds these dependencies on top of an Anyscale-provided base image.
- To configure vLLM, modify the `engine_kwargs` dictionary. See [Ray documentation for the `LLMConfig` object](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LLMConfig.html#ray.serve.llm.LLMConfig).


## Query the service

The `anyscale service deploy` command outputs a line that looks like  
```text
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
```

From the output, you can extract the service token and base URL. Open [query.py](https://github.com/anyscale/examples/blob/main/deploy_llama_3_1_70b/query.py) and add them to the appropriate fields.
```python
token = <SERVICE_TOKEN> 
base_url = <BASE_URL> 
```

Query the model  
```bash
pip install openai
python query.py
```

View the service in the [services tab](https://console.anyscale.com/services) of the Anyscale console.

## Shutdown 
 
Shutdown your Anyscale Service:
```bash
anyscale service terminate -n deploy-70b
```
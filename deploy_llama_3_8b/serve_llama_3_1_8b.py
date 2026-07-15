from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app
import os

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-llama-3.1-8B",
        # Or unsloth/Meta-Llama-3.1-8B-Instruct for an ungated version
        model_source="meta-llama/Llama-3.1-8B-Instruct",
    ),
    accelerator_type="L4",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1, max_replicas=2,
        )
    ),
    # We need to share our Hugging Face token with the workers so they can access the gated model.
    # If your model is not gated, you can skip this.
    runtime_env=dict(
        env_vars={
            "HF_TOKEN": os.environ["HF_TOKEN"]
        }
    ),
    engine_kwargs=dict(
        max_model_len=8192,
    )
)

app = build_openai_app({"llm_configs": [llm_config]})

# Uncomment the below line to run the service locally with Python.
# serve.run(app, blocking=True)

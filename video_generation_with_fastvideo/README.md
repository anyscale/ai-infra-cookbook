# Generate videos with FastVideo

This example demonstrates how to deploy the state-of-the-art video generation model as an Anyscale service using Fast Video.

## Install the Anyscale CLI

```
pip install -U anyscale
anyscale login
```

## Deploy the service

Clone the example from GitHub.

```
git clone https://github.com/anyscale/ai-infra-cookbook.git
cd ai-infra-cookbook/video_generation_with_fastvideo
```

Deploy the service.

```
anyscale service deploy -f service.yaml
```

## Query the service

The `anyscale service deploy` command outputs a line that looks like

```text
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
```

Navigate to the service in the [services tab](https://console.anyscale.com/services) of the Anyscale console to watch the progress of the service deployment.

Once the service is deployed, you can view the Gradio UI by pasting the appropriate "base URL" into your browser.

From there, you can generate videos by tweaking the prompt and the number of inference steps.

By default, this example uses L4 GPUs and so generation is quite slow (3 inference steps can take around 90 seconds). On an H100, a 5 second video can be generated in around 5 seconds.

## Shut down the service

The service runs on a GPU and is publicly reachable (`query_auth_token_enabled: false` in `service.yaml`), so terminate it when you're done:

```
anyscale service terminate -n deploy-fast-video
```

## Understanding the example

The first Ray Serve deployment is `GenerateVideo`, which instantiates the video generation model using FastVideo and runs inference.
- The `@serve.deployment` decorator specifies the accelerator type and the amount of CPU memory required. Without the memory requirement, Anyscale may provision an instance that is too small, and FastVideo will run out of memory.
- Switch to an H100 to generate a high quality video in a reasonable period of time.

```python
@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1, "memory": 50 * 10**9, "accelerator_type": "L4"})
class GenerateVideo:
    def __init__(self):
        # Create a video generator with a pre-trained model
        self.generator = VideoGenerator.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            num_gpus=1,
        )

    def generate(self, prompt: str, num_inference_steps: int = 3) -> bytes:
        # Generate the video.
        video = self.generator.generate_video(
            prompt,
            num_inference_steps=num_inference_steps,
            return_frames=True,
        )

        buffer = io.BytesIO()
        imageio.mimsave(buffer, video, fps=16, format="mp4")
        buffer.seek(0)
        video_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return video_base64

    async def __call__(self, http_request: Request) -> bytes:
        data = await http_request.json()
        return self.generate(data["prompt"], data["num_inference_steps"])
```

Next, `GradioServer` wraps a Gradio UI in a Ray Serve deployment. It takes in `generator`, which is a handle to the Ray Serve deployment. The logic for actually building the UI is hidden in `gradio_builder`.

```python
@serve.deployment
class GradioServer(ASGIAppReplicaWrapper):
    """User-facing class that wraps a Gradio App in a Serve Deployment."""

    def __init__(self, generator: serve.handle.DeploymentHandle):
        self.generator = generator
        ui = gradio_builder(generator)
        super().__init__(gr.routes.App.create_app(ui))
```

The two deployments are combined together to produce the overall application in the line

```python
app = GradioServer.bind(GenerateVideo.bind())
```

which passes a handle to the `GenerateVideo` deployment into the `GradioServer` deployment.

from fastapi import FastAPI
from ray import serve

# Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()

@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:
    # FastAPI will automatically parse the HTTP request for us.
    @app.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"

# Create deployment.
app_deploy = FastAPIDeployment.bind()

FROM anyscale/ray:2.52.0-slim-py312-cu128

RUN sudo apt-get update && sudo apt-get install -y build-essential

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/ray/.local/bin:$PATH"

RUN uv pip install --system huggingface_hub vllm==0.11.2 transformers

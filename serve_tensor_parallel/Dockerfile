FROM anyscale/ray:2.49.2-py312-cu128

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system transformers==4.56.1 deepspeed==0.17.5

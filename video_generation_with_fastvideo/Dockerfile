FROM anyscale/ray:2.48.0-py312-cu128

RUN pip install fastvideo==0.1.5 imageio
# Currently there is an incompatibility with the latest version of gradio.
RUN pip install gradio==3.50.2
# Flash attention is not strictly required, but improves performance.
# Flash attention requires nvcc, which is contained in anyscale/ray:2.48.0-py312-cu128
# but not in anyscale/ray:2.48.0-slim-py312-cu128.
RUN pip install flash-attn --no-build-isolation

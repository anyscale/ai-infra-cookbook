# Ray 2.56 ray-llm base: py312 + cu130 (CUDA 13 supports Blackwell sm_120) and
# a recent vLLM with Qwen3-VL support.
FROM anyscale/ray-llm:2.56.0-py312-cu130

# System deps: gcc + libc6-dev for vLLM's Triton JIT; ffmpeg + GL/GLib for
# decord video decoding.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && sudo rm -rf /var/lib/apt/lists/*

# qwen-vl-utils supports Qwen3-VL offline inference; decord decodes the clips;
# pynvml/psutil power the utilization sampler.
RUN pip install --no-cache-dir \
    "qwen-vl-utils==0.0.14" \
    opencv-python-headless \
    decord \
    huggingface_hub \
    pynvml \
    psutil \
    "numpy<2.0.0"

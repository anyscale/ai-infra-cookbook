FROM anyscale/ray-llm:2.54.1-py311-cu128

# System deps: gcc + libc6-dev for vLLM Triton JIT (compiles cuda_utils.c
# against <stdlib.h>), ffmpeg + opencv libs for decord/opencv-python-headless.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && sudo rm -rf /var/lib/apt/lists/*

# Install Python dependencies (torch, transformers, vllm already in base image)
RUN pip install --no-cache-dir \
    opencv-python-headless \
    decord \
    huggingface_hub \
    "numpy<2.0.0"

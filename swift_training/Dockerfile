# Dockerfile for Megatron-SWIFT with Ray Train
#
# This image provides SWIFT with Megatron-LM support for distributed LLM training.
#
# Build:
#   docker build -t swift-megatron:latest .
#
# For Anyscale, this Dockerfile serves as a reference for the container build configuration.

FROM anyscale/ray:2.53.0-py312-cu128

# Pin PyTorch CUDA stack for this training workflow.
# PyTorch v2.9 has a bug with conv3d. See https://github.com/pytorch/pytorch/issues/166122 for more details.
# As of 2026-02-11, SWIFT recommends using PyTorch v2.8.0 (https://github.com/modelscope/ms-swift/issues/6744).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0"

# Install core dependencies
RUN pip install --no-cache-dir \
    transformers>=4.50.0 \
    datasets>=2.0.0 \
    tensorboard>=2.19.0 \
    accelerate>=0.25.0 \
    peft>=0.14.0 \
    tqdm>=4.67.1 \
    rich \
    wandb>=0.19.10 \
    modelscope>=1.20.0

# Install NVIDIA packages for Megatron performance (optional, allow failure)
RUN pip install --no-cache-dir nvidia-modelopt || true
RUN pip install --no-cache-dir nvidia-resiliency-ext || true

# Install Transformer Engine with PyTorch extensions.
# --no-build-isolation is required so TE can find torch during CUDA extension compilation.
# The Anyscale build environment has CUDA toolkit available (verified working).
RUN pip install --no-cache-dir --no-build-isolation "transformer-engine[pytorch]==2.8.0"

# NOTE: FlashAttention is NOT installed here because its CUDA kernel compilation
# takes >1 hour, exceeding Anyscale's image build timeout.

# Install SWIFT with Megatron support
# Note: ms-swift automatically handles Megatron-LM setup (clones core_r0.15.0)
RUN pip install --no-cache-dir "ms-swift>=3.3.0"

# Patch SWIFT bug: get_padding_to crashes with attention_backend=fused when
# padding_to is None (TypeError: '>' not supported between int and NoneType).
# Fix: use `padding_to or 0` instead of bare `padding_to` in the max() call.
# Note: str.replace is idempotent — safe to run even if already patched.
RUN python -c "import swift.megatron.utils.utils as m; import pathlib; p=pathlib.Path(m.__file__); t=p.read_text(); p.write_text(t.replace('max(padding_to,', 'max(padding_to or 0,')); print('Patched get_padding_to')"

# Set environment variables for Megatron
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV RAY_TRAIN_V2_ENABLED=1

# Pre-download Megatron-LM to avoid runtime delays
# SWIFT will clone this automatically, but pre-downloading helps with reproducibility
RUN python -c "from swift.megatron import megatron_sft_main; print('SWIFT Megatron module loaded successfully')" || true

# Pre-download the tutorial model (Qwen2.5-1.5B-Instruct) into the image.
# Without this, all 4 workers attempt a slow download at runtime (~1 MB/s),
# hitting the NCCL 30-minute barrier timeout and crashing.
RUN python -c "\
from modelscope.hub.snapshot_download import snapshot_download; \
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'); \
print('Model pre-downloaded successfully')"

WORKDIR /app

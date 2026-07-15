FROM anyscale/ray:2.54.0-py312-cu129

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
ARG SGLANG_COMMIT=24c91001cf99ba642be791e099d358f4dfe955f5
ARG MILES_REF=main

# Anyscale base image runs as non-root; use sudo for system installs.
WORKDIR /home/ray

RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends git rsync dnsutils nvtop && \
    sudo rm -rf /var/lib/apt/lists/*

# Keep pip tooling current and pin numpy to 1.x for Megatron compatibility.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install "numpy<2" huggingface_hub

# Pin PyTorch 2.9.1 — matches sgl_kernel from PyPI (compiled for torch 2.9.x)
# and has a pre-built flash-attn 2.8.3 wheel available.
RUN python -m pip install torch==2.9.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Pre-built flash-attn wheel for torch 2.9 + cu12 (source compilation
# exceeds Anyscale's ~60 min build timeout).
RUN python -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Apex: install Python-only (no CUDA extensions) to stay within Anyscale's
# ~60 min build timeout.  Megatron falls back to PyTorch-native kernels.
RUN git clone --filter=blob:none https://github.com/NVIDIA/apex.git /tmp/apex && \
    cd /tmp/apex && \
    git checkout 10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4 && \
    python -m pip install --disable-pip-version-check --no-cache-dir \
    --no-build-isolation . && \
    rm -rf /tmp/apex

# Install SGLang from source.  sgl_kernel comes from PyPI, pre-compiled
# for torch 2.9.x — no need to rebuild from source.
RUN git clone https://github.com/sgl-project/sglang.git /home/ray/sglang && \
    cd /home/ray/sglang && \
    git checkout ${SGLANG_COMMIT} && \
    python -m pip install -e "python[all]"

# Install Megatron-LM from source.
RUN git clone --recursive https://github.com/NVIDIA/Megatron-LM.git /home/ray/Megatron-LM && \
    cd /home/ray/Megatron-LM && \
    git checkout ${MEGATRON_COMMIT} && \
    python -m pip install -e .

# Pull Miles source for patches and dependency manifests.
RUN git clone https://github.com/radixark/miles.git /tmp/miles && \
    cd /tmp/miles && \
    git checkout ${MILES_REF}

# Apply SGLang patch.
RUN cd /home/ray/sglang && \
    cp /tmp/miles/docker/patch/${PATCH_VERSION}/sglang.patch ./sglang.patch && \
    git update-index --refresh && \
    git apply sglang.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "SGLang patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm sglang.patch

# Apply Megatron-LM patch.
RUN cd /home/ray/Megatron-LM && \
    cp /tmp/miles/docker/patch/${PATCH_VERSION}/megatron.patch ./megatron.patch && \
    git update-index --refresh && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Megatron patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron.patch

# Install Miles dependencies.
RUN python -m pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps && \
    python -m pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall && \
    python -m pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation && \
    python -m pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# Make MXFP8 quantizer import conditional — mxfp8_group_quantize was added
# in a newer SGLang than our pinned commit. Not needed for Qwen3-8B training.
RUN python -c "\
import pathlib; \
p = pathlib.Path('/tmp/miles/miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_mxfp8.py'); \
t = p.read_text(); \
t = t.replace( \
    'from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize', \
    'try:\\n    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize\\nexcept ImportError:\\n    mxfp8_group_quantize = None' \
); \
p.write_text(t)"

# Install Miles itself.
RUN python -m pip install -r /tmp/miles/requirements.txt && \
    python -m pip install -e /tmp/miles --no-deps && \
    cd /tmp/miles/miles/backends/megatron_utils/kernels/int4_qat && \
    python -m pip install . --no-build-isolation

# Re-pin PyTorch 2.9.1 and reinstall flash-attn + TE at the end.
# Earlier installs may have upgraded torch, breaking pre-built binary wheels.
RUN python -c "import torch; print(f'Before re-pin: PyTorch {torch.__version__}')"
RUN python -m pip install torch==2.9.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
RUN python -m pip install --force-reinstall --no-deps \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN python -m pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

# Verify torch + flash-attn ABI compatibility.
# sgl_kernel is skipped here — it requires libcuda.so.1 (GPU hardware) to import.
RUN python -c "\
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}'); \
assert torch.__version__.startswith('2.9'), f'Expected 2.9.x, got {torch.__version__}'; \
from flash_attn import flash_attn_func; print('flash-attn OK')"

ENV PYTHONPATH=/home/ray/Megatron-LM:$PYTHONPATH

WORKDIR /tmp/miles

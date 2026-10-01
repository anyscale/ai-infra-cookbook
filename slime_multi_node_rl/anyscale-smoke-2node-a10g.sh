#!/bin/bash
# Anyscale entrypoint: Qwen3-1.7B smoke test on 2 workers × 4x A10G
# Downloads model/dataset, converts weights, and runs a full RL training run.
#
# Head node (m5.2xlarge): driver only, no GPUs
# Layout (GPU workers):
#   Worker 0 (4 GPUs): [GPU 0-1: Training TP=2, PP Stage 0] [GPU 2-3: Rollout]
#   Worker 1 (4 GPUs): [GPU 0-1: Training TP=2, PP Stage 1] [GPU 2-3: Rollout]

set -ex

export PYTHONBUFFERED=16
STORAGE=/mnt/cluster_storage

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Qwen3-1.7B model architecture args
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)

# ======================== Step 1: Download model & dataset ========================

echo "=== Downloading model ==="
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ${STORAGE}/Qwen3-1.7B

echo "=== Downloading dataset ==="
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir ${STORAGE}/dapo-math-17k

# ======================== Step 2: Convert HF weights to torch_dist ========================

if [ ! -d "${STORAGE}/Qwen3-1.7B_torch_dist/iter_0000000" ]; then
  echo "=== Converting weights (HF -> torch_dist) on GPU worker ==="
  # Run conversion on a GPU worker via Ray (head node is CPU-only; Triton
  # crashes without GPU drivers).  Output goes to shared /mnt/cluster_storage.
  CONVERT_ENV_JSON='{
    "env_vars": {
      "PYTHONPATH": "/root/Megatron-LM/"
    }
  }'
  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${CONVERT_ENV_JSON}" \
    --entrypoint-num-gpus 1 \
    -- python3 /tmp/slime/tools/convert_hf_to_torch_dist.py \
      ${MODEL_ARGS[@]} \
      --no-gradient-accumulation-fusion \
      --hf-checkpoint ${STORAGE}/Qwen3-1.7B \
      --save ${STORAGE}/Qwen3-1.7B_torch_dist
else
  echo "=== Converted weights already exist, skipping ==="
fi

# ======================== Step 2.5: Patch fake_impl.py on all nodes ========================
# sgl_kernel from PyPI ships SM100-only ops; on A10G (SM86) the C++ operators
# never load, so @torch.library.register_fake(...) crashes with
# "operator ... does not exist".  This patches torch's fake_impl.py on every
# node to catch that RuntimeError gracefully.

echo "=== Patching fake_impl.py on all nodes ==="
cp ${SCRIPT_DIR}/patch_all_nodes.py ${STORAGE}/patch_all_nodes.py
ray job submit --address="http://127.0.0.1:8265" \
    -- python3 ${STORAGE}/patch_all_nodes.py

# ======================== Step 3: Run training ========================

CKPT_ARGS=(
   --hf-checkpoint ${STORAGE}/Qwen3-1.7B
   --ref-load ${STORAGE}/Qwen3-1.7B_torch_dist
   --load ${STORAGE}/Qwen3-1.7B_torch_dist
)

ROLLOUT_ARGS=(
   --prompt-data ${STORAGE}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 20
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size 64
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-attention-backend flashinfer
)

MISC_ARGS=(
   --no-gradient-accumulation-fusion
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # A10G (Ampere) — use FA2, not FA3
   --attention-backend flash
   # Metrics: loss, reward, TFLOPS, pass@k are logged by default.
   # Save to tensorboard for post-hoc analysis.
   --use-tensorboard
   --tensorboard-dir ${STORAGE}/tensorboard_logs
)

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "TENSORBOARD_DIR": "/mnt/cluster_storage/tensorboard_logs"
  }
}'

echo "=== Submitting training job ==="
# Run the driver on a GPU worker (needs CUDA for Megatron arg validation).
# Reserve 1 GPU for driver; reduce rollout to 3 (4 train + 3 rollout + 1 driver = 8).
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --entrypoint-num-gpus 1 \
   -- python3 /tmp/slime/train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 3 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

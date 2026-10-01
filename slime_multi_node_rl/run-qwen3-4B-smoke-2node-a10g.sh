#!/bin/bash

# Smoke test: Qwen3-4B on 2 nodes × 4x A10G (bare-metal)
# Layout:
#   Node 0 (4 GPUs): [GPU 0-1: Training TP=2, PP Stage 0] [GPU 2-3: Rollout]
#   Node 1 (4 GPUs): [GPU 0-1: Training TP=2, PP Stage 1] [GPU 2-3: Rollout]
# Training: 4 GPUs (TP=2 × PP=2 × DP=1), Rollout: 4 GPUs (disaggregated)

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Qwen3-4B model architecture args
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
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

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --ref-load /root/Qwen3-4B_torch_dist
   --load /root/Qwen3-4B_torch_dist
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 3
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size 32
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-smoke-2node-a10g
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-attention-backend flashinfer
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # A10G (Ampere) — use FA2, not FA3
   --attention-backend flash
)

# --- Multi-Node Ray Cluster Setup ---
# On Anyscale: skip ray start commands, the Ray cluster is managed automatically.
# For bare-metal / non-Anyscale, uncomment the following:
#
# export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
#
# On each worker node:
# ray start --address=${MASTER_ADDR}:6379 --node-ip-address ${WORKER_ADDR} --num-gpus 4 --disable-usage-stats

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /tmp/slime/train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

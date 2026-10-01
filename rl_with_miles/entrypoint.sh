#!/bin/bash
# Anyscale entrypoint: Qwen3-8B GRPO training on 2 workers × 8x H100
# Downloads model/dataset, converts weights, and runs async RL training.
#
# Head node (m5.2xlarge): driver only, no GPUs
# GPU Placement (determined by MILES using Ray Placement Groups with PACK strategy):
#   Node 1 (8x H100): Training GPUs 0-7 (TP=2, DP=8)
#   Node 2 (8x H100): Rollout GPUs 0-7 (8 SGLang engines, 1 GPU each)

set -ex

export PYTHONUNBUFFERED=1
STORAGE=/mnt/cluster_storage

# Qwen3-8B model architecture args (from scripts/models/qwen3-8B.sh)
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 4096
   --ffn-hidden-size 12288
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
   --untie-embeddings-and-output-weights
)

# ======================== Step 1: Download model & dataset ========================

echo "=== Downloading model ==="
huggingface-cli download Qwen/Qwen3-8B --local-dir ${STORAGE}/Qwen3-8B

echo "=== Downloading dataset ==="
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir ${STORAGE}/dapo-math-17k

# ======================== Step 2: Convert HF weights to torch_dist ========================

if [ ! -d "${STORAGE}/Qwen3-8B_torch_dist/iter_0000000" ]; then
  echo "=== Converting weights (HF -> torch_dist) on GPU worker ==="
  python convert_weights_remote.py \
    ${MODEL_ARGS[@]} \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint ${STORAGE}/Qwen3-8B \
    --save ${STORAGE}/Qwen3-8B_torch_dist
else
  echo "=== Converted weights already exist, skipping ==="
fi

# ======================== Step 3: Run training ========================

CKPT_ARGS=(
   --hf-checkpoint ${STORAGE}/Qwen3-8B
   --ref-load ${STORAGE}/Qwen3-8B_torch_dist
   --load ${STORAGE}/Qwen3-8B_torch_dist
   --save ${STORAGE}/Qwen3-8B_miles/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${STORAGE}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type dapo
   --reward-key score
   --num-rollout 5
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
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
)

MISC_ARGS=(
   --no-gradient-accumulation-fusion
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-tensorboard
)

echo "=== Starting training ==="
python train_remote.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

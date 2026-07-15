# Multi-Node RL Training with Slime

[Slime](https://github.com/THUDM/slime) is an RL training framework that uses Megatron-LM for distributed training with disaggregated rollout via SGLang. This example runs GRPO training of Qwen3-1.7B on **2 workers x 4x A10G** (8 GPUs, 24 GB VRAM each) using Anyscale.

## Cluster Layout

```
Head node (m5.2xlarge):  driver only, no GPUs
Worker 0 (4 GPUs):  [GPU 0-1: Training TP=2, Stage 0] [GPU 2-3: Rollout]
Worker 1 (4 GPUs):  [GPU 0-1: Training TP=2, Stage 1] [GPU 2-3: Rollout]
```

- **Training**: 4 GPUs — TP=2 x PP=2 x DP=1 (Megatron backend, PP spans workers)
- **Rollout**: 3 GPUs — disaggregated SGLang inference, 1 GPU per engine (1 GPU reserved for driver)

## Files

| File | Description |
|------|-------------|
| `job.yaml` | Anyscale job config (`m5.2xlarge` head + 2x `g5.12xlarge` workers) |
| `Dockerfile.anyscale` | Docker image with Slime, Megatron-LM, SGLang, and A10G compatibility patches |
| `anyscale-smoke-2node-a10g.sh` | Anyscale entrypoint (downloads model/data, converts weights, runs training) |
| `patch_all_nodes.py` | Runtime patches for sgl_kernel compatibility on A10G (SM86) |
| `run-qwen3-4B-smoke-2node-a10g.sh` | Bare-metal variant for Qwen3-4B (manual Ray cluster setup) |

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Quick Start

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/slime_multi_node_rl
```

Submit the job.

```bash
anyscale job submit -f job.yaml
```

The entrypoint automatically:
1. Downloads `Qwen/Qwen3-1.7B` and `zhuzilin/dapo-math-17k` to `/mnt/cluster_storage`
2. Converts HF weights to Megatron torch_dist format (on a GPU worker)
3. Patches all nodes for A10G compatibility (sgl_kernel SM86 workaround)
4. Runs GRPO training with `deepscaler` reward model

## Understanding the Example

- The [Dockerfile.anyscale](Dockerfile.anyscale) builds on `anyscale/ray:2.54.0-py312-cu129` and installs Slime with all dependencies. It downgrades PyTorch from 2.9.1 to 2.7.1 for compatibility with pre-built flash-attn and transformer_engine wheels, and includes patches for sgl_kernel (which only ships SM100 ops from PyPI, incompatible with A10G SM86).
- The entrypoint uses `ray job submit --entrypoint-num-gpus 1` to schedule weight conversion and the training driver on GPU worker nodes (the head node is CPU-only and cannot run Triton/CUDA code).
- Training uses Megatron-LM with TP=2, PP=2 across the two workers. Pipeline parallelism spans nodes via NCCL.
- Rollout uses disaggregated SGLang inference engines (1 GPU each) for generating training samples.

## A10G-Specific Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `NCCL_NVLS_ENABLE` | `0` | No NVLink on cloud A10G |
| `--attention-backend` | `flash` | FA2 only (Ampere, no FA3) |
| `--sglang-attention-backend` | `flashinfer` | For SGLang on Ampere |
| `--max-tokens-per-gpu` | `4096` | Conservative for 24 GB VRAM |
| No FP8 | — | Ampere does not support FP8 |

## Verification

A successful run shows:
- SGLang engine startup on rollout GPUs
- Cross-node NCCL init for pipeline parallelism
- Training loss values printed each step
- Weight sync between training and rollout engines

## If You Hit OOM

**Training GPUs:**
1. `--max-tokens-per-gpu` -> `2048`
2. `--rollout-max-response-len` -> `1024`
3. `--n-samples-per-prompt` -> `2` and `--global-batch-size` -> `16`
4. Add `--optimizer-cpu-offload`

**Rollout GPUs:**
1. `--sglang-mem-fraction-static` -> `0.5`
2. Add `--sglang-chunked-prefill-size 2048`

## View the Job

View the job in the [jobs tab](https://console.anyscale.com/jobs) of the Anyscale console.

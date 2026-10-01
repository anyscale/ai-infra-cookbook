# GRPO Training for Qwen3-8B with MILES

This example demonstrates reinforcement learning fine-tuning of Qwen3-8B using **Group Relative Policy Optimization (GRPO)** on the DAPO-Math-17k dataset. It uses the [MILES](https://github.com/radixark/miles) framework for distributed RL training with disaggregated rollouts on Anyscale.

The training runs on **2 nodes with 8x H100 GPUs each** (16 GPUs total), using:
- **8 GPUs for training** (TP=2, DP=8 with Megatron-LM across 2 nodes)
- **8 GPUs for rollout inference** (disaggregated SGLang engines, 8 total)

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Submit the job

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/rl_with_miles
```

Submit the job.

```bash
anyscale job submit -f job.yaml
```

The entrypoint will automatically download the model and dataset, convert weights to Megatron format, and start training. Training progress can be monitored via TensorBoard logs in `/mnt/cluster_storage/tensorboard_logs`.

## Understanding the example

- **Algorithm**: This example uses GRPO with DAPO-style asymmetric clipping (ε_low=0.2, ε_high=0.28), which is particularly effective for math reasoning tasks.
- **Dataset**: [DAPO-Math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) contains 17k integer math problems with deterministic reward signals based on answer correctness.
- **Disaggregated architecture**: Training and rollout happen on separate GPUs for maximum throughput. GPU placement is handled automatically by MILES using Ray placement groups, which uses node 1 for all training GPUs and node 2 for all rollout GPUs.
- **Weight conversion**: On the first run, HuggingFace weights are converted to Megatron-LM's `torch_dist` format. Converted weights are cached in `/mnt/cluster_storage/Qwen3-8B_torch_dist` for subsequent runs.
- **Async training**: The pipeline uses `train_async.py` which overlaps rollout generation and policy updates for better GPU utilization.
- **Ray remote wrappers**: The MILES scripts are wrapped in Ray remote functions (`convert_weights_remote.py` and `train_remote.py`) to ensure they execute on GPU worker nodes rather than the CPU-only head node. Both wrappers use `label_selector={"ray.io/accelerator-type": "H100"}` to match the accelerator type specified in `job.yaml`, ensuring placement on H100 GPU nodes. Both wrappers explicitly set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` in the subprocess environment to provide access to all 8 GPUs. Neither wrapper reserves GPUs with `num_gpus` to allow the subprocesses to manage GPU allocation internally.

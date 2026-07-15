# Ray Train + Tensor Parallelism Tutorial

A simple tutorial demonstrating how to train large language models with tensor parallelism using PyTorch native FSDP2+DTensor and Ray Train.

## Key Concepts

- **Tensor Parallelism (TP)**: Shards model weights across GPUs within a TP group
- **Data Parallelism (DP)**: Replicates the model across DP groups, each processing different data
- **2D Parallelism**: Combines TP and DP for scaling to many GPUs

## Quick Start

```bash
# 4 GPUs: 2-way tensor parallelism, 2-way data parallelism
python train.py \
    --model_name Qwen/Qwen2-7B \
    --tp_size 2 \
    --dp_size 2 \
    --num_workers 4 \
    --num_epochs 3

# 8 GPUs: 4-way tensor parallelism, 2-way data parallelism
python train.py \
    --model_name Qwen/Qwen2-7B \
    --tp_size 4 \
    --dp_size 2 \
    --num_workers 8 \
    --batch_size 2 \
    --seq_length 2048 \
    --num_epochs 3
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | HuggingFace model name | `Qwen/Qwen2-7B` |
| `--tp_size` | Tensor parallel degree | Required |
| `--dp_size` | Data parallel degree | `1` |
| `--num_workers` | Total workers (must equal tp_size * dp_size) | Required |
| `--dataset_name` | HuggingFace dataset | `wikitext` |
| `--dataset_percentage` | Percentage of dataset to use (0-100) | `10.0` |
| `--batch_size` | Per-GPU micro batch size | `1` |
| `--seq_length` | Maximum sequence length | `2048` |
| `--num_epochs` | Number of training epochs | `3` |
| `--learning_rate` | Learning rate | `1e-5` |
| `--weight_decay` | Weight decay | `0.01` |
| `--storage_path` | Checkpoint storage path | `/mnt/cluster_storage` |
| `--experiment_name` | Experiment name (auto-generated if not provided) | None |
| `--log_interval` | Logging interval (steps) | `10` |
| `--debug_steps` | Stop after N steps per epoch (0 = full epoch) | `0` |
| `--seed` | Random seed | `42` |

## Anyscale Job

```bash
anyscale job submit -f job.yaml
```

## File Structure

```
train_tensor_parallel_simple/
├── train.py      # Main training script
├── args.py       # Command line arguments
├── job.yaml      # Anyscale job config
└── README.md     # This file
```

## How 2D Parallelism Works

With `tp_size=2` and `dp_size=2` on 4 GPUs:

```
Device Mesh (2x2):
        TP Dim
      [0]  [1]
 DP   +---+---+
 Dim  | 0 | 1 |  <- TP Group 0 (same data, sharded model)
      +---+---+
      | 2 | 3 |  <- TP Group 1 (same data, sharded model)
      +---+---+
        ^   ^
       DP Groups (different data, gradient sync)
```

- **TP Groups** (rows): GPUs 0,1 and GPUs 2,3 share the same input data but have sharded model weights
- **DP Groups** (columns): GPUs 0,2 and GPUs 1,3 see different data and synchronize gradients

## Key Implementation Details

### TP-Aware Data Loading

Standard data loaders shard by `world_rank`, giving each GPU different data. With tensor parallelism, all GPUs in a TP group must see identical data. This is handled by sharding based on `dp_rank` instead:

```python
# All TP ranks in same DP group get identical batches
sampler = DistributedSampler(
    dataset,
    num_replicas=dp_size,  # NOT world_size
    rank=dp_rank,          # NOT world_rank
)
```

### Checkpointing

All workers save their model shards independently. Ray Train aggregates these into a single checkpoint that can be used for resuming training.

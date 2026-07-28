# Ray Train + Megatron-SWIFT LLM Fine-tuning Example

This example demonstrates distributed LLM fine-tuning using:
- **Ray Train**: Orchestrates distributed workers across GPUs/nodes
- **Megatron-SWIFT**: Provides efficient tensor and pipeline parallelism for training (See the [document](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html) for more details)

## Overview

The integration combines SWIFT's easy-to-use training interface with Megatron-LM's parallelism capabilities, orchestrated by Ray Train for multi-node scaling.

### Architecture

```
main() -> TorchTrainer -> train_loop() on each GPU
                          |
                     megatron_sft_main()
                          |
                     Megatron pretrain()
```

## Files

- `llm_sft_ray_train_swift.py` - Main training script with Ray Train integration
- `job.yaml` - Anyscale job configuration for cloud deployment
- `Dockerfile` - Container image with SWIFT and Megatron dependencies

## Prerequisites

The job builds a Docker image from the included `Dockerfile` at submit time. No pre-built image is needed.

## Quick Start

```bash
# Submit job to Anyscale (passes HF token for model downloads)
anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN

# Monitor logs
anyscale job logs <job-id>
```

**What this job does:**
1. **Builds** a Docker image with SWIFT and Megatron dependencies (using `Dockerfile`).
2. **Provisions** 4 GPUs (tested with 4×L4 GPUs).
3. **Runs** the distributed training script `llm_sft_ray_train_swift.py`.

## Configuration

### Parallelism Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tensor_parallel_size` | 2 | Split model layers across GPUs |
| `--pipeline_parallel_size` | 1 | Split model stages across GPUs |
| `--num_workers` | 4 | Total number of GPUs |

**Note**: `num_workers` must be divisible by `tensor_parallel_size * pipeline_parallel_size`.

Data Parallelism (DP) is automatically calculated as:
```
DP = num_workers / (TP * PP)
```

### Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Qwen/Qwen2.5-1.5B-Instruct | HuggingFace model ID |
| `--dataset` | AI-ModelScope/alpaca-gpt4-data-en#500 | Dataset (append #N for sampling) |
| `--train_iters` | 100 | Number of training iterations |
| `--micro_batch_size` | 2 | Batch size per GPU |
| `--seq_length` | 512 | Maximum sequence length |
| `--learning_rate` | 1e-5 | Learning rate |

### LoRA Settings

Enable parameter-efficient fine-tuning with LoRA:

```bash
python llm_sft_ray_train_swift.py --use_lora --lora_rank 8 --lora_alpha 32
```

## Example Configurations

### 8 GPUs with TP=2, PP=1 (DP=4)

```bash
python llm_sft_ray_train_swift.py \
    --num_workers 8 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 1 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_iters 200
```

### 8 GPUs with TP=4, PP=2 (DP=1)

For larger models requiring more parallelism:

```bash
python llm_sft_ray_train_swift.py \
    --num_workers 8 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 2 \
    --model Qwen/Qwen2.5-72B-Instruct \
    --micro_batch_size 1
```

## Supported Models

SWIFT's Megatron integration supports many HuggingFace models, including:
- Qwen2/Qwen2.5 series
- Llama 2/3 series
- Mistral/Mixtral series
- DeepSeek series

Check [SWIFT documentation](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/index.html) for the full list.

## Troubleshooting

### CUDA Out of Memory

- Reduce `--micro_batch_size`
- Reduce `--seq_length`
- Increase `--tensor_parallel_size`
- Enable LoRA with `--use_lora`

### Distributed Initialization Errors

- Ensure all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`
- For multi-node, ensure `MODELSCOPE_CACHE` points to shared storage

### Slow Data Loading

- For multi-node training, set `MODELSCOPE_CACHE` to a shared storage path
- Consider using streaming datasets for large datasets

## References

- [SWIFT Megatron Documentation](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/index.html)
- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

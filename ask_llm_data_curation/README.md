# Data Curation with Qwen LLM (Ask-LLM Approach)

This example demonstrates LLM-based data curation using the Ask-LLM methodology from the [DCLM paper](https://arxiv.org/abs/2406.11794). It uses Qwen2.5-3B-Instruct via vLLM to score text quality and filter the FineWeb-edu dataset.

## Overview

The pipeline:
1. Loads FineWeb-edu dataset from HuggingFace
2. Applies Ask-LLM quality scoring using Qwen2.5-3B-Instruct
3. Filters samples based on quality threshold (P(Yes) > 0.5)
4. Writes curated data to parquet

## Ask-LLM Methodology

Based on [How to Train Data-Efficient LLMs](https://arxiv.org/abs/2402.09668), the Ask-LLM approach:
- Prompts an LLM to judge if text is suitable for training
- Uses the softmax probability of "Yes" as a quality score
- Enables nuanced quality filtering that outperforms simple heuristics

## Configuration

Edit `main.py` to adjust:
- `num_samples_to_process`: Number of samples to process (default: 100,000)
- `num_gpus`: GPU count matching `job.yaml` (default: 8)
- `quality_threshold`: Minimum quality score for filtering (default: 0.5)

## Running the Job

```bash
# Set your HuggingFace token
export HF_TOKEN="your_hf_token"

# Submit the job
anyscale job submit job.yaml
```

## Output

Curated parquet files are written to:
```
/mnt/shared_storage/fineweb_curated/{timestamp}/
```

## Scaling

To scale up:
1. Increase `num_gpus` in `main.py`
2. Update `min_nodes`/`max_nodes` in `job.yaml`
3. Increase `num_samples_to_process` for larger datasets

For production (10M+ samples), consider:
- 64 GPUs with `g5.xlarge` instances
- Increase `batch_size` and `max_concurrent_batches`

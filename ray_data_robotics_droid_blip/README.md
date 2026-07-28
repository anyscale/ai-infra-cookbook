# DROID → BLIP Captioning Pipeline

A [Ray Data](https://docs.ray.io/en/latest/data/overview.html) pipeline that reads the [DROID Raw 1.0.1](https://droid-dataset.github.io/) robotics dataset from a public S3 bucket, generates image captions for each wrist-camera frame using [BLIP-large](https://huggingface.co/Salesforce/blip-image-captioning-large) on GPU, and writes annotated parquet partitioned by episode.

Demonstrates **heterogeneous Ray Data pipelines** — CPU workers stream data from S3 while GPU workers run model inference, with Ray Data managing backpressure automatically.

## Files

| File | Description |
|---|---|
| `pipeline.py` | Main entry point — pipeline wiring, CPU/GPU stages, and config knobs |
| `job.yaml` | Anyscale job config — compute, env vars, T4 GPUs |
| `pyproject.toml` | Dependencies (managed by `uv`) |

**Note:** The pipeline automatically downloads the DROID episode manifest (`episodes_droid_v1.0.1_s3.parquet`) from S3 on first run.

## Usage

```bash
# Install dependencies
uv sync

# Run on an existing Ray cluster
uv run python pipeline.py

# Smoke test (limit to N episodes)
# Set NUM_EPISODES in pipeline.py, then run as above
```

### Run on Anyscale

Submit as a job using the included `job.yaml`, which provisions T4 GPU workers:

```bash
uv run anyscale job submit -f job.yaml

# Override GPU concurrency or batch size via env vars
uv run anyscale job submit -f job.yaml --env GPU_STAGE_CONCURRENCY=8 --env GPU_BATCH_SIZE=64
```

## Configuration

Key settings in `pipeline.py`:

| Setting | Default | Description |
|---|---|---|
| `NUM_EPISODES` | `1000` | Limit episodes for testing (set to `None` for all) |
| `MODEL_NAME` | `Salesforce/blip-image-captioning-large` | HuggingFace model ID |
| `CPU_LOADER_NUM_CPUS` | `0.5` | Fractional CPUs per data-loading worker |
| `GPU_BATCH_SIZE` | `32` | Frames per GPU batch (env: `GPU_BATCH_SIZE`) |
| `GPU_CONCURRENCY` | `2` | Number of GPU actors (env: `GPU_STAGE_CONCURRENCY`) |
| `OUTPUT` | `/mnt/cluster_storage/droid-annotated` | Parquet output path |

# Distributed VLA Fine-Tuning with Ray

This example fine-tunes the [PI0.5](https://www.physicalintelligence.company/blog/pi05) Vision-Language-Action (VLA) model on a [LeRobot](https://github.com/huggingface/lerobot) robotics dataset stored in S3, using [Ray Data](https://docs.ray.io/en/latest/data/data.html) for distributed preprocessing and [Ray Train](https://docs.ray.io/en/latest/train/train.html) for distributed GPU training on [Anyscale](https://anyscale.com).

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Clone the example

```bash
git clone https://github.com/anyscale/examples.git
cd examples/vla_fine_tuning
```

## Accept the PaliGemma license

PI0.5 uses [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) as its vision backbone. You must accept the license before the weights can be downloaded.

1. Visit [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) and accept the license agreement.
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Submit the job

Pass your Hugging Face token via `--env`:

```bash
anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN
```

The job runs 2 training epochs on the `xvla-soft-fold` dataset across 8× L40S GPUs and writes checkpoints to `/mnt/cluster_storage/ray_train_runs/pi05_xvla_soft_fold`.

## Interactive notebook

For a step-by-step walkthrough, open **[vla.ipynb](https://github.com/anyscale/examples/blob/main/vla_fine_tuning/vla.ipynb)** in an Anyscale workspace and run cells top-to-bottom.

## Understanding the example

Ray separates the two distinct compute workloads in VLA training:

```
+-----------+       +------------+       +-----------+
| S3 Bucket | ----> | Ray Data   | ----> | Ray Train |
| (LeRobot  |       | (CPU pool) |       | (N GPUs)  |
| mp4+pqt)  |       |            |       |           |
+-----------+       +------------+       +-----------+
                     |                    |
                     | - read parquet     | - load PI0.5
                     | - decode mp4       | - freeze backbone
                     | - rename cameras   | - train action heads
                     | - transpose HWC    | - BF16 mixed-precision
                     |   -> CHW float32   | - gradient accum
                     | - normalise /255   | - checkpoint & resume
                     | - stream batches   |
                     +--------------------+---------------------
```

- **[lerobot_datasource.py](https://github.com/anyscale/examples/blob/main/vla_fine_tuning/lerobot_datasource.py)** is a custom Ray Data datasource that streams LeRobot v3 datasets (parquet metadata + AV1-encoded mp4 video) directly from S3, decoding video frames on CPU workers.
- **[vla.py](https://github.com/anyscale/examples/blob/main/vla_fine_tuning/vla.py)** builds the Ray Data preprocessing pipeline (camera renaming, HWC→CHW transposition, /255 normalisation) and launches distributed training with `TorchTrainer`.
- **[util.py](https://github.com/anyscale/examples/blob/main/vla_fine_tuning/util.py)** contains model loading, checkpointing, collation, and training step helpers. PI0.5 is loaded in BF16 with only the four action/time projection heads unfrozen — the 3B-parameter PaliGemma backbone stays frozen, dramatically reducing memory and compute.
- The LR schedule uses linear warmup (10%) followed by cosine decay, computed from the actual number of training rows so the schedule is correct whether you use a row limit (`LIMIT_ROWS`) or train on the full dataset.
- Fault tolerance is built in: if a worker dies, Ray restarts from the last checkpoint automatically.
- PI0.5 requires approximately 28 GB of VRAM. The default configuration targets **L40S** GPUs (48 GB).
- By default `LIMIT_ROWS = 2000` in `vla.py` caps each epoch at 2,000 rows for a quick smoke test. Set it to `None` to train on the full dataset (~2.8M frames).

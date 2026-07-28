"""
DROID → BLIP Captioning Pipeline (Ray Data)
============================================

This script demonstrates a **heterogeneous Ray Data pipeline** that reads
video files from a public S3 bucket, decodes the first frame of each,
generates an image caption using a vision-language model on GPU, and writes
the results to Parquet.

The dataset is DROID Raw 1.0.1 — a large-scale robotics manipulation
dataset with ~59K episodes across 13 research labs.

Key Ray Data concepts demonstrated
-----------------------------------
1. **read_binary_files** — parallel I/O from S3, one file per task.
2. **map (stateless)** — lightweight per-row transforms on CPU workers.
3. **map_batches (stateful, GPU)** — actor-pool pattern for model inference.
   The model loads once per actor in __init__; __call__ runs per batch.
4. **Streaming execution** — all stages run concurrently with automatic
   backpressure. No stage waits for the previous one to finish.

Pipeline
--------
    ┌─────────────────────────────────────────────────────┐
    │  read_binary_files(mp4_paths)          [CPU, I/O]   │
    │  Download each MP4 from S3 in parallel.             │
    ├─────────────────────────────────────────────────────┤
    │  map(attach_meta)                      [CPU]        │
    │  Look up episode uuid + task from manifest.         │
    ├─────────────────────────────────────────────────────┤
    │  map(decode_first_frame)               [CPU]        │
    │  Decode the first video frame with PyAV.            │
    ├─────────────────────────────────────────────────────┤
    │  map_batches(BlipCaptionStage)         [GPU]        │
    │  BLIP-large fp16 captioning, batched across GPUs.   │
    ├─────────────────────────────────────────────────────┤
    │  drop_columns(["frame"])                            │
    │  Remove the raw image; keep uuid, task, caption.    │
    ├─────────────────────────────────────────────────────┤
    │  write_parquet(OUTPUT)                              │
    │  Materialize the full pipeline and write results.   │
    └─────────────────────────────────────────────────────┘

Usage
-----
    # Smoke test with 100 episodes on 4 GPUs:
    GPU_STAGE_CONCURRENCY=4 uv run python pipeline.py

    # Full dataset:
    NUM_EPISODES=None  (edit below)
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import ray
import ray.data
import torch
from PIL import Image
from pyarrow.fs import S3FileSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# -- Dataset --
MANIFEST = "episodes_droid_v1.0.1_s3.parquet"
DATASET_BUCKET = "s3://anyscale-public-droid-dataset"
DATASET_PREFIX = "droid/1.0.1"

# -- Output --
OUTPUT = "/mnt/cluster_storage/droid-annotated"

# -- Limits --
# Set to an integer to cap the number of episodes (useful for smoke tests).
# Set to None to process the full dataset.
NUM_EPISODES: int | None = None

# -- Model --
MODEL_NAME: str = "Salesforce/blip-image-captioning-large"

# -- Runtime environment --
# Example: propagate a custom env var to all Ray workers.
# See: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
CUSTOM_ENV_VAR: str = "hello"

# -- CPU stage resources --
# Each CPU map worker needs little CPU since the work is I/O-bound
# (downloading from S3 + video decoding). Setting < 1 lets Ray schedule
# more workers per node, increasing download parallelism.
# See: https://docs.ray.io/en/latest/data/transforming-data.html#configuring-parallelism
CPU_LOADER_NUM_CPUS: float = 0.5

# -- GPU stage resources --
# Number of BLIP actors. Each actor claims one GPU and processes batches
# of images. Override at runtime: GPU_STAGE_CONCURRENCY=8 uv run ...
# See: https://docs.ray.io/en/latest/data/transforming-data.html#stateful-transforms
GPU_CONCURRENCY: int = int(os.environ.get("GPU_STAGE_CONCURRENCY", "2"))
GPU_WORKER_NUM_GPUS: float = 1.0

# Rows per GPU batch. Increase until VRAM is ~80% utilized.
# BLIP-large fp16 uses ~1.5 GB model weight; batch of 32 images fits
# comfortably on most GPUs.
GPU_BATCH_SIZE: int = int(os.environ.get("GPU_BATCH_SIZE", "32"))


# ============================================================================
# Stage 1 (CPU): Decode first video frame
# ============================================================================
# This is a *stateless* map function — Ray Data calls it once per row.
# It receives the raw MP4 bytes downloaded by read_binary_files and
# returns a single decoded RGB frame (HWC uint8 numpy array).


def decode_first_frame(row: dict[str, Any]) -> dict[str, Any]:
    """Decode the first RGB frame from in-memory MP4 bytes."""
    buf = io.BytesIO(row["bytes"])
    frame = None
    with av.open(buf, mode="r") as container:
        for f in container.decode(video=0):
            frame = f.to_ndarray(format="rgb24")
            break
    row["frame"] = frame
    del row["bytes"]
    return row


# ============================================================================
# Stage 2 (GPU): BLIP image captioning
# ============================================================================
# This is a *stateful* map class — Ray Data creates a pool of actors
# (one per GPU). __init__ runs once to load the model; __call__ is
# invoked on each batch. This avoids reloading the model per batch.
#
# See: https://docs.ray.io/en/latest/data/transforming-data.html#stateful-setup-with-actors


class BlipCaptionStage:
    """BLIP-large image captioning actor for Ray Data map_batches."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        log.info("Loading BLIP model: %s …", model_name)
        t0 = time.monotonic()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = (
            BlipForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

        log.info(
            "BLIP loaded on %s in %.1fs  (fp16, %d params).",
            self.device,
            time.monotonic() - t0,
            sum(p.numel() for p in self.model.parameters()),
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Caption a batch of images.

        Ray Data calls this with a dict of column → list[value],
        not one row at a time. This enables efficient batched GPU inference.
        """
        images_raw: list[np.ndarray] = batch["frame"]
        n = len(images_raw)
        t0 = time.monotonic()

        pil_images = [Image.fromarray(arr, mode="RGB") for arr in images_raw]

        with torch.inference_mode():
            inputs = self.processor(
                images=pil_images, return_tensors="pt", padding=True,
            ).to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        batch["caption"] = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        elapsed = time.monotonic() - t0
        log.info(
            "GPU batch: %d frames in %.2fs  (%.1f frames/s)",
            n, elapsed, n / elapsed if elapsed > 0 else 0.0,
        )
        return batch


# ============================================================================
# Prepare: read manifest and resolve S3 paths
# ============================================================================
# The manifest is a small Parquet file (~85K rows) listing every episode.
# We read it directly from S3 on the driver to build:
#   1. A list of MP4 paths to pass to read_binary_files.
#   2. A path → {uuid, task} lookup so we can attach metadata later.

# Anonymous S3 filesystem for the public DROID bucket.
s3_fs = S3FileSystem(anonymous=True, region="us-east-2")

# Construct S3 path (remove s3:// prefix for PyArrow filesystem)
bucket_name = DATASET_BUCKET.replace("s3://", "")
manifest_path = f"{bucket_name}/{DATASET_PREFIX}/{MANIFEST}"
log.info("Reading manifest from S3: s3://%s", manifest_path)
manifest = pq.read_table(manifest_path, filesystem=s3_fs)
s3_base = f"{DATASET_BUCKET}/{DATASET_PREFIX}/"

mp4_paths: list[str] = []
path_to_meta: dict[str, dict[str, str]] = {}

for i in range(len(manifest)):
    rel = manifest.column("ext1_mp4_path")[i].as_py()
    if not rel:
        continue
    full = rel if "://" in rel else s3_base + rel
    mp4_paths.append(full)
    path_to_meta[full] = {
        "uuid": manifest.column("uuid")[i].as_py(),
        "task": manifest.column("task")[i].as_py(),
    }

if NUM_EPISODES is not None:
    mp4_paths = mp4_paths[:NUM_EPISODES]

log.info("Processing %d episodes (of %d total).", len(mp4_paths), len(manifest))


# ============================================================================
# Connect to the Ray cluster
# ============================================================================
# ray.init connects to an existing cluster (started by Anyscale or `ray start`).
# runtime_env propagates environment variables to every worker.

ray.init(
    address="auto",
    ignore_reinit_error=True,
    runtime_env={"env_vars": {"CUSTOM_ENV_VAR": CUSTOM_ENV_VAR}},
)

# Put the metadata lookup in the Ray object store so all workers can
# access it without serializing it per task.
meta_ref = ray.put(path_to_meta)


def attach_meta(row: dict[str, Any]) -> dict[str, Any]:
    """Look up episode uuid and task from the manifest by file path."""
    meta = ray.get(meta_ref)
    info = meta.get(row["path"], {})
    row["uuid"] = info.get("uuid", "")
    row["task"] = info.get("task", "")
    del row["path"]
    return row


# ============================================================================
# Build and run the pipeline
# ============================================================================
# Everything below is lazy — no work happens until write_parquet() is called.
# Ray Data then executes all stages as a streaming pipeline with automatic
# backpressure between CPU and GPU stages.

ds = (
    # --- Read: download MP4 files from S3 in parallel ---
    # Each file becomes one row with columns: bytes, path.
    # See: https://docs.ray.io/en/latest/data/loading-data.html
    ray.data.read_binary_files(
        mp4_paths,
        filesystem=s3_fs,
        include_paths=True,
        ignore_missing_paths=True,
    )
    # --- CPU: attach episode metadata from manifest ---
    .map(attach_meta)
    # --- CPU: decode first video frame from MP4 bytes ---
    .map(decode_first_frame, num_cpus=CPU_LOADER_NUM_CPUS)
    # --- GPU: generate captions with BLIP-large ---
    # ActorPoolStrategy creates a fixed pool of GPU actors.
    # See: https://docs.ray.io/en/latest/data/transforming-data.html#computing-over-batches
    .map_batches(
        BlipCaptionStage,
        batch_size=GPU_BATCH_SIZE,
        num_gpus=GPU_WORKER_NUM_GPUS,
        concurrency=GPU_CONCURRENCY,
    )
    # --- Drop the raw image; keep only uuid, task, caption ---
    .drop_columns(["frame"])
)

# write_parquet triggers execution of the full pipeline.
# Everything above is lazy — this call materializes the DAG.
ds.write_parquet(OUTPUT)

log.info("Done. Output written to %s", OUTPUT)

"""
Streaming Video Curation with Ray Data (HuggingFace-native)
===========================================================

Curates raw video collections into clean, semantically-annotated clip
datasets with a single streaming Ray Data pipeline. Videos are read
directly from HuggingFace (no prefetch step), CPU and GPU stages run
concurrently, and curated Parquet is written to shared storage.

ARCHITECTURE
------------

    +-----------+       +-------------------+       +-----------+
    | HF        | ----> | Ray Data          | ----> | Curated   |
    | parquet   |       | Streaming         |       | Parquet   |
    | (mp4      |       | (CPU + GPU)       |       | (metadata |
    |  bytes)   |       |                   |       |  + embed) |
    +-----------+       +-------------------+       +-----------+
                         |
                         | CPU stages (autoscaling):
                         |   0. HF parquet stream (read_parquet,
                         |      pinned to cpu_only workers)
                         |   1. Fused CPU xform: scene detect +
                         |      quality filter + keyframe extract
                         |      in ONE decord open per video
                         |   3. Safety filter (drop is_safe=False)
                         |   4. CLIP ViT-B/32 embeddings
                         |      (CPU actor pool, 2:1 vs GPUs)
                         |
                         | GPU stages (1x A10G per replica):
                         |   2. VLM: Qwen2.5-VL-3B via vLLM
                         |      one replica per GPU (bottleneck,
                         |      ~100x heavier than CLIP per call)
                         |
                         | <- backpressure keeps all resources busy ->
                         +---------------------------------------------

DATA SOURCE
-----------
HuggingFaceFV/finevideo -- 1357 parquet shards, ~33 mp4 videos each. Ray Data
reads only the shards needed for `--num-videos`, streams block-by-block, and
authenticates with HF_TOKEN (required env var).

END-TO-END DATA FLOW
--------------------
  HF parquet     -> read_parquet                         (stream)
  {mp4: bytes}   -> flat_map(process_video_bytes)        (1 video -> ~10 clips)
  clip rows      -> vLLM (Qwen2.5-VL, one replica/GPU)   (enrich + classify)
  clip rows      -> filter(is_safe)                      (drop unsafe)
  clip rows      -> map_batches(CLIPEmbedder, CPU)       (attach embedding)
  clip rows      -> write_parquet                        (to shared storage)

Usage:
  # Anyscale job (cluster shape from job.yaml) on 20 videos:
  anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN --env NUM_VIDEOS=20

  # To run the job on the full dataset, simply omit the `NUM_VIDEOS` flag:
  anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import ray
from huggingface_hub import HfFileSystem
from ray.data.llm import (
    PrepareImageStageConfig,
    build_processor,
    vLLMEngineProcessorConfig,
)

from stages import (
    CLIP_BATCH_SIZE,
    CLIPEmbedder,
    MAX_MODEL_LEN,
    MODEL_SOURCE,
    VLM_BATCH_SIZE,
    process_video_bytes,
    vlm_postprocess,
    vlm_preprocess,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("video_curation")

# ===========================================================================
# Helpers
# ===========================================================================

HF_DATA_URL = "hf://datasets/HuggingFaceFV/finevideo"

# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Streaming video curation pipeline.")
    parser.add_argument(
        "--output-dir", default="/mnt/shared_storage/finevideo/curated"
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help=(
            "Cap the number of input videos. If omitted, the entire HF "
            "dataset is processed. Values larger than the dataset size are "
            "silently clamped to the available rows by ds.limit() — no failure."
        ),
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN env var is required to access HuggingFaceFV/finevideo")
        sys.exit(1)

    os.environ.setdefault("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", "0.5")
    ray.init(ignore_reinit_error=True)
    ray.data.DataContext.get_current().enable_progress_bars = True

    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus < 2:
        logger.error(
            f"Need at least 2 GPUs (found {total_gpus}); "
            "the streaming pipeline runs VLM and CLIP concurrently."
        )
        sys.exit(1)

    num_vlm = total_gpus
    num_clip_cpu = max(8, num_vlm * 2)
    hffs = HfFileSystem(token=hf_token)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = os.path.join(args.output_dir, f"streaming_{timestamp}")

    logger.info("=" * 60)
    logger.info("VIDEO CURATION PIPELINE (STREAMING)")
    logger.info("=" * 60)
    logger.info(f"  Source:     {HF_DATA_URL}")
    logger.info(
        f"  Videos:     {'ALL (full dataset)' if args.num_videos is None else args.num_videos}"
    )
    logger.info(f"  GPUs:       {num_vlm} VLM (1 per GPU)")
    logger.info(f"  CPU actors: {num_clip_cpu} CLIP")
    logger.info(f"  Output:     {output_path}")
    logger.info("=" * 60)

    pipeline_start = time.time()

    # ======================================================================
    # Stage 0 — HF parquet stream
    #
    # INPUT:  hf://datasets/HuggingFaceFV/finevideo  (dataset root).
    #         Ray Data lists the parquet shards directly from HF;
    #         `file_extensions=["parquet"]` filters out README/config files.
    # OUTPUT: {mp4: bytes} rows, one per video (1 shard ≈ 33 rows ≈ 500MB).
    # Only the `mp4` column is projected; `.limit()` bounds the read to the
    # shards needed for num_videos under the streaming executor.
    # ======================================================================
    # The read is pinned to cpu_only workers so multi-MB mp4 blobs never share
    # RAM with vLLM engines on GPU nodes; downstream stages are unpinned so
    # Ray Data can co-locate decode and embedding with their consumers.
    ds = ray.data.read_parquet(
        HF_DATA_URL,
        columns=["mp4"],
        file_extensions=["parquet"],
        filesystem=hffs,
        ray_remote_args={"label_selector": {"cpu_only": "true"}},
    )
    # ds.limit(N) returns at most N rows; if N exceeds the dataset's actual
    # row count Ray Data simply yields everything available, so a too-large
    # --num-videos is a no-op rather than a failure.
    if args.num_videos is not None:
        ds = ds.limit(args.num_videos)

    # ======================================================================
    # Stage 1 — Fused CPU transform (process_video_bytes)
    #
    # INPUT:  {mp4: bytes}                                       (1 row / video)
    # OUTPUT: {video_id, clip_index, start/end_frame, start/end_sec,
    #          clip_duration_sec, clip_num_frames, fps, width, height,
    #          brightness, contrast, sharpness, motion_score,
    #          keyframe_bytes,              # 384x384 JPEG for VLM
    #          keyframe_bytes_list}         # list of 3 224x224 JPEGs for CLIP
    # Fan-out: 1 video  ->  ~10 quality-passing clip rows (average on FineVideo).
    # ======================================================================
    ds = ds.flat_map(
        process_video_bytes,
        num_cpus=1,
        compute=ray.data.TaskPoolStrategy(size=36),
    )

    # ======================================================================
    # Stage 2 — VLM semantic analysis (Qwen2.5-VL-3B via vLLM, GPU)
    #
    # INPUT:  clip row with {keyframe_bytes, ...passthrough}
    # OUTPUT: same row + {category, is_safe, vlm_description, vlm_quality}
    # Fan-out: 1:1.
    #
    # Engine sizing for A10G-24GB:
    #   - `gpu_memory_utilization=0.85` leaves ~3 GiB headroom so a
    #     restarted engine can reclaim memory if one worker dies.
    #   - `max_num_seqs=256` + `max_num_batched_tokens=8192` give vLLM
    #     headroom for continuous batching.
    # Ray Data layer:
    #   - `batch_size=64` (VLM_BATCH_SIZE in stages.py) is what the actor
    #     dispatcher sends per call; vLLM continuous-batches internally.
    #   - `max_concurrent_batches=64` keeps each actor's intake queue deep
    #     so engines never starve between dispatcher round-trips.
    # CUDA graphs (enforce_eager=False) amortize the ~100s capture cost
    # only for large runs; sub-200-clip runs stay eager.
    # ======================================================================
    estimated_clips = (args.num_videos * 10) if args.num_videos is not None else 10**6
    use_eager = estimated_clips < 200
    vlm_processor = build_processor(
        vLLMEngineProcessorConfig(
            model_source=MODEL_SOURCE,
            engine_kwargs={
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": 0.85,
                "enforce_eager": use_eager,
                "distributed_executor_backend": "mp",
                "max_num_seqs": 256,
                "max_num_batched_tokens": 8192,
            },
            batch_size=VLM_BATCH_SIZE,
            max_concurrent_batches=64,
            accelerator_type="A10G",
            concurrency=num_vlm,
            prepare_image_stage=PrepareImageStageConfig(enabled=True),
            should_continue_on_error=True,
        ),
        preprocess=vlm_preprocess,
        postprocess=vlm_postprocess,
    )
    ds = vlm_processor(ds)

    # ======================================================================
    # Stage 3 — Safety filter
    #
    # INPUT:  clip rows from the VLM stage
    # OUTPUT: only rows with is_safe=True
    # Fan-out: N -> ~N (drops a small fraction flagged unsafe by the VLM).
    # ======================================================================
    ds = ds.filter(lambda row: row.get("is_safe", True))

    # ======================================================================
    # Stage 4 — CLIP embeddings (CPU actor pool)
    #
    # INPUT:  clip row with {keyframe_bytes_list: list[3 x 224x224 JPEGs], ...}
    # OUTPUT: same row + {embedding: np.ndarray(512)}; keyframe_bytes_list
    #         is dropped after use so rows flowing to write are text + embed.
    # Fan-out: 1:1.
    # ======================================================================
    ds = ds.map_batches(
        CLIPEmbedder,
        batch_size=CLIP_BATCH_SIZE,
        num_cpus=2,
        compute=ray.data.ActorPoolStrategy(size=num_clip_cpu),
    )

    # ======================================================================
    # Stage 5 — Terminal write (triggers execution of the streaming pipeline)
    #
    # INPUT:  clip rows with {category, is_safe, vlm_description, vlm_quality,
    #          embedding, ...passthrough}
    # OUTPUT: none (writes to shared storage)
    # ======================================================================
    ds.write_parquet(output_path)
    total_time = time.time() - pipeline_start

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Time:         {total_time:.1f}s")
    logger.info(f"  Input videos: {'ALL' if args.num_videos is None else args.num_videos}")
    logger.info(f"  Output:       {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

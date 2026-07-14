"""The same video captioning pipeline, rebuilt on Ray Data.

The DAG is identical to caption_ray_core.py — read → CPU decode → GPU caption
→ write — but every piece of orchestration hand-rolled there (in-flight caps,
batching, round-robin dispatch, partial-batch draining, backpressure, fault
tolerance) is now the framework's job:

    read_parquet(...)                # streamed, pinned to cpu_only nodes
        .flat_map(decode_row)        # CPU: decode + uniform keyframe sample
        |> vLLMEngineProcessor       # GPU: Qwen3-VL, one replica per GPU
        .write_parquet(...)          # terminal op triggers the stream

Ray Data streams blocks between stages with automatic backpressure, so the CPU
decode stage and the GPU caption stage run concurrently and each stays busy.
Compare report.json from this run against the Ray Core run: same hardware, far
less code — how do throughput and GPU/CPU utilization compare?

Usage:
    python caption_ray_data.py --num-videos 500
"""

import argparse
import logging
import os
import sys
import time

import ray
from huggingface_hub import HfFileSystem
from ray.data.llm import (
    PrepareImageStageConfig,
    build_processor,
    vLLMEngineProcessorConfig,
)

from utils import (
    MAX_MODEL_LEN,
    MODEL_SOURCE,
    NUM_KEYFRAMES,
    VLM_BATCH_SIZE,
    UtilizationMonitor,
    decode_row,
    default_output,
    filesystem_and_path,
    vlm_postprocess,
    vlm_preprocess,
    wait_for_gpus,
    worker_setup,
    write_report,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("video_captioning")


def main():
    parser = argparse.ArgumentParser(description="Video captioning on Ray Data.")
    parser.add_argument("--input", default="hf://datasets/HuggingFaceFV/finevideo")
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-videos", type=int, default=None)
    args = parser.parse_args()
    output = args.output or default_output("ray_data")

    hf_token = os.environ.get("HF_TOKEN")
    if args.input.startswith("hf://") and not hf_token:
        logger.error("HF_TOKEN is required to read %s", args.input)
        sys.exit(1)

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"worker_process_setup_hook": worker_setup},
    )
    num_gpus = wait_for_gpus(min_gpus=int(os.environ.get("EXPECTED_GPUS", "8")))
    if num_gpus < 1:
        logger.error("No GPUs joined the cluster within the timeout.")
        sys.exit(1)
    logger.info("input=%s  model=%s  GPUs=%d  output=%s",
                args.input, MODEL_SOURCE, num_gpus, output)

    # ---- Stage 0: streamed parquet read, pinned to CPU-only nodes ----------
    # Pinning the multi-MB mp4 read to cpu_only-labeled workers keeps big blobs
    # off the GPU nodes; downstream stages are unpinned.
    read_kwargs = dict(
        columns=["mp4"],
        file_extensions=["parquet"],
        ray_remote_args={"label_selector": {"cpu_only": "true"}},
    )
    input_path = args.input
    if args.input.startswith("hf://"):
        read_kwargs["filesystem"] = HfFileSystem(token=hf_token)
    elif "://" in args.input:
        read_kwargs["filesystem"], input_path = filesystem_and_path(args.input)
    ds = ray.data.read_parquet(input_path, **read_kwargs)
    if args.num_videos is not None:
        ds = ds.limit(args.num_videos)

    # ---- Stage 1: CPU decode + keyframe sample (1 video -> 0/1 rows) -------
    ds = ds.flat_map(decode_row, num_cpus=1)

    # ---- Stage 2: GPU VLM caption (Qwen3-VL, one engine per GPU) -----------
    config_kwargs = dict(
        model_source=MODEL_SOURCE,
        engine_kwargs={
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": 0.90,
            "limit_mm_per_prompt": {"image": NUM_KEYFRAMES},
            "mm_processor_cache_gb": 0,  # inputs are unique; skip the cache
            "trust_remote_code": True,
            # Small runs stay eager; large runs amortize CUDA-graph capture.
            "enforce_eager": (args.num_videos or 10**6) < 200,
        },
        batch_size=VLM_BATCH_SIZE,
        concurrency=num_gpus,
        prepare_image_stage=PrepareImageStageConfig(enabled=True),
        should_continue_on_error=True,
    )
    # Pin engine replicas to a specific GPU pool when the label is present;
    # set ACCELERATOR_TYPE="" for a mixed-GPU cluster.
    accelerator = os.environ.get("ACCELERATOR_TYPE", "RTX-PRO-6000")
    if accelerator:
        config_kwargs["accelerator_type"] = accelerator

    vlm_processor = build_processor(
        vLLMEngineProcessorConfig(**config_kwargs),
        preprocess=vlm_preprocess,
        postprocess=vlm_postprocess,
    )
    ds = vlm_processor(ds)

    monitor = UtilizationMonitor().start()
    t0 = time.time()

    # ---- Stage 3: terminal write (this is what actually runs the stream) ---
    # try_create_dir=False skips the pre-write existence check, which some
    # object stores reject (they have no real directories).
    if "://" in output:
        write_fs, write_path = filesystem_and_path(output)
        ds.write_parquet(write_path, filesystem=write_fs, try_create_dir=False)
    else:
        write_fs, write_path = None, output
        ds.write_parquet(write_path)

    wall = time.time() - t0
    utilization = monitor.stop()

    # Read the captions back (small: metadata + text, no images) to tally exact
    # counts and total video-seconds for the report.
    out_ds = ray.data.read_parquet(write_path, filesystem=write_fs)
    write_report(
        pipeline="ray_data",
        num_gpus=num_gpus,
        num_captions=out_ds.count(),
        video_seconds=float(out_ds.sum("duration_sec") or 0.0),
        wall_seconds=wall,
        utilization=utilization,
        output_uri=output,
    )


if __name__ == "__main__":
    main()

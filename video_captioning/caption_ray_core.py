"""Video captioning hand-built on Ray Core — the baseline half of the example.

The pipeline is read → CPU decode → GPU caption → write. Here every piece of
orchestration is written by hand out of Ray Core primitives:

    shards ──(read_and_decode_shard tasks)──► {keyframes, metadata}   CPU pool
           ──(CaptionActor.caption)──────────► {caption, metadata}    one actor / GPU
           ──(pyarrow write)──────────────────► parquet parts

The driver loop below does its own backpressure (capping in-flight tasks per
stage), batches decoded rows into VLM-sized requests, round-robins them across
the GPU actor pool, and drains the final partial batch. It also has no fault
tolerance — a dead task or actor fails the run. caption_ray_data.py expresses
the same DAG in a few Ray Data operators and gets all of that for free; that
contrast is the point of the example.

Usage:
    python caption_ray_core.py --num-videos 500
"""

import argparse
import logging
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import ray
from huggingface_hub import HfFileSystem

from utils import (
    MAX_MODEL_LEN,
    MODEL_SOURCE,
    NUM_KEYFRAMES,
    OUTPUT_COLUMNS,
    VLM_BATCH_SIZE,
    UtilizationMonitor,
    build_messages_b64,
    clean_caption,
    decode_and_sample,
    default_output,
    filesystem_and_path,
    sampling_params,
    wait_for_gpus,
    worker_setup,
    write_report,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("video_captioning")


# ---------------------------------------------------------------------------
# Stage tasks / actors
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=1)
def read_and_decode_shard(
    path: str, hf_token: Optional[str], is_hf: bool
) -> List[Dict[str, Any]]:
    """Read one parquet shard's mp4 column and decode each clip to keyframes.

    Read and decode are fused into one task on purpose: the task returns only
    the compact keyframe payloads (a few MB), never the multi-MB mp4 blobs, so
    the driver's scheduling loop never pulls raw video into head-node memory.
    """
    if is_hf:
        fs = HfFileSystem(token=hf_token)
        with fs.open(path, "rb") as f:
            table = pq.read_table(f, columns=["mp4"])
    else:
        fs, rel = filesystem_and_path(path)
        table = pq.read_table(rel, filesystem=fs, columns=["mp4"])

    decoded = []
    for mp4 in table.column("mp4").to_pylist():
        payload = decode_and_sample(mp4)
        if payload is not None:
            decoded.append(payload)
    return decoded


@ray.remote(num_gpus=1)
class CaptionActor:
    """GPU stage: a persistent vLLM engine, one per GPU. No tensor parallelism —
    Qwen3-VL-8B fits comfortably on a single 96 GB GPU."""

    def __init__(self):
        from vllm import LLM

        worker_setup()
        self.llm = LLM(
            model=MODEL_SOURCE,
            max_model_len=MAX_MODEL_LEN,
            limit_mm_per_prompt={"image": NUM_KEYFRAMES},
            gpu_memory_utilization=0.90,
            mm_processor_cache_gb=0,  # inputs are unique; skip the cache overhead
            trust_remote_code=True,
        )

    def caption(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from vllm import SamplingParams

        conversations = [build_messages_b64(list(r["keyframes"])) for r in batch]
        outputs = self.llm.chat(conversations, SamplingParams(**sampling_params()), use_tqdm=False)
        results = []
        for row, out in zip(batch, outputs):
            result = {c: row[c] for c in OUTPUT_COLUMNS if c != "caption"}
            result["caption"] = clean_caption(out.outputs[0].text)
            results.append(result)
        return results

    def ready(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Input enumeration + output writing
# ---------------------------------------------------------------------------


def enumerate_shards(input_uri: str, hf_token: Optional[str]) -> tuple:
    """Return (list_of_shard_paths, is_hf)."""
    import pyarrow.fs

    if input_uri.startswith("hf://"):
        fs = HfFileSystem(token=hf_token)
        paths = fs.glob(f"{input_uri[len('hf://'):]}/**/*.parquet")
        return sorted(paths), True
    fs, rel = filesystem_and_path(input_uri)
    infos = fs.get_file_info(pyarrow.fs.FileSelector(rel, recursive=True))
    prefix = input_uri.split("://", 1)[0] + "://" if "://" in input_uri else ""
    paths = [prefix + i.path for i in infos if i.path.endswith(".parquet")]
    return sorted(paths), False


def write_chunk(rows: List[Dict[str, Any]], output_uri: str, part_idx: int):
    """Write one output parquet part."""
    table = pa.table({c: [r[c] for r in rows] for c in OUTPUT_COLUMNS})
    name = f"part-{part_idx:05d}.parquet"
    if "://" in output_uri:
        fs, rel = filesystem_and_path(output_uri)
        with fs.open_output_stream(f"{rel}/{name}") as f:
            pq.write_table(table, f)
    else:
        os.makedirs(output_uri, exist_ok=True)
        pq.write_table(table, os.path.join(output_uri, name))


# ---------------------------------------------------------------------------
# Hand-rolled streaming scheduler
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Video captioning on raw Ray Core.")
    parser.add_argument("--input", default="hf://datasets/HuggingFaceFV/finevideo")
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-videos", type=int, default=None)
    parser.add_argument("--write-chunk-rows", type=int, default=2000)
    args = parser.parse_args()
    output = args.output or default_output("ray_core")

    hf_token = os.environ.get("HF_TOKEN")
    if args.input.startswith("hf://") and not hf_token:
        logger.error("HF_TOKEN is required to read %s", args.input)
        sys.exit(1)

    ray.init(ignore_reinit_error=True)
    num_gpus = wait_for_gpus(min_gpus=int(os.environ.get("EXPECTED_GPUS", "8")))
    if num_gpus < 1:
        logger.error("No GPUs joined the cluster within the timeout.")
        sys.exit(1)

    # Manual backpressure caps — the knobs Ray Data sets automatically.
    max_shard_tasks = max(8, int(ray.cluster_resources().get("CPU", 0)))
    max_caption_tasks = num_gpus * 2  # two batches in flight per GPU hides dispatch latency

    shard_paths, is_hf = enumerate_shards(args.input, hf_token)
    limit = args.num_videos
    logger.info("input=%s (%d shards)  model=%s  GPUs=%d  output=%s",
                args.input, len(shard_paths), MODEL_SOURCE, num_gpus, output)

    # Spin up one caption actor per GPU and block on weight load so the timed
    # region measures steady-state throughput, not cold start.
    actors = [CaptionActor.remote() for _ in range(num_gpus)]
    logger.info("Loading %d vLLM engines (one per GPU)...", num_gpus)
    ray.get([a.ready.remote() for a in actors])
    logger.info("Engines ready.")

    monitor = UtilizationMonitor().start()
    t0 = time.time()

    shard_q = deque(shard_paths)
    decoded_q: deque = deque()
    inflight: Dict[Any, str] = {}  # ObjectRef -> "shard" | "caption"
    videos_enqueued = 0
    captions_done = 0
    total_video_seconds = 0.0
    result_buffer: List[Dict[str, Any]] = []
    part_idx = 0
    actor_rr = 0

    def n(kind: str) -> int:
        return sum(1 for k in inflight.values() if k == kind)

    def limit_reached() -> bool:
        return limit is not None and videos_enqueued >= limit

    while shard_q or decoded_q or inflight:
        # 1. Keep read+decode tasks flowing (unless we've hit --num-videos).
        while shard_q and n("shard") < max_shard_tasks and not limit_reached():
            ref = read_and_decode_shard.remote(shard_q.popleft(), hf_token, is_hf)
            inflight[ref] = "shard"

        # 2. Dispatch full VLM batches round-robin across the actor pool.
        while len(decoded_q) >= VLM_BATCH_SIZE and n("caption") < max_caption_tasks:
            batch = [decoded_q.popleft() for _ in range(VLM_BATCH_SIZE)]
            inflight[actors[actor_rr % num_gpus].caption.remote(batch)] = "caption"
            actor_rr += 1

        # 3. Drain the final partial batch once no more decoded rows will arrive.
        decoding_done = n("shard") == 0 and (not shard_q or limit_reached())
        if decoding_done and decoded_q and n("caption") < max_caption_tasks:
            batch = [decoded_q.popleft() for _ in range(len(decoded_q))]
            inflight[actors[actor_rr % num_gpus].caption.remote(batch)] = "caption"
            actor_rr += 1

        if not inflight:
            continue

        # 4. Wait for any in-flight task and route its result onward.
        ready_refs, _ = ray.wait(list(inflight.keys()), num_returns=1)
        ref = ready_refs[0]
        kind = inflight.pop(ref)
        out = ray.get(ref)
        if kind == "shard":
            take = out if limit is None else out[: max(0, limit - videos_enqueued)]
            decoded_q.extend(take)
            videos_enqueued += len(take)
            if limit_reached():
                shard_q.clear()
        else:  # caption
            captions_done += len(out)
            total_video_seconds += sum(r["duration_sec"] for r in out)
            result_buffer.extend(out)
            if len(result_buffer) >= args.write_chunk_rows:
                write_chunk(result_buffer, output, part_idx)
                part_idx += 1
                result_buffer = []
            logger.info(
                "captions=%d  decoded_q=%d  inflight[shard=%d cap=%d]",
                captions_done, len(decoded_q), n("shard"), n("caption"),
            )

    if result_buffer:
        write_chunk(result_buffer, output, part_idx)

    write_report(
        pipeline="ray_core",
        num_gpus=num_gpus,
        num_captions=captions_done,
        video_seconds=total_video_seconds,
        wall_seconds=time.time() - t0,
        utilization=monitor.stop(),
        output_uri=output,
    )


if __name__ == "__main__":
    main()

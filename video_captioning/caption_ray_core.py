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
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import ray

from utils import (
    MAX_MODEL_LEN,
    MODEL_LOAD_FORMAT,
    NUM_KEYFRAMES,
    OUTPUT_COLUMNS,
    VLM_BATCH_SIZE,
    UtilizationMonitor,
    build_messages_b64,
    clean_caption,
    decode_and_sample,
    dataset_source,
    default_output,
    filesystem_and_path,
    model_source,
    require_mirror_complete,
    require_s3_uri,
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
def read_and_decode_shard(path: str) -> List[Dict[str, Any]]:
    """Read one parquet shard's mp4 column and decode each clip to keyframes,
    one row per caption window.

    Read and decode are fused into one task on purpose: the task returns only
    the compact keyframe payloads (a few MB), never the multi-MB mp4 blobs, so
    the driver's scheduling loop never pulls raw video into head-node memory.
    """
    fs, rel = filesystem_and_path(path)
    table = pq.read_table(rel, filesystem=fs, columns=["mp4"])

    decoded = []
    for mp4 in table.column("mp4").to_pylist():
        decoded.extend(decode_and_sample(mp4))
    return decoded


@ray.remote(num_gpus=1)
class CaptionActor:
    """GPU stage: a persistent vLLM engine, one per GPU. No tensor parallelism —
    Qwen3-VL-8B fits comfortably on a single 96 GB GPU."""

    def __init__(self):
        # Configure the process-private model-streamer cache before importing
        # vLLM, whose environment settings may be resolved during import.
        worker_setup()
        from vllm import LLM

        self.llm = LLM(
            model=model_source(),
            load_format=MODEL_LOAD_FORMAT,
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


def enumerate_shards(input_uri: str) -> List[str]:
    """Return every Parquet shard under an AI Object Storage prefix."""
    import pyarrow.fs

    fs, rel = filesystem_and_path(input_uri)
    infos = fs.get_file_info(pyarrow.fs.FileSelector(rel, recursive=True))
    prefix = input_uri.split("://", 1)[0] + "://"
    paths = [prefix + i.path for i in infos if i.path.endswith(".parquet")]
    return sorted(paths)


def write_chunk(rows: List[Dict[str, Any]], output_uri: str, part_idx: int):
    """Write one output parquet part."""
    table = pa.table({c: [r[c] for r in rows] for c in OUTPUT_COLUMNS})
    name = f"part-{part_idx:05d}.parquet"
    fs, rel = filesystem_and_path(output_uri)
    with fs.open_output_stream(f"{rel}/{name}") as f:
        pq.write_table(table, f)


# ---------------------------------------------------------------------------
# Hand-rolled streaming scheduler
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Video captioning on raw Ray Core.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-videos", type=int, default=None)
    parser.add_argument("--write-chunk-rows", type=int, default=2000)
    args = parser.parse_args()
    input_uri = require_s3_uri(args.input or dataset_source(), "--input")
    output = require_s3_uri(args.output or default_output("ray_core"), "--output")
    model = model_source()
    require_mirror_complete(input_uri, "FineVideo")
    require_mirror_complete(model, "Qwen3-VL")

    num_gpus = wait_for_gpus(min_gpus=int(os.environ.get("EXPECTED_GPUS", "8")))
    if num_gpus < 1:
        logger.error("No GPUs joined the cluster within the timeout.")
        sys.exit(1)

    # Manual backpressure caps — the knobs Ray Data sets automatically.
    max_shard_tasks = max(8, int(ray.cluster_resources().get("CPU", 0)))
    max_caption_tasks = num_gpus * 2  # two batches in flight per GPU hides dispatch latency
    # Every decoded row funnels through decoded_q on the head node, and
    # windowed decode fans each shard out ~25x — decode outruns the GPUs, so
    # an uncapped queue OOMs the head on a 1M-caption run. Gate new shard
    # launches on rows already queued plus a running estimate of what the
    # in-flight shards will add. (Ray Data derives this from object sizes.)
    max_buffered_rows = int(os.environ.get("MAX_BUFFERED_ROWS", "150000"))
    rows_per_shard_est = 1000.0  # deliberately high until real shards report in
    shards_done = 0

    shard_paths = enumerate_shards(input_uri)
    limit = args.num_videos
    logger.info("input=%s (%d shards)  model=%s  GPUs=%d  output=%s",
                input_uri, len(shard_paths), model, num_gpus, output)

    # The hand-rolled pipeline must pre-provision its fleet: one engine per
    # GPU, all weights loaded, before the first caption. That fixed cost — and
    # the num_gpus GPUs it holds while paying it — belongs inside the timed
    # region; the Ray Data rebuild autoscales its pool instead of paying it.
    monitor = UtilizationMonitor().start()
    t0 = time.time()

    actors = [CaptionActor.remote() for _ in range(num_gpus)]
    logger.info("Loading %d vLLM engines (one per GPU)...", num_gpus)
    ray.get([a.ready.remote() for a in actors])
    logger.info("Engines ready.")

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
        # 1. Keep read+decode tasks flowing (unless we've hit --num-videos or
        #    the head-side row buffer is projected to overflow).
        while (
            shard_q
            and n("shard") < max_shard_tasks
            and not limit_reached()
            and len(decoded_q) + n("shard") * rows_per_shard_est < max_buffered_rows
        ):
            ref = read_and_decode_shard.options(
                label_selector={"cpu_only": "true"}
            ).remote(shard_q.popleft())
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
            shards_done += 1
            rows_per_shard_est += (len(out) - rows_per_shard_est) / shards_done
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
        input_uri=input_uri,
        num_gpus=num_gpus,
        num_captions=captions_done,
        video_seconds=total_video_seconds,
        wall_seconds=time.time() - t0,
        utilization=monitor.stop(),
        output_uri=output,
    )


if __name__ == "__main__":
    main()

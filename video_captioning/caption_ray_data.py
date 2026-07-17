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
from ray.data.llm import (
    PrepareImageStageConfig,
    build_processor,
    vLLMEngineProcessorConfig,
)

from utils import (
    MAX_MODEL_LEN,
    MODEL_LOAD_FORMAT,
    NUM_KEYFRAMES,
    VLM_BATCH_SIZE,
    UtilizationMonitor,
    dataset_source,
    decode_row,
    default_output,
    filesystem_and_path,
    model_source,
    require_mirror_complete,
    require_s3_uri,
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


def _decode_augmented(row, *, bucket: str):
    """Manifest-mode stage 1: manifest row -> GET the augmented mp4 from CAIOS
    (LOTA-accelerated) -> decode + window-sample. Fused GET+decode keeps the
    ~10 MB blobs out of the inter-stage object store entirely. A failed GET
    drops the clip (logged), mirroring decode_row's undecodable handling.
    The ok filter lives here rather than as parquet filter pushdown — the
    pushdown made 87k tiny-block manifest reads pathologically slow."""
    from augment_workers import s3_client
    from utils import decode_and_sample

    if int(row.get("ok", 1)) != 1:
        return []
    vid = f"{row['video_id']}-v{int(row['variant']):05d}"
    # CAPTION_SLICE="i/k": deterministic shard of the corpus, so a petabyte
    # run becomes k sequential jobs and a failure costs one slice, not the
    # run. crc32 is stable across runs/processes (hash() is not).
    slice_spec = os.environ.get("CAPTION_SLICE", "")
    if slice_spec:
        import zlib

        idx, _, k = slice_spec.partition("/")
        if zlib.crc32(vid.encode()) % int(k) != int(idx):
            return []
    # Manual retry loop: botocore retries the GET request itself, but a
    # timeout during the streaming Body.read() raises straight through — and
    # at ~50k concurrent cold reads those stalls are routine. Three attempts
    # with backoff turns per-mille clip drops into per-million.
    body = None
    for attempt in range(3):
        try:
            body = s3_client().get_object(
                Bucket=bucket, Key=row["out_key"]
            )["Body"].read()
            break
        except Exception:  # noqa: BLE001 — one lost clip must not kill 70M others
            if attempt == 2:
                logger.exception("GET failed for %s", row["out_key"])
                return []
            time.sleep(2 ** (2 * attempt))  # 1 s, 4 s
    return decode_and_sample(body, vid)


def _tolerate_engine_boot_failures():
    """Make vLLM engine-boot failures cost one actor, not the whole job.

    Ray Data's _ActorPool.pending_to_running() fully cleans up a
    failed-creation actor's bookkeeping and then RE-RAISES; the exception
    propagates through the streaming executor and kills the run. At a
    1,500-engine pinned pool, any per-boot failure probability p becomes job
    death 1-(1-p)^1500 — three separate boot-flake classes (port races, VRAM
    residue on init retries, ramp-wave timeouts) each killed a full run.
    Returning None instead takes the caller's benign "actor already killed"
    path, and the pool's min-size deficit spawns a fresh replacement actor
    (new process, new port, clean CUDA context).
    """
    from ray.data._internal.execution.operators import (
        actor_pool_map_operator as _apmo,
    )

    _orig = _apmo._ActorPool.pending_to_running

    def _safe(self, ready_ref):
        try:
            return _orig(self, ready_ref)
        except Exception:  # noqa: BLE001 — bookkeeping already cleaned pre-raise
            logger.warning(
                "Engine actor failed during boot; replaced instead of failing "
                "the job.", exc_info=True,
            )
            return None

    _apmo._ActorPool.pending_to_running = _safe


def main():
    parser = argparse.ArgumentParser(description="Video captioning on Ray Data.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-videos", type=int, default=None)
    args = parser.parse_args()
    input_uri = require_s3_uri(args.input or dataset_source(), "--input")
    output = require_s3_uri(args.output or default_output("ray_data"), "--output")
    model = model_source()
    # INPUT_KIND=augment_manifest: INPUT is an augment_ray_data.py output root
    # (70M raw mp4 objects + manifest/); the pipeline reads the manifest and
    # fetches clips directly instead of scanning parquet shards with an mp4
    # column. report.json is only written after a complete augmentation run,
    # so it doubles as this mode's _SUCCESS marker.
    input_kind = os.environ.get("INPUT_KIND", "parquet")
    if input_kind == "augment_manifest":
        import pyarrow.fs as pafs

        fs, in_path = filesystem_and_path(input_uri)
        if fs.get_file_info(f"{in_path}/report.json").type != pafs.FileType.File:
            raise RuntimeError(
                f"{input_uri} has no report.json — augmentation incomplete."
            )
    else:
        require_mirror_complete(input_uri, "FineVideo")
    require_mirror_complete(model, "Qwen3-VL")

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"worker_process_setup_hook": worker_setup},
    )
    if os.environ.get("TOLERATE_ENGINE_BOOT_FAILURES", "1") == "1":
        _tolerate_engine_boot_failures()
    # PREPROVISION_FLEET=1: demand both worker fleets as node-shaped bundles
    # at t0. Ray Data submits only what fits the current cluster, so the
    # autoscaler never sees pending demand from a backpressured pipeline and
    # the fleet stalls at a handful of nodes (measured: 1,140 CPUs / 64 GPUs
    # after an hour on a 70M-clip run). Node-shaped bundles are the form the
    # Anyscale autoscaler acts on.
    if os.environ.get("PREPROVISION_FLEET", "0") == "1":
        from ray.autoscaler.sdk import request_resources

        cpu_nodes = int(os.environ.get("FLEET_CPU_NODES", "199"))
        gpu_nodes = int(os.environ.get("FLEET_GPU_NODES", "200"))
        request_resources(
            bundles=[{"CPU": 180}] * cpu_nodes
            + [{"CPU": 120, "GPU": 8}] * gpu_nodes
        )
        logger.info(
            "Requested %d CPU nodes + %d GPU nodes from the autoscaler.",
            cpu_nodes, gpu_nodes,
        )
    expected_gpus = int(os.environ.get("EXPECTED_GPUS", "8"))
    if os.environ.get("PREPROVISION_GPUS", "1") == "1":
        # Benchmark mode: request and wait for the full fleet up front so both
        # pipelines run on identical, fully-provisioned hardware. The whole
        # fleet is billed for the whole run, held or not.
        num_gpus = wait_for_gpus(min_gpus=expected_gpus)
        if num_gpus < 1:
            logger.error("No GPUs joined the cluster within the timeout.")
            sys.exit(1)
    else:
        # Production mode: no fleet gate. The engine pool's pending actors
        # drive the node autoscaler, so GPU nodes arrive as the queue deepens
        # and release after the drain — provisioned GPU-seconds track held
        # GPU-seconds instead of fleet × wall. EXPECTED_GPUS is only the
        # pool's ceiling.
        num_gpus = expected_gpus
    logger.info(
        "input=%s  model=%s  GPUs=%d  output=%s",
        input_uri,
        model,
        num_gpus,
        output,
    )

    # ---- Stage 0: streamed parquet read, pinned to CPU-only nodes ----------
    # Pinning the multi-MB mp4 read to cpu_only-labeled workers keeps big blobs
    # off the GPU nodes; downstream stages are unpinned.
    # Truthful reservations, both dimensions. num_cpus=0.25: shard reads are
    # IO-bound (~17 s CPU inside a ~71 s task, the rest is S3 wait); the
    # autoscaler provisions against *reserved* CPUs, so a fractional value
    # stops the opening burst from summoning nodes that arrive after the
    # burst ends. memory=3 GiB: read tasks peak ~1.9 GiB USS *on average*
    # (tail higher) while materializing a ~476 MiB shard — without a memory
    # reservation the scheduler packs reads by CPU alone (480/node at 0.25)
    # and OOM-kills nodes when a deep read queue meets a small early cluster.
    read_kwargs = dict(
        columns=["mp4"],
        file_extensions=["parquet"],
        ray_remote_args={
            "label_selector": {"cpu_only": "true"},
            "num_cpus": 0.25,
            "memory": 3 * 1024**3,
        },
    )
    read_kwargs["filesystem"], input_path = filesystem_and_path(input_uri)
    repeats = int(os.environ.get("DATASET_REPEATS", "1"))
    pass_shards = int(os.environ.get("PASS_SHARDS", "0"))
    if input_kind == "augment_manifest":
        # The manifest is small (metadata rows only); the heavy GET happens
        # fused into decode. Read tasks need no mp4 columns or big memory.
        from functools import partial as _partial

        bucket = input_uri[len("s3://"):].partition("/")[0]
        # MANIFEST_SUBDIR selects one run's census (augment runs write
        # manifest/<run-stamp>/); the bare manifest/ root may hold multiple
        # overlapping generations from resumed runs.
        manifest_subdir = os.environ.get("MANIFEST_SUBDIR", "manifest")
        # Read fat and fast (default blocking; the pushdown-filtered 87k-block
        # read ground at ~40 s/task), then split — repartition without shuffle
        # is a block-split, no data movement. Block count is decode's
        # parallelism ceiling: manifest rows are ~200 B, so default blocking
        # packs 70M rows into a few hundred blocks, and one block = one
        # GET+decode task — that capped a 432-GPU cluster at ~150 decode
        # tasks (~500 windows/s). ~87k blocks ≈ 800 clips ≈ 30 min per task:
        # wide, driver-safe, bounded drain tail.
        ds = ray.data.read_parquet(
            f"{input_path}/{manifest_subdir}",
            columns=["video_id", "variant", "out_key", "ok"],
            file_extensions=["parquet"],
            filesystem=read_kwargs["filesystem"],
            ray_remote_args={"label_selector": {"cpu_only": "true"},
                             "num_cpus": 0.25},
        )
        if args.num_videos is not None:
            ds = ds.limit(args.num_videos)
        ds = ds.repartition(
            int(os.environ.get("MANIFEST_BLOCKS", "87500")), shuffle=False
        )
        # Stage 1 (manifest mode): GET (LOTA) + decode + window sample, fused.
        # Pinned to the CPU pool: feeding 1,600 engines needs only ~1,600
        # concurrent decodes (~2.2 s/clip vs ~110 ms/window consumed), and an
        # unpinned decode packs GPU-node CPUs/RAM until the OOM killer takes
        # the raylet down mid-engine-boot — measured, not hypothetical.
        decode_kwargs: dict = {
            "num_cpus": 1,
            "label_selector": {"cpu_only": "true"},
        }
        # DECODE_CONCURRENCY right-sizes the decode stage to what the engine
        # pool can consume (~1,600 concurrent at 2.2 s/clip feeds 1,600
        # engines). Uncapped, a wide CPU fleet spawns tens of thousands of
        # one-CPU workers whose GCS registrations alone overwhelm the head
        # (measured: worker joins timing out at ~60k workers/400 nodes).
        decode_conc = int(os.environ.get("DECODE_CONCURRENCY", "0"))
        if decode_conc:
            decode_kwargs["concurrency"] = decode_conc
        ds = ds.flat_map(_partial(_decode_augmented, bucket=bucket), **decode_kwargs)
        paths = None
        n_reads = 0
    elif repeats > 1 or pass_shards > 0:
        # Emulate a corpus N× larger — the continuous-workload regime where
        # engine boot and pool ramp amortize away — without storing one:
        # stream the same shards N times (LOTA serves repeats from cache).
        # A repeated *file list* keeps this a single read operator; unioning
        # N dataset branches would explode the DAG at large N. PASS_SHARDS
        # bounds each pass (~777 captions/shard; 129 shards ≈ 100k captions).
        import pyarrow.fs as pafs

        infos = read_kwargs["filesystem"].get_file_info(
            pafs.FileSelector(input_path, recursive=True)
        )
        shards = sorted(i.path for i in infos if i.path.endswith(".parquet"))
        if pass_shards > 0:
            shards = shards[:pass_shards]
        paths = shards * repeats
        # ~32 files per read task: 129k single-file tasks melted the head
        # node (GCS + queued block metadata) on the 100M run. Grouping cuts
        # driver-side task/metadata load ~30×; dynamic block splitting keeps
        # the output blocks right-sized regardless of grouping.
        ds = ray.data.read_parquet(
            paths, override_num_blocks=max(1, len(paths) // 32), **read_kwargs
        )
    else:
        paths = None
        ds = ray.data.read_parquet(input_path, **read_kwargs)

    # ---- Pre-flight: print the fan-out arithmetic before spending on it ----
    # Every scale failure in this pipeline's history was a *predictable*
    # fan-out: read tasks × peak task memory (node OOM), output files ×
    # gateway error rate (one 502 kills the run), engine count × boot flakes.
    # Predict the multipliers; don't discover them forty minutes in.
    if input_kind == "augment_manifest":
        # ~20 windows per augmented clip (mean span 0.85 × ~283 s / 12 s).
        n_clips = args.num_videos or 70_000_000
        est_captions = int(n_clips * 20)
    else:
        n_reads = len(paths) if paths else 1357
        est_captions = int(n_reads * 777)  # ~777 windows per FineVideo shard
    est_files = max(1, est_captions // 50_000)
    logger.info(
        "PREFLIGHT input_kind=%s est_captions=%d est_output_files=%d "
        "max_engines=%d est_read_peak_mem_per_cpu_node=%s",
        input_kind, est_captions, est_files, num_gpus,
        "bounded by 3GiB/task reservation",
    )
    if est_files > 20_000:
        logger.warning(
            "PREFLIGHT est_output_files=%d — at ~1e-4 gateway error rate this "
            "run should EXPECT multipart failures; verify write retries are on.",
            est_files,
        )

    # ---- Stage 1: CPU decode + keyframe sample (1 video -> 0/N rows) -------
    # (manifest mode fused GET+decode above; limit applied at its read)
    if input_kind != "augment_manifest":
        if args.num_videos is not None:
            ds = ds.limit(args.num_videos)
        ds = ds.flat_map(decode_row, num_cpus=1)

    # ---- Stage 2: GPU VLM caption (Qwen3-VL, one engine per GPU) -----------
    config_kwargs = dict(
        model_source=model,
        engine_kwargs={
            "load_format": MODEL_LOAD_FORMAT,
            "max_model_len": MAX_MODEL_LEN,
            # Overridable: after a transient engine-boot failure the in-actor
            # retry sees the dead attempt's unreleased VRAM, and 0.90 demands
            # more free memory than remains — retries then always fail. At
            # 16% KV usage this workload never misses the headroom.
            "gpu_memory_utilization": float(os.environ.get("GPU_MEM_UTIL", "0.90")),
            "limit_mm_per_prompt": {"image": NUM_KEYFRAMES},
            "mm_processor_cache_gb": 0,  # inputs are unique; skip the cache
            "trust_remote_code": True,
            # Small runs stay eager; large runs amortize CUDA-graph capture.
            "enforce_eager": (args.num_videos or 10**6) < 200,
        },
        batch_size=VLM_BATCH_SIZE,
        # (floor, n): an autoscaling engine pool with a small floor. The pool grows
        # only as queued work demands and releases engines at the tail, so the
        # job holds a fraction of the GPU-seconds of a pre-provisioned
        # n-engine fleet. The floor keeps a few engines alive through the
        # drain — without it, straggler blocks late in the stream can wait
        # minutes on a replacement engine boot (a 1M-caption run once spent
        # 21 minutes finishing its last 1% that way). Pin (n, n) to trade
        # GPU-hours for wall time outright. POOL_FLOOR scales the floor for
        # larger fleets, where 8 drain engines would stretch the tail.
        concurrency=(int(os.environ.get("POOL_FLOOR", "8")), num_gpus),
        # Engines ran at ~16-18% KV-cache usage with the default 4 in-flight
        # batches; deeper per-engine queues raise tokens/s at zero risk of
        # preemption at this sequence length. This workload is prefill-heavy
        # (6 images/window), so occupancy is the whole game.
        max_concurrent_batches=int(os.environ.get("MAX_CONCURRENT_BATCHES", "8")),
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

    # ---- Live progress: one greppable line every 2 min ---------------------
    # GPUs held + output parquet count (min_rows_per_file=50k, so files×50k
    # ≈ captions durably written). This is what distinguishes "ramping",
    # "healthy", and "stuck" on a day-scale run without downloading logs.
    def _caption_progress(stop_evt: "threading.Event", fs, path: str):
        import json as _json

        import pyarrow.fs as pafs

        while not stop_evt.wait(120):
            try:
                total, avail = ray.cluster_resources(), ray.available_resources()
                try:
                    files = sum(
                        1 for i in fs.get_file_info(pafs.FileSelector(path))
                        if i.path.endswith(".parquet")
                    )
                except Exception:  # noqa: BLE001 — dir absent until first write
                    files = 0
                print("CAPTION_PROGRESS " + _json.dumps({
                    "elapsed_s": round(time.time() - t0, 1),
                    "gpus_held": round(total.get("GPU", 0.0)
                                       - avail.get("GPU", 0.0), 1),
                    "gpus_total": round(total.get("GPU", 0.0), 1),
                    "cpus_held": round(total.get("CPU", 0.0)
                                       - avail.get("CPU", 0.0), 1),
                    "cpus_total": round(total.get("CPU", 0.0), 1),
                    "output_files": files,
                    "est_captions_written": files * 50_000,
                }), flush=True)
            except Exception:  # noqa: BLE001 — observability must not kill work
                pass

    import threading

    # ---- Stage 3: terminal write (this is what actually runs the stream) ---
    # try_create_dir=False skips the pre-write existence check, which some
    # object stores reject (they have no real directories). min_rows_per_file
    # coalesces the one-file-per-block default (>100k multipart creates at
    # 100M rows — enough calls that a single transient gateway 502 becomes a
    # near-certainty) into ~2k files, and retry_exceptions absorbs the 502s
    # that remain instead of failing the whole run.
    write_fs, write_path = filesystem_and_path(output)
    stop_progress = threading.Event()
    threading.Thread(
        target=_caption_progress, args=(stop_progress, write_fs, write_path),
        daemon=True,
    ).start()
    try:
        ds.write_parquet(
            write_path,
            filesystem=write_fs,
            try_create_dir=False,
            min_rows_per_file=50_000,
            ray_remote_args={"retry_exceptions": [OSError], "max_retries": 8},
        )
    finally:
        stop_progress.set()
        if os.environ.get("PREPROVISION_FLEET", "0") == "1":
            try:
                from ray.autoscaler.sdk import request_resources

                request_resources(bundles=[])  # release the fleet for the drain
            except Exception:  # noqa: BLE001
                pass

    wall = time.time() - t0
    utilization = monitor.stop()

    # Read the captions back (small: metadata + text, no images) to tally exact
    # counts and total video-seconds for the report.
    out_ds = ray.data.read_parquet(write_path, filesystem=write_fs)
    write_report(
        pipeline="ray_data",
        input_uri=input_uri,
        num_gpus=num_gpus,
        num_captions=out_ds.count(),
        video_seconds=float(out_ds.sum("duration_sec") or 0.0),
        wall_seconds=wall,
        utilization=utilization,
        output_uri=output,
    )


if __name__ == "__main__":
    main()

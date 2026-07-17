"""Synthetic data augmentation: scale FineVideo ~1000× with distributed
ffmpeg transcoding, measuring CAIOS write throughput while it runs.

The 630 GiB corpus becomes a ~0.6 PB augmented dataset (AUGMENT_FACTOR
variants per clip; see augment_workers.py for the two-tier remux/re-encode
design and why full re-encoding at this scale is arithmetic suicide). The
pipeline is three phases:

    read_parquet(finevideo)                  # mp4 shards from CAIOS
        .map(stage_source)                   # 1) each clip -> one addressable
                                             #    object under datasets/finevideo_mp4/
    from_items(chunks)                       # 2) (clip, variant-range) items,
        .flat_map(transcode_chunk)           #    GET (LOTA-cached) -> ffmpeg ×N
        .write_parquet(manifest)             #    -> PUT each variant
    read_parquet(manifest)                   # 3) aggregate the measurement

Every PUT and GET in phase 2 is timed on the worker and lands in the manifest,
so the job doubles as a sustained CAIOS write benchmark under a real
workload: report.json carries aggregate + peak + per-node write GiB/s, PUT
latency percentiles, read-side numbers, and CPU utilization — comparable
against the synthetic ceilings from storage_bench.py (~40 GiB/s aggregate PUT,
100-150 concurrency/node, on this fleet).

Sizing: output ≈ source_bytes × AUGMENT_FACTOR × 0.85. The full corpus at the
default factor 1000 is ~0.6 PB — far beyond CAIOS's default 20 TiB/AZ/account
quota, so full runs need a quota raise first. The pre-flight budget check
(OUTPUT_BUDGET_TIB) fails fast before any transcoding starts.

Usage:
    python augment_ray_data.py --num-videos 32   # smoke: ~16 GiB out at factor 40
    python augment_ray_data.py                   # whole corpus (quota!)
"""

import argparse
import json
import logging
import os
import threading
import time
from functools import partial
from typing import Any, Dict, List, Tuple

import ray

import augment_workers as aw
from utils import (
    UtilizationMonitor,
    ai_storage_uri,
    dataset_source,
    default_output,
    filesystem_and_path,
    require_mirror_complete,
    require_s3_uri,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("video_augment")

TRANSCODE_CPUS = float(os.environ.get("TRANSCODE_CPUS", "1"))
OUTPUT_BUDGET_TIB = float(os.environ.get("OUTPUT_BUDGET_TIB", "700"))
# Explicit up-front CPU demand for wide fleets. Ray Data only submits tasks
# the cluster can schedule, so the autoscaler never sees pending demand
# beyond the initial burst and the fleet stalls at a handful of nodes.
# request_resources() is the same fix the captioning benchmark uses for its
# 256-GPU fleet: demand the whole target at t0 and let nodes join in
# parallel. Two details matter: the demand must be expressed as a modest
# number of node-shaped bundles (hundreds of [{"CPU": <node>}] bundles work;
# tens of thousands of 1-CPU bundles and the aggregate num_cpus= form are
# both ignored on the Anyscale K8s stack), and it should be issued before
# staging so nodes provision while staging runs.
PREPROVISION_CPUS = int(os.environ.get("PREPROVISION_CPUS", "0"))
PREPROVISION_BUNDLE_CPUS = int(os.environ.get("PREPROVISION_BUNDLE_CPUS", "180"))
# Hard cap on concurrent transcode tasks (0 = uncapped). This is how a run is
# held to a CPU budget on shared capacity: 200 caps reserved CPUs at 200 ×
# TRANSCODE_CPUS, and the node autoscaler follows that demand instead of the
# pool ceiling.
TRANSCODE_CONCURRENCY = int(os.environ.get("TRANSCODE_CONCURRENCY", "0"))
# Skip chunks whose variants already exist (resume after quota/capacity/
# timeout interruptions). One HEAD per fresh chunk.
RESUME_SKIP_EXISTING = os.environ.get("RESUME_SKIP_EXISTING", "1") == "1"
PROGRESS_INTERVAL_S = float(os.environ.get("PROGRESS_INTERVAL_S", "60"))
# Work items per input block (= per task). One task per block is Ray Data's
# scheduling unit: from_items' default ~200 blocks caps the whole fleet at
# ~200 concurrent tasks, while one block per item (437k blocks at full
# corpus) crushes the driver — it died mid-run tracking them. A few chunks
# per block keeps tasks minutes-long and block count near 100k.
CHUNKS_PER_BLOCK = max(1, int(os.environ.get("CHUNKS_PER_BLOCK", "4")))
GiB = 2**30


@ray.remote(num_cpus=0)
class _AugmentStats:
    """Named actor the transcode tasks report chunk totals to, so the driver
    can log fleet-wide read/write throughput every minute of a multi-hour
    run — not just in the post-hoc report."""

    def __init__(self):
        self.totals = {"bytes_written": 0, "bytes_read": 0,
                       "ok": 0, "failed": 0, "skipped": 0}

    def report(self, bytes_written: int, bytes_read: int,
               ok: int, failed: int, skipped: int) -> None:
        self.totals["bytes_written"] += int(bytes_written)
        self.totals["bytes_read"] += int(bytes_read)
        self.totals["ok"] += int(ok)
        self.totals["failed"] += int(failed)
        self.totals["skipped"] += int(skipped)

    def snapshot(self) -> Dict[str, int]:
        return dict(self.totals)


def _progress_logger(stats: Any, t0: float, stop: threading.Event) -> None:
    prev: Dict[str, int] = {"bytes_written": 0, "bytes_read": 0, "ok": 0}
    prev_t = time.time()
    while not stop.wait(PROGRESS_INTERVAL_S):
        try:
            snap = ray.get(stats.snapshot.remote(), timeout=30)
        except Exception:  # noqa: BLE001 — logging must never kill the run
            continue
        now = time.time()
        dt = max(now - prev_t, 1e-9)
        line = {
            "elapsed_s": round(now - t0, 1),
            "written_tib": round(snap["bytes_written"] / 2**40, 3),
            "write_gib_s": round(
                (snap["bytes_written"] - prev["bytes_written"]) / dt / GiB, 2),
            "read_gib_s": round(
                (snap["bytes_read"] - prev["bytes_read"]) / dt / GiB, 2),
            "variants_done": snap["ok"],
            "variants_per_s": round((snap["ok"] - prev["ok"]) / dt, 1),
            "failed": snap["failed"],
            "skipped": snap["skipped"],
        }
        print("AUGMENT_PROGRESS " + json.dumps(line), flush=True)
        prev, prev_t = snap, now


def split_uri(uri: str) -> Tuple[str, str]:
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    return bucket, prefix.rstrip("/")


def build_chunks(
    staged: List[Dict[str, Any]], factor: int, n_reencode: int,
    out_bucket: str, out_prefix: str,
) -> List[Dict[str, Any]]:
    """Variant indices [0, n_reencode) re-encode, the rest remux; chunk sizes
    differ per tier so task durations stay comparable."""
    chunks = []
    for r in staged:
        base = {
            "video_id": r["video_id"],
            "src_bucket": r["src_bucket"],
            "src_key": r["src_key"],
            "src_bytes": r["src_bytes"],
            "duration_sec": r["duration_sec"],
            "out_bucket": out_bucket,
            "out_prefix": out_prefix,
            "skip_existing": RESUME_SKIP_EXISTING,
        }
        # cold_get marks each clip's first chunk: its GET pulls the staged
        # object from the backend (writes aren't LOTA-cached), while every
        # later chunk of the clip reads the distributed cache. Approximate —
        # chunks of one clip can interleave across the fleet — but at 25-50
        # chunks/clip the mislabeled fraction is small.
        for start in range(0, n_reencode, aw.REENCODE_VARIANTS_PER_TASK):
            n = min(start + aw.REENCODE_VARIANTS_PER_TASK, n_reencode) - start
            chunks.append({**base, "start": start, "n": n, "reencode": True,
                           "cold_get": start == 0})
        for start in range(n_reencode, factor, aw.REMUX_VARIANTS_PER_TASK):
            n = min(start + aw.REMUX_VARIANTS_PER_TASK, factor) - start
            chunks.append({**base, "start": start, "n": n, "reencode": False,
                           "cold_get": start == 0 and n_reencode == 0})
    return chunks


def _pct(series, q: float) -> float:
    return round(float(series.quantile(q)), 4) if len(series) else 0.0


def _read_stats(df, wall: float) -> Dict[str, Any]:
    """Cold (backend) vs warm (LOTA distributed cache) GET split. Each clip's
    first chunk is the cold read; the per-GET MiB/s difference between the two
    groups is the cache's contribution, per-connection."""
    gets = df[df["get_bytes"] > 0]
    out: Dict[str, Any] = {
        "bytes": int(gets["get_bytes"].sum()),
        "gib_per_s_mean": round(
            float(gets["get_bytes"].sum()) / max(wall, 1e-9) / 2**30, 2
        ),
    }
    for name, group in (("cold", gets[gets["get_cold"] == 1]),
                        ("warm", gets[gets["get_cold"] == 0])):
        out[name] = {
            "gets": int(len(group)),
            "mib_per_s_per_get": round(
                float(group["get_bytes"].sum())
                / max(float(group["get_seconds"].sum()), 1e-9) / 2**20, 1
            ),
            "get_p50_s": _pct(group["get_seconds"], 0.50),
            "get_p99_s": _pct(group["get_seconds"], 0.99),
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="1000× ffmpeg augmentation + CAIOS write benchmark."
    )
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-videos", type=int, default=None)
    args = parser.parse_args()

    input_uri = require_s3_uri(args.input or dataset_source(), "--input")
    output = require_s3_uri(args.output or default_output("augment"), "--output")
    sources_uri = require_s3_uri(
        os.environ.get("AUGMENT_SOURCES") or ai_storage_uri("datasets/finevideo_mp4"),
        "AUGMENT_SOURCES",
    )
    require_mirror_complete(input_uri, "FineVideo")

    factor = aw.AUGMENT_FACTOR
    n_reencode = round(factor * aw.REENCODE_FRACTION)
    logger.info(
        "input=%s  output=%s  sources=%s  factor=%d (%d reencode + %d remux per clip)",
        input_uri, output, sources_uri, factor, n_reencode, factor - n_reencode,
    )

    ray.init(ignore_reinit_error=True)
    if PREPROVISION_CPUS:
        from ray.autoscaler.sdk import request_resources

        n_bundles = -(-PREPROVISION_CPUS // PREPROVISION_BUNDLE_CPUS)
        request_resources(bundles=[{"CPU": PREPROVISION_BUNDLE_CPUS}] * n_bundles)
        logger.info(
            "Requested %d × %d-CPU bundles (%d CPUs) from the autoscaler; "
            "nodes provision while staging runs.",
            n_bundles, PREPROVISION_BUNDLE_CPUS, n_bundles * PREPROVISION_BUNDLE_CPUS,
        )
    monitor = UtilizationMonitor().start()
    t0 = time.time()

    # ---- Phase 1: stage each source clip as one addressable CAIOS object ---
    # Same read shape as the captioning pipelines: pinned to cpu_only nodes,
    # truthful fractional CPU reservation so the IO-bound burst doesn't summon
    # nodes that outlive it.
    src_bucket, src_prefix = split_uri(sources_uri)
    # The memory reservation matters as much as the CPU one: a shard read
    # materializes ~2.4 GiB of decoded blocks (Ray's issue detector's own
    # number), and 0.25 CPU alone would admit up to 480 concurrent reads on a
    # 480 Gi node — enough to OOM the raylet. Reserving the real footprint
    # caps reads per node by memory instead.
    read_kwargs = dict(
        columns=["mp4"],
        file_extensions=["parquet"],
        ray_remote_args={
            "label_selector": {"cpu_only": "true"},
            "num_cpus": 0.25,
            "memory": 3 * GiB,
        },
    )
    read_kwargs["filesystem"], input_path = filesystem_and_path(input_uri)
    ds = ray.data.read_parquet(input_path, **read_kwargs)
    if args.num_videos is not None:
        ds = ds.limit(args.num_videos)
    staged_rows = ds.map(
        partial(aw.stage_source, bucket=src_bucket, prefix=src_prefix),
        num_cpus=0.5,
        memory=512 * 2**20,  # one clip in RAM + its tmp copy for ffprobe
    ).take_all()
    stage_wall = time.time() - t0

    seen, staged, failed = set(), [], []
    for r in staged_rows:
        if not r["ok"]:
            failed.append(r)
        elif r["video_id"] not in seen:  # dup clips would collide on out_keys
            seen.add(r["video_id"])
            staged.append({**r, "src_bucket": src_bucket})
    # int(): take_all() yields numpy scalars, which json.dumps rejects later
    src_bytes_total = int(sum(r["src_bytes"] for r in staged))
    staged_bytes = int(sum(r["src_bytes"] for r in staged if r["staged"]))
    logger.info(
        "staged %d clips (%.1f GiB, %.1f GiB newly written in %.0f s; "
        "%d skipped as already staged, %d undecodable/failed, %d duplicates)",
        len(staged), src_bytes_total / GiB, staged_bytes / GiB, stage_wall,
        sum(1 for r in staged if not r["staged"]), len(failed),
        len(staged_rows) - len(failed) - len(staged),
    )
    for r in failed[:5]:
        logger.warning("stage failure %s: %s", r["video_id"], r["error"])
    if not staged:
        raise RuntimeError("No source clips staged; nothing to augment.")

    # ---- Pre-flight output budget: fail before transcoding, not mid-PB -----
    est_out = src_bytes_total * (
        (factor - n_reencode) * aw.EST_REMUX_RATIO + n_reencode * aw.EST_REENCODE_RATIO
    )
    est_tib = est_out / 2**40
    logger.info(
        "estimated output: %.2f TiB (%d variants). CAIOS default quota is "
        "20 TiB STANDARD per AZ per account — raise it via CoreWeave support "
        "before full-scale runs.", est_tib, len(staged) * factor,
    )
    if est_tib > OUTPUT_BUDGET_TIB:
        raise RuntimeError(
            f"Estimated output {est_tib:.1f} TiB exceeds OUTPUT_BUDGET_TIB="
            f"{OUTPUT_BUDGET_TIB}. Raise the budget (and your CAIOS quota) or "
            "shrink NUM_VIDEOS / AUGMENT_FACTOR."
        )

    # ---- Phase 2: distributed transcode fan-out + timed PUTs ---------------
    out_bucket, out_prefix = split_uri(output)
    chunks = build_chunks(staged, factor, n_reencode, out_bucket, out_prefix)
    logger.info(
        "transcode fan-out: %d tasks over %d clips × %d variants",
        len(chunks), len(staged), factor,
    )
    t1 = time.time()
    stats = _AugmentStats.options(name="augment_stats", namespace="augment").remote()
    ray.get(stats.snapshot.remote())  # actor must exist before tasks look it up
    stop_progress = threading.Event()
    threading.Thread(
        target=_progress_logger, args=(stats, t1, stop_progress), daemon=True
    ).start()

    # Pinned to the labeled worker pool: the head node may share the workers'
    # shape, and 180 ffmpeg processes competing with GCS + a ~430k-block
    # driver is exactly the failure mode being avoided.
    flat_map_kwargs: Dict[str, Any] = {
        "num_cpus": TRANSCODE_CPUS,
        "label_selector": {"cpu_only": "true"},
    }
    if TRANSCODE_CONCURRENCY:
        flat_map_kwargs["concurrency"] = TRANSCODE_CONCURRENCY
        logger.info(
            "transcode concurrency capped at %d tasks (%.0f reserved CPUs)",
            TRANSCODE_CONCURRENCY, TRANSCODE_CONCURRENCY * TRANSCODE_CPUS,
        )
    # Per-run manifest generation: resumed runs each write a complete census
    # (fresh + skipped + failed rows). Appending generations into one dir
    # double-counts the report and would double-caption downstream consumers;
    # a run-stamped subdir keeps the latest census authoritative.
    manifest_uri = f"{output}/manifest/{time.strftime('%Y%m%d-%H%M%S')}"
    logger.info("manifest census for this run: %s", manifest_uri)
    write_fs, manifest_path = filesystem_and_path(manifest_uri)
    try:
        (
            ray.data.from_items(
                chunks,
                override_num_blocks=max(1, -(-len(chunks) // CHUNKS_PER_BLOCK)),
            )
            .flat_map(aw.transcode_chunk, **flat_map_kwargs)
            .write_parquet(manifest_path, filesystem=write_fs, try_create_dir=False)
        )
    finally:
        stop_progress.set()
        if PREPROVISION_CPUS:
            try:
                from ray.autoscaler.sdk import request_resources

                request_resources(bundles=[])  # release the fleet for the drain
            except Exception:  # noqa: BLE001
                pass
    transcode_wall = time.time() - t1
    utilization = monitor.stop()
    total_wall = time.time() - t0

    # ---- Phase 3: aggregate the measurement from the manifest --------------
    numeric_cols = [
        "ok", "skipped", "out_bytes", "out_duration_sec", "transcode_seconds",
        "put_seconds", "put_end_ts", "get_seconds", "get_bytes", "get_cold",
    ]
    df = (
        ray.data.read_parquet(
            manifest_path, filesystem=write_fs,
            columns=numeric_cols + ["tier", "node_ip"],
        ).to_pandas()
    )
    okd = df[df["ok"] == 1]
    out_bytes = int(okd["out_bytes"].sum())  # dataset size incl. resumed chunks
    video_seconds_out = float(okd["out_duration_sec"].sum())

    # Peak sustained write rate: 30 s buckets over PUT completion times.
    # Skipped (resumed) variants have put_end_ts=0 and drop out here, so the
    # write section measures only bytes actually PUT by this run.
    puts = okd[okd["put_end_ts"] > 0]
    written_bytes = int(puts["out_bytes"].sum())
    bucket_gib_s = (
        puts.groupby((puts["put_end_ts"] // 30).astype("int64"))["out_bytes"].sum()
        / 30 / GiB
    )
    per_node = {
        str(ip): round(float(b) / max(transcode_wall, 1e-9) / GiB, 3)
        for ip, b in puts.groupby("node_ip")["out_bytes"].sum().items()
    }
    tiers = {
        tier: {
            "variants": int(len(g)),
            "out_bytes": int(g["out_bytes"].sum()),
            "transcode_core_seconds": round(float(g["transcode_seconds"].sum()), 1),
        }
        for tier, g in okd.groupby("tier")
    }

    report = {
        "pipeline": "augment_ffmpeg",
        "dataset": input_uri,
        "sources": sources_uri,
        "output": output,
        "source_clips": len(staged),
        "source_bytes": src_bytes_total,
        "augment_factor": factor,
        "reencode_fraction": aw.REENCODE_FRACTION,
        "variants_requested": len(staged) * factor,
        "variants_ok": int(len(okd)),
        "variants_failed": int(len(df) - len(okd)),
        "variants_skipped_resume": int(okd["skipped"].sum()),
        "output_bytes": out_bytes,
        "output_tib": round(out_bytes / 2**40, 3),
        "bytes_amplification": round(out_bytes / max(src_bytes_total, 1), 1),
        "video_hours_out": round(video_seconds_out / 3600, 1),
        "wall_seconds": {
            "stage": round(stage_wall, 1),
            "transcode": round(transcode_wall, 1),
            "total": round(total_wall, 1),
        },
        "write": {
            "bytes": written_bytes,
            "gib_per_s_mean": round(written_bytes / max(transcode_wall, 1e-9) / GiB, 2),
            "gib_per_s_peak_30s": round(float(bucket_gib_s.max()), 2)
            if len(bucket_gib_s) else 0.0,
            "put_p50_s": _pct(puts["put_seconds"], 0.50),
            "put_p90_s": _pct(puts["put_seconds"], 0.90),
            "put_p99_s": _pct(puts["put_seconds"], 0.99),
            "puts": int(len(puts)),
            "per_node_gib_per_s": per_node,
            "stage_bytes": staged_bytes,
            "stage_gib_per_s": round(staged_bytes / max(stage_wall, 1e-9) / GiB, 2),
        },
        "read": _read_stats(df, transcode_wall),
        "throughput": {
            "variants_per_sec": round(len(okd) / max(transcode_wall, 1e-9), 1),
            "video_hours_out_per_wall_hour": round(
                video_seconds_out / max(transcode_wall, 1e-9), 1
            ),
        },
        "tiers": tiers,
        "utilization": utilization,
    }

    logger.info("=" * 62)
    logger.info("AUGMENTATION REPORT")
    logger.info("=" * 62)
    logger.info("  variants ok/failed      : %d / %d",
                report["variants_ok"], report["variants_failed"])
    logger.info("  output                  : %.3f TiB (%.0f× the %d-clip source)",
                report["output_tib"], report["bytes_amplification"],
                report["source_clips"])
    logger.info("  wall stage/transcode    : %.0f s / %.0f s",
                stage_wall, transcode_wall)
    logger.info("  CAIOS write mean/peak   : %.2f / %.2f GiB/s over %d nodes",
                report["write"]["gib_per_s_mean"],
                report["write"]["gib_per_s_peak_30s"], len(per_node))
    logger.info("  PUT p50/p90/p99         : %.2f / %.2f / %.2f s",
                report["write"]["put_p50_s"], report["write"]["put_p90_s"],
                report["write"]["put_p99_s"])
    logger.info("  CPU util mean/p95       : %s / %s",
                utilization["cpu_util"]["mean"], utilization["cpu_util"]["p95"])
    logger.info("=" * 62)
    print("AUGMENT_REPORT " + json.dumps(report), flush=True)

    if report["variants_failed"]:
        sample = (
            ray.data.read_parquet(
                manifest_path, filesystem=write_fs,
                columns=["video_id", "variant", "tier", "ok", "error"],
            )
            .filter(lambda r: r["ok"] == 0)
            .take(5)
        )
        for r in sample:
            logger.warning(
                "variant failure %s/v%05d (%s): %s",
                r["video_id"], r["variant"], r["tier"], r["error"],
            )

    fs, out_path = filesystem_and_path(output)
    with fs.open_output_stream(f"{out_path}/report.json") as f:
        f.write(json.dumps(report, indent=2).encode())
    logger.info("Wrote report.json to %s", output)


if __name__ == "__main__":
    main()

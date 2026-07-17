"""ffmpeg + CAIOS workers for augment_ray_data.py.

Lives in its own importable module (like bench_workers.py) so Ray Data
serializes the map functions by reference, and so the ffmpeg command builders
can be tested locally without ray/boto3 installed — top-level imports are
stdlib only.

Two augmentation tiers, chosen by arithmetic rather than taste. Scaling the
630 GiB corpus 1000× means ~44M output clips (~3.4M video-hours). Fully
re-encoding all of them is ~1M CPU-hours of x264 — months on any reasonable
fleet — while the write path only needs ~4.3 hours at the measured ~40 GiB/s
PUT ceiling. So:

  - remux tier (the volume): keyframe-aligned temporal crop + playback-speed
    retime, `-c copy`. No pixels are touched, so each variant costs ~0 CPU and
    the tier runs at IO speed — this is what makes a 0.6 PB write job take
    hours instead of months, and what actually exercises the CAIOS write path.
  - reencode tier (the pixel diversity): spatial crop, h-flip, color jitter
    (eq/hue), optional gaussian noise, and a speed change, re-encoded with
    libx264. REENCODE_FRACTION of each clip's variants (default 2%) go through
    this tier; it dominates CPU cost, so the fraction is the wall-clock knob.

Every variant's parameters derive from random.Random(f"{video_id}:{index}"),
so the augmented dataset is reproducible and any variant can be regenerated
in isolation.
"""

import functools
import json
import os
import random
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

MiB = 1024 * 1024

# Variants generated per source clip (the "1000×").
AUGMENT_FACTOR = int(os.environ.get("AUGMENT_FACTOR", "1000"))
# Fraction of each clip's variants that get the full libx264 re-encode.
REENCODE_FRACTION = float(os.environ.get("REENCODE_FRACTION", "0.02"))
# Fan-out granularity: variants produced per Ray task. Remux variants are
# ~1-2 s each, re-encodes are ~1-2 min, so the chunk sizes differ to keep
# task durations in the tens-of-seconds-to-minutes band either way.
REMUX_VARIANTS_PER_TASK = int(os.environ.get("REMUX_VARIANTS_PER_TASK", "25"))
REENCODE_VARIANTS_PER_TASK = int(os.environ.get("REENCODE_VARIANTS_PER_TASK", "4"))

FFMPEG_PRESET = os.environ.get("FFMPEG_PRESET", "veryfast")
FFMPEG_THREADS = int(os.environ.get("FFMPEG_THREADS", "1"))
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", "1800"))

# Background uploader threads per transcode task. Synchronous PUTs between
# ffmpeg runs would dilute per-node write concurrency far below the measured
# optimum (~100-150 concurrent PUTs/node saturates CAIOS at ~40 GiB/s
# aggregate; the knee in the latency curve is ~100). With 120 one-CPU tasks
# per node, 2 uploader threads each puts the node in that band whenever the
# remux tier is producing, and overlaps encode with upload for free.
UPLOAD_THREADS_PER_TASK = int(os.environ.get("UPLOAD_THREADS_PER_TASK", "2"))

# Bytes-out estimate per variant, as a fraction of source bytes — measured on
# the 2026-07-15 staging runs (600k variants): remux averages 0.64× (span
# mean ~0.85 × container-overhead compression), re-encodes 0.57×. Used only
# for the pre-flight output-budget check.
EST_REMUX_RATIO = 0.64
EST_REENCODE_RATIO = 0.57

# Remux parameter ranges: temporal span kept, and the -itsscale retime factor
# (>1 = slower playback). Mean span sets the bytes amplification: factor 1000
# at mean 0.85 span ≈ 850× bytes ≈ 0.6 PB from the 630 GiB corpus.
REMUX_SPAN = (0.72, 0.98)
REMUX_SCALE = (0.80, 1.25)


# ---------------------------------------------------------------------------
# S3 client (CoreWeave AI Object Storage via the in-cluster LOTA endpoint)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def s3_client():
    """One client per worker process; virtual-hosted addressing is required."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
        or "http://cwlota.com",
        region_name=os.environ.get("AWS_REGION", "US-EAST-14A"),
        config=Config(
            s3={"addressing_style": "virtual"},
            max_pool_connections=64,
            connect_timeout=30,
            # Default 60 s read timeout trips routinely when tens of
            # thousands of concurrent cold reads queue at the backend.
            read_timeout=120,
            retries={
                "max_attempts": int(os.environ.get("S3_MAX_ATTEMPTS", "8")),
                "mode": "standard",
            },
        ),
    )


@functools.lru_cache(maxsize=1)
def _transfer_config():
    """Multipart at >=64 MiB with >=50 MiB parts spreads large objects across
    CAIOS nodes (a single PutObject lands the whole object on one node)."""
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=64 * MiB,
        multipart_chunksize=64 * MiB,
        max_concurrency=8,
    )


@functools.lru_cache(maxsize=1)
def _node_ip() -> str:
    import ray

    return ray.util.get_node_ip_address()


def _stats_actor():
    """The driver's live-throughput actor, if one was registered."""
    global _STATS
    if _STATS is _UNSET:
        try:
            import ray

            _STATS = ray.get_actor("augment_stats", namespace="augment")
        except Exception:  # noqa: BLE001 — standalone runs have no actor
            _STATS = None
    return _STATS


_UNSET = object()
_STATS = _UNSET


def _report_progress(rows: List[Dict[str, Any]]) -> None:
    """Fire-and-forget one summary per chunk so the driver can log read/write
    throughput continuously during multi-hour runs instead of only at the end."""
    actor = _stats_actor()
    if actor is None:
        return
    try:
        actor.report.remote(
            sum(r["out_bytes"] for r in rows if r["put_end_ts"] > 0),
            sum(r["get_bytes"] for r in rows),
            sum(r["ok"] for r in rows),
            sum(1 for r in rows if not r["ok"]),
            sum(r.get("skipped", 0) for r in rows),
        )
    except Exception:  # noqa: BLE001 — stats must never fail the work
        pass


def _timed_put(bucket: str, key: str, path: str, size: int) -> Tuple[float, float]:
    """Upload one file; returns (seconds, end_timestamp) for the latency and
    time-bucketed throughput measurements."""
    t0 = time.time()
    if size < 64 * MiB:
        with open(path, "rb") as f:
            s3_client().put_object(Bucket=bucket, Key=key, Body=f.read())
    else:
        s3_client().upload_file(path, bucket, key, Config=_transfer_config())
    t1 = time.time()
    return t1 - t0, t1


# ---------------------------------------------------------------------------
# Variant parameters + ffmpeg command builders (pure; locally testable)
# ---------------------------------------------------------------------------


def variant_params(video_id: str, index: int, duration_sec: float,
                   reencode: bool) -> Dict[str, Any]:
    """Deterministic augmentation parameters for one variant."""
    rnd = random.Random(f"{video_id}:{index}")
    if reencode:
        span = rnd.uniform(0.80, 1.0)
        params = {
            "tier": "reencode",
            "span_frac": round(span, 4),
            "start_frac": round(rnd.uniform(0.0, 1.0 - span), 4),
            # log-uniform so 0.75× and 1.33× are equally likely
            "speed": round(2.0 ** rnd.uniform(-0.415, 0.415), 4),
            "crop_scale": round(rnd.uniform(0.85, 0.98), 4),
            "crop_x": round(rnd.uniform(0.0, 1.0), 4),
            "crop_y": round(rnd.uniform(0.0, 1.0), 4),
            "hflip": rnd.random() < 0.5,
            "brightness": round(rnd.uniform(-0.08, 0.08), 4),
            "contrast": round(rnd.uniform(0.92, 1.08), 4),
            "saturation": round(rnd.uniform(0.85, 1.15), 4),
            "gamma": round(rnd.uniform(0.95, 1.05), 4),
            "hue_deg": round(rnd.uniform(-12.0, 12.0), 2),
            "noise": rnd.randint(2, 8) if rnd.random() < 0.5 else 0,
            "crf": rnd.randint(21, 29),
        }
    else:
        span = rnd.uniform(*REMUX_SPAN)
        params = {
            "tier": "remux",
            "span_frac": round(span, 4),
            "start_frac": round(rnd.uniform(0.0, 1.0 - span), 4),
            "scale": round(
                REMUX_SCALE[0]
                * (REMUX_SCALE[1] / REMUX_SCALE[0]) ** rnd.random(), 4
            ),
        }
    params["start_sec"] = round(params["start_frac"] * duration_sec, 3)
    params["span_sec"] = round(params["span_frac"] * duration_sec, 3)
    return params


def output_duration_sec(params: Dict[str, Any]) -> float:
    if params["tier"] == "remux":
        return round(params["span_sec"] * params["scale"], 3)
    return round(params["span_sec"] / params["speed"], 3)


def build_ffmpeg_cmd(params: Dict[str, Any], src: str, out: str) -> List[str]:
    """One variant = one ffmpeg invocation. Video stream only (-map 0:v:0):
    the downstream captioning consumers sample frames and never read audio,
    and stream-copied speed retimes would desync it anyway."""
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if params["tier"] == "remux":
        # -itsscale rescales input timestamps (retime without re-encoding), so
        # the input-side seek and read-duration are expressed on the *scaled*
        # timeline. -ss before -i snaps to the previous keyframe under -c copy,
        # which is what keeps every output independently decodable.
        scale = params["scale"]
        return base + [
            "-itsscale", f"{scale}",
            "-ss", f"{params['start_sec'] * scale:.3f}",
            "-t", f"{params['span_sec'] * scale:.3f}",
            "-i", src,
            "-map", "0:v:0",
            "-c", "copy",
            # keyframe-snapped trims otherwise start at negative DTS; shift to
            # zero for maximum reader compatibility
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            out,
        ]
    # crop uses floor(...*0.5)*2 so output dimensions stay even for yuv420p
    # regardless of source dimensions; x/y place the window from the seeded
    # normalized offsets.
    cs = params["crop_scale"]
    filters = [
        (
            f"crop=floor(iw*{cs}/2)*2:floor(ih*{cs}/2)*2:"
            f"floor((iw-ow)*{params['crop_x']}):floor((ih-oh)*{params['crop_y']})"
        ),
        *(["hflip"] if params["hflip"] else []),
        (
            f"eq=brightness={params['brightness']}:contrast={params['contrast']}:"
            f"saturation={params['saturation']}:gamma={params['gamma']}"
        ),
        f"hue=h={params['hue_deg']}",
        *([f"noise=alls={params['noise']}:allf=t+u"] if params["noise"] else []),
        f"setpts=PTS/{params['speed']}",
    ]
    return base + [
        "-ss", f"{params['start_sec']:.3f}",
        "-t", f"{params['span_sec']:.3f}",
        "-i", src,
        "-map", "0:v:0",
        "-vf", ",".join(filters),
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", str(params["crf"]),
        "-threads", str(FFMPEG_THREADS),
        "-movflags", "+faststart",
        out,
    ]


def probe_duration_sec(path: str) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=120,
        )
        value = float(out.stdout.strip())
        return value if value > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ray Data stage functions
# ---------------------------------------------------------------------------


def stage_source(row: Dict[str, Any], *, bucket: str, prefix: str) -> Dict[str, Any]:
    """Phase 1 map fn: one parquet-embedded mp4 -> one addressable CAIOS object.

    The transcode fan-out needs per-clip access (each clip is fetched by up to
    ~50 chunk tasks), and clips embedded in 476 MiB parquet shards can't be
    ranged-GET'd individually. Staging writes each mp4 once to a shared,
    run-independent prefix — repeat runs skip objects that already exist with
    the right size, exactly like the mirror job. Repeated chunk-task GETs of
    the same staged object are LOTA's distributed-cache best case, the same
    pattern that serves model weights in the captioning benchmark.
    """
    import hashlib

    mp4 = row.get("mp4")
    vid = hashlib.sha1(mp4[:4096]).hexdigest()[:12] if mp4 else ""
    result = {
        "video_id": vid,
        "src_key": f"{prefix}/{vid}.mp4",
        "src_bytes": len(mp4 or b""),
        "duration_sec": 0.0,
        "staged": 0,
        "put_seconds": 0.0,
        "put_end_ts": 0.0,
        "ok": 0,
        "error": "",
        "node_ip": _node_ip(),
    }
    if not mp4:
        result["error"] = "empty mp4 column"
        return result
    try:
        with tempfile.TemporaryDirectory(prefix="augstage-") as tmp:
            local = os.path.join(tmp, "src.mp4")
            with open(local, "wb") as f:
                f.write(mp4)
            duration = probe_duration_sec(local)
            if duration is None:
                result["error"] = "ffprobe failed (undecodable clip)"
                return result
            result["duration_sec"] = round(duration, 3)

            already = False
            try:
                head = s3_client().head_object(Bucket=bucket, Key=result["src_key"])
                already = head["ContentLength"] == len(mp4)
            except Exception:
                already = False
            if not already:
                put_s, end_ts = _timed_put(bucket, result["src_key"], local, len(mp4))
                result.update(staged=1, put_seconds=round(put_s, 4),
                              put_end_ts=end_ts)
            result["ok"] = 1
    except Exception as exc:  # noqa: BLE001 — one bad clip must not kill the run
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


def transcode_chunk(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Phase 2 flat_map fn: one (clip, variant-range) work item -> N variants.

    GET the staged source once (LOTA-cached after the clip's first chunk),
    run one ffmpeg per variant, and hand each finished file to background
    uploader threads — encode and PUT overlap, and the extra in-flight PUTs
    per core are what push a 120-task node into CAIOS's optimal write-
    concurrency band. Timing fields on the returned manifest rows are the
    measurement: put_seconds/put_end_ts feed the write-throughput curve and
    latency percentiles, get_seconds/get_cold the read-side split. ffmpeg or
    PUT failures produce an ok=0 row and never fail the task.
    """
    rows: List[Dict[str, Any]] = []
    # Ray Data delivers numeric row fields as numpy scalars; coerce once so
    # json.dumps(params) and downstream arithmetic stay on Python types.
    item = {
        **item,
        "duration_sec": float(item["duration_sec"]),
        "src_bytes": int(item["src_bytes"]),
        "start": int(item["start"]),
        "n": int(item["n"]),
    }
    reencode = bool(item["reencode"])

    # Resume path: petabyte runs get interrupted (quota, capacity, timeout).
    # One HEAD on the chunk's last variant is the cheap probe; only if it
    # exists do we HEAD the rest, and any gap (uploads finish out of order
    # around a crash) falls through to an idempotent full redo of the chunk.
    if item.get("skip_existing"):
        sizes = _chunk_sizes_if_complete(item)
        if sizes is not None:
            rows = []
            for offset, idx in enumerate(
                    range(item["start"], item["start"] + item["n"])):
                params = variant_params(item["video_id"], idx,
                                        item["duration_sec"], reencode)
                row = _manifest_row(item, idx, params)
                row.update(ok=1, skipped=1, out_bytes=sizes[offset],
                           out_duration_sec=output_duration_sec(params))
                rows.append(row)
            _report_progress(rows)
            return rows

    with tempfile.TemporaryDirectory(prefix="augment-") as tmp:
        src = os.path.join(tmp, "src.mp4")
        t0 = time.time()
        try:
            # Single-stream GET, deliberately. download_file's default 8-way
            # ranged fan-out put ~1,400 concurrent streams on every node when
            # a full-width fleet entered the fresh phase at once (~286k
            # streams cluster-wide) — past the measured ~800/node cliff where
            # the LOTA node agent stops responding, which stalled the fleet
            # and killed three runs. One ~15 MB source per stream at
            # 35-135 MB/s costs ~0.1-0.4 s; per-node streams stay ≈ task
            # count (~180), inside CoreWeave's ~300/node guidance.
            body = s3_client().get_object(
                Bucket=item["src_bucket"], Key=item["src_key"]
            )["Body"]
            with open(src, "wb") as f:
                for piece in iter(lambda: body.read(8 * MiB), b""):
                    f.write(piece)
            get_seconds = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            # Source fetch failed: emit one error row per variant so the
            # manifest still accounts for every requested variant.
            err = f"GET {item['src_key']}: {type(exc).__name__}: {exc}"[:500]
            rows = [
                _manifest_row(item, idx, {"tier": "reencode" if reencode else "remux"},
                              error=err)
                for idx in range(item["start"], item["start"] + item["n"])
            ]
            _report_progress(rows)
            return rows

        # Bound finished-but-not-uploaded files on local disk; release happens
        # in the uploader when the file is deleted.
        pending = threading.Semaphore(UPLOAD_THREADS_PER_TASK + 2)

        def upload(row: Dict[str, Any], path: str, size: int) -> Dict[str, Any]:
            try:
                put_s, end_ts = _timed_put(item["out_bucket"], row["out_key"],
                                           path, size)
                row.update(ok=1, put_seconds=round(put_s, 4), put_end_ts=end_ts)
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"PUT: {type(exc).__name__}: {exc}"[:500]
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
                pending.release()
            return row

        futures = []
        with ThreadPoolExecutor(max_workers=UPLOAD_THREADS_PER_TASK) as pool:
            for idx in range(item["start"], item["start"] + item["n"]):
                params = variant_params(item["video_id"], idx,
                                        item["duration_sec"], reencode)
                out = os.path.join(tmp, f"v{idx}.mp4")
                t1 = time.time()
                try:
                    proc = subprocess.run(
                        build_ffmpeg_cmd(params, src, out),
                        capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
                    )
                    if proc.returncode != 0 or not os.path.exists(out):
                        raise RuntimeError(
                            f"ffmpeg rc={proc.returncode}: "
                            f"{proc.stderr.strip()[:400]}"
                        )
                    size = os.path.getsize(out)
                    row = _manifest_row(item, idx, params)
                    row.update(
                        out_bytes=size,
                        out_duration_sec=output_duration_sec(params),
                        transcode_seconds=round(time.time() - t1, 4),
                    )
                    pending.acquire()
                    futures.append(pool.submit(upload, row, out, size))
                except Exception as exc:  # noqa: BLE001
                    if os.path.exists(out):
                        os.remove(out)
                    rows.append(_manifest_row(
                        item, idx, params, error=f"{type(exc).__name__}: {exc}"[:500]
                    ))
            rows.extend(f.result() for f in futures)

    rows[0]["get_seconds"] = round(get_seconds, 4)
    rows[0]["get_bytes"] = item["src_bytes"]
    rows[0]["get_cold"] = int(item.get("cold_get", False))
    _report_progress(rows)
    return rows


def _chunk_sizes_if_complete(item: Dict[str, Any]) -> Optional[List[int]]:
    """Sizes of every variant object in the chunk, or None if any is absent.
    Probes the last variant first so fresh chunks cost one HEAD."""
    client = s3_client()
    indices = list(range(item["start"], item["start"] + item["n"]))
    sizes: Dict[int, int] = {}
    for probe in (indices[-1], *indices[:-1]):
        try:
            head = client.head_object(Bucket=item["out_bucket"],
                                      Key=_variant_key(item, probe))
        except Exception:  # noqa: BLE001 — 404 or transient: redo the chunk
            return None
        sizes[probe] = int(head["ContentLength"])
    return [sizes[i] for i in indices]


def _variant_key(item: Dict[str, Any], idx: int) -> str:
    return f"{item['out_prefix']}/videos/{item['video_id']}/v{idx:05d}.mp4"


def _manifest_row(item: Dict[str, Any], idx: int, params: Dict[str, Any],
                  error: str = "") -> Dict[str, Any]:
    """One manifest row per variant, same keys always (stable parquet schema).
    get_seconds/get_bytes are recorded on the chunk's first row only, so
    plain column sums aggregate correctly. skipped=1 marks variants found
    already written by a previous attempt (counted in the dataset, excluded
    from this run's write throughput)."""
    return {
        "video_id": item["video_id"],
        "variant": idx,
        "tier": params.get("tier", ""),
        "out_key": _variant_key(item, idx),
        "ok": 0,
        "skipped": 0,
        "error": error,
        "params": json.dumps(params, sort_keys=True),
        "src_bytes": item["src_bytes"],
        "out_bytes": 0,
        "out_duration_sec": 0.0,
        "transcode_seconds": 0.0,
        "put_seconds": 0.0,
        "put_end_ts": 0.0,
        "get_seconds": 0.0,
        "get_bytes": 0,
        "get_cold": 0,
        "node_ip": _node_ip(),
    }

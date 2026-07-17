"""Shared pieces for the video captioning example.

Both pipelines import from this module so the *work* is identical and only the
*orchestration* differs:

  - caption_ray_core.py — hand-rolled Ray Core (tasks + actors + a manual loop)
  - caption_ray_data.py — Ray Data + the native vLLM batch integration

The pipeline has exactly two compute-heavy stages:

    mp4 bytes ──► decode_and_sample()  (CPU: decord open, uniform frame sample)
              ──► Qwen3-VL caption      (GPU: vLLM, one replica per GPU)

This module holds those stages plus the small amount of cluster plumbing both
scripts share: filesystem resolution, GPU wait, a per-node utilization sampler,
and the throughput report.
"""

import base64
import hashlib
import json
import logging
import os
import threading
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
from PIL import Image

logger = logging.getLogger("video_captioning")

# ---------------------------------------------------------------------------
# Storage + workload knobs (env-overridable so the job YAMLs can tune without
# code edits).
# ---------------------------------------------------------------------------


def require_s3_uri(uri: str, name: str) -> str:
    """Return a normalized S3 URI or fail before expensive cluster startup."""
    value = (uri or "").rstrip("/")
    if not value.startswith("s3://") or not value[5:].partition("/")[0]:
        raise ValueError(
            f"{name} must be a CoreWeave AI Object Storage URI beginning with "
            f"s3://; got {uri!r}"
        )
    return value


def ai_storage_root() -> str:
    """Resolve the dedicated object-storage prefix for this example.

    AI_STORAGE_ROOT can select another explicitly authorized CoreWeave prefix.
    By default, keep all data under the Anyscale cloud's scoped artifact path,
    which is backed by CoreWeave AI Object Storage on this cloud.
    """
    root = os.environ.get("AI_STORAGE_ROOT")
    if not root:
        artifact_root = os.environ.get("ANYSCALE_ARTIFACT_STORAGE")
        if not artifact_root:
            raise RuntimeError(
                "Set AI_STORAGE_ROOT=s3://<bucket>/<prefix> or run inside an "
                "Anyscale cloud that provides ANYSCALE_ARTIFACT_STORAGE."
            )
        root = f"{artifact_root.rstrip('/')}/video_captioning"
    return require_s3_uri(root, "AI_STORAGE_ROOT")


def ai_storage_uri(relative_path: str) -> str:
    return f"{ai_storage_root()}/{relative_path.lstrip('/')}"


def model_source() -> str:
    """Object-storage source consumed by vLLM's Run:ai model streamer."""
    return require_s3_uri(
        os.environ.get("CAPTION_MODEL")
        or ai_storage_uri("models/Qwen3-VL-8B-Instruct"),
        "CAPTION_MODEL",
    )


def dataset_source() -> str:
    return require_s3_uri(
        os.environ.get("CAPTION_DATASET") or ai_storage_uri("datasets/finevideo"),
        "CAPTION_DATASET",
    )


# Qwen3-VL-8B matches much larger VLMs on video benchmarks while fitting one
# replica per GPU (no tensor parallelism) — the throughput sweet spot.
MODEL_LOAD_FORMAT = os.environ.get("MODEL_LOAD_FORMAT", "runai_streamer")

# Frames sampled per caption window, fed to the VLM as an ordered sequence.
NUM_KEYFRAMES = int(os.environ.get("NUM_KEYFRAMES", "6"))
# Seconds of video per caption. Each clip is split into ceil(duration / W)
# windows and captioned once per window, so caption density — and total row
# count — scales with the corpus's video-hours, not its clip count. 12 s over
# FineVideo's ~12.4M video-seconds yields ~1M captions. 0 = one caption per clip.
CAPTION_WINDOW_SEC = float(os.environ.get("CAPTION_WINDOW_SEC", "12"))
# Square resize before JPEG encode: bounds vision-token count and decode cost.
IMAGE_SIZE = (int(os.environ.get("FRAME_SIZE", "448")),) * 2
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
CAPTION_MAX_TOKENS = int(os.environ.get("CAPTION_MAX_TOKENS", "256"))
CAPTION_TEMPERATURE = float(os.environ.get("CAPTION_TEMPERATURE", "0.2"))

# Dispatch batch size per caption request (vLLM continuous-batches internally).
VLM_BATCH_SIZE = int(os.environ.get("VLM_BATCH_SIZE", "32"))

SYSTEM_PROMPT = (
    "You are a video captioning assistant. You are shown an ordered sequence of "
    "frames sampled uniformly from a single video clip. Describe what happens "
    "across the clip in one to three sentences: the setting, the main subjects, "
    "and any action or change over time. Reply with the caption only — no "
    "preamble, no markdown."
)

# Schema of the output parquet, shared by both pipelines. One row per caption
# window; start/end locate the window inside its source clip.
#
# WRITE_KEYFRAMES=1 additionally persists each window's JPEG keyframes into
# the output, turning the job from caption-only output (~283 MiB over the full
# corpus) into materializing a frames+captions training dataset (~185 GiB) —
# the write-heavy variant of the benchmark.
WRITE_KEYFRAMES = os.environ.get("WRITE_KEYFRAMES", "0") == "1"
OUTPUT_COLUMNS = [
    "video_id",
    "window",
    "start_sec",
    "end_sec",
    "duration_sec",
    "fps",
    "width",
    "height",
    "num_keyframes",
    *(["keyframes"] if WRITE_KEYFRAMES else []),
    "caption",
]


def _user_prompt(num_frames: int) -> str:
    return (
        f"These {num_frames} frames are in temporal order from one video clip. "
        "Write a single concise caption describing the whole clip."
    )


# ---------------------------------------------------------------------------
# Stage 1 (CPU): decode + uniform keyframe sample
# ---------------------------------------------------------------------------


def _encode_jpeg(frame_rgb: np.ndarray) -> bytes:
    img = Image.fromarray(frame_rgb).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def decode_and_sample(
    video_bytes: bytes, video_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Open one mp4 with decord and emit one row per CAPTION_WINDOW_SEC window.

    Each window samples up to NUM_KEYFRAMES frames evenly inside its own span,
    so a five-minute clip fans out to ~25 densely-captioned rows. Windows are
    decoded in temporal order (one forward pass over the stream, bounded
    memory per window). Returns [] if the clip is undecodable / empty. The raw
    mp4 blob is never carried downstream.
    """
    from decord import VideoReader, cpu as decord_cpu

    if not video_bytes:
        return []
    vid = video_id or hashlib.sha1(video_bytes[:4096]).hexdigest()[:12]
    try:
        vr = VideoReader(BytesIO(video_bytes), ctx=decord_cpu(0))
        total_frames = len(vr)
        fps = float(vr.get_avg_fps() or 30.0)
        if total_frames == 0 or fps <= 0:
            return []

        window_frames = (
            max(1, int(CAPTION_WINDOW_SEC * fps))
            if CAPTION_WINDOW_SEC > 0
            else total_frames
        )
        rows = []
        for w, start in enumerate(range(0, total_frames, window_frames)):
            end = min(start + window_frames, total_frames)  # exclusive
            n = min(NUM_KEYFRAMES, end - start)
            if n == 1:
                # A single endpoint-inclusive sample would be frame `start` —
                # for whole-clip windows that's frame 0, often a fade-in or
                # title card. The window's midpoint is the representative one.
                indices = [(start + end - 1) // 2]
            else:
                indices = np.linspace(start, end - 1, n).astype(int).tolist()
            frames = vr.get_batch(indices).asnumpy()
            rows.append(
                {
                    "video_id": vid,
                    "window": w,
                    "start_sec": round(start / fps, 3),
                    "end_sec": round(end / fps, 3),
                    "duration_sec": round((end - start) / fps, 3),
                    "fps": round(fps, 3),
                    "width": int(frames[0].shape[1]),
                    "height": int(frames[0].shape[0]),
                    "num_keyframes": n,
                    "keyframes": [_encode_jpeg(f) for f in frames],
                }
            )
        return rows
    except Exception:
        logger.exception("decode_and_sample failed for video_id=%s", vid)
        return []


def decode_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ray Data flat_map adapter: {mp4: bytes} -> 0..N decoded window rows.

    flat_map (not map) so undecodable clips are dropped and each clip fans out
    to one row per caption window. The mp4 column is projected out — only
    keyframes + metadata flow on.
    """
    return decode_and_sample(row.get("mp4"), row.get("video_id"))


# ---------------------------------------------------------------------------
# Stage 2 (GPU): VLM prompt construction + response parsing
#
# Two message builders because the two engines take images differently:
#   - build_messages_pil -> Ray Data's ray.data.llm path wants PIL objects
#   - build_messages_b64 -> raw vLLM `llm.chat` wants image_url data URIs
# Both emit the same ordered-frame prompt, so captions are comparable.
# ---------------------------------------------------------------------------


def build_messages_pil(keyframes: List[bytes]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": _user_prompt(len(keyframes))}]
    for kf in keyframes:
        content.append({"type": "image", "image": Image.open(BytesIO(kf))})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_messages_b64(keyframes: List[bytes]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": _user_prompt(len(keyframes))}]
    for kf in keyframes:
        uri = "data:image/jpeg;base64," + base64.b64encode(kf).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def sampling_params() -> Dict[str, Any]:
    return {"temperature": CAPTION_TEMPERATURE, "max_tokens": CAPTION_MAX_TOKENS}


def clean_caption(text: str) -> str:
    """Normalize a raw VLM response into a single-line caption."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return " ".join(text.split())


# Ray Data pre/post adapters for ray.data.llm (used by caption_ray_data.py).

_PASSTHROUGH_COLS = [c for c in OUTPUT_COLUMNS if c != "caption"]


def vlm_preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decoded row -> chat-completion request + passthrough metadata columns."""
    result: Dict[str, Any] = {
        "messages": build_messages_pil(list(row["keyframes"])),
        "sampling_params": sampling_params(),
    }
    for col in _PASSTHROUGH_COLS:
        result[col] = row[col]
    return result


def vlm_postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
    """vLLM response -> output row: metadata + the cleaned caption."""
    result = {col: row[col] for col in _PASSTHROUGH_COLS}
    result["caption"] = clean_caption(row.get("generated_text", ""))
    return result


# ---------------------------------------------------------------------------
# Cluster plumbing shared by both pipelines
# ---------------------------------------------------------------------------


def _fix_vllm_shutdown_order():
    """Fix AsyncLLM.shutdown's teardown ordering (vLLM <= 0.11.x).

    Upstream kills the EngineCore *before* cancelling the output-handler task,
    so on every clean shutdown the handler briefly awaits a dead core and logs
    "AsyncLLM output_handler failed / EngineDeadError" — a failure-looking log
    on a successful run. Cancelling the handler first (and letting the loop
    process the cancellation) makes clean shutdowns clean. An *unexpected*
    core death still logs loudly, because no one called shutdown() there.

    Runs on a daemon thread that waits for vLLM to appear in sys.modules, so
    only engine workers pay for it — CPU-stage workers never import vLLM.
    """
    import sys

    mod = None
    for _ in range(600):  # engines import vLLM within their first seconds
        # sys.modules holds the module from the *start* of its import; wait
        # for the class attribute so we never touch a partially initialized
        # module (AttributeError mid-import otherwise).
        candidate = sys.modules.get("vllm.v1.engine.async_llm")
        if candidate is not None and getattr(candidate, "AsyncLLM", None) is not None:
            mod = candidate
            break
        time.sleep(1)
    if mod is None:
        return
    orig = mod.AsyncLLM.shutdown

    def shutdown(self):
        handler = getattr(self, "output_handler", None)
        if handler is not None:
            try:
                cancel = getattr(mod, "cancel_task_threadsafe", None)
                if cancel is not None:
                    cancel(handler)
                else:
                    handler.get_loop().call_soon_threadsafe(handler.cancel)
                # Wait for the loop to actually process the cancellation; at
                # stream end it may still be busy with final-batch bookkeeping.
                for _ in range(1000):
                    if handler.done():
                        break
                    time.sleep(0.01)
            except Exception:
                pass
        return orig(self)

    mod.AsyncLLM.shutdown = shutdown


def _graceful_vllm_shutdown():
    """atexit hook: stop vLLM engines cleanly before interpreter teardown.

    Ray tears actors down by letting their process exit. Without this hook the
    engine's background output handler watches its EngineCore die mid-teardown
    and spams `EngineDeadError` at ERROR level, and destructor-time logging
    races interpreter shutdown ("sys.meta_path is None") — so every successful
    job ends with failure-looking logs. `AsyncLLM.shutdown()` cancels the
    output handler *before* killing the core, which is the ordering the
    destructor path can't guarantee.
    """
    import sys

    if "vllm" not in sys.modules:
        return  # not an engine worker; nothing to shut down
    import gc

    # Nothing useful is logged past this point; logging.disable() beats the
    # explicit per-module levels vLLM sets on its child loggers. The unraisable
    # hook covers "Exception ignored in __del__" prints, which the interpreter
    # writes to stderr directly during finalization, bypassing logging.
    logging.disable(logging.CRITICAL)
    sys.unraisablehook = lambda *args: None
    try:
        from vllm.v1.engine.async_llm import AsyncLLM

        engines = [o for o in gc.get_objects() if isinstance(o, AsyncLLM)]
    except Exception:
        return
    for engine in engines:
        try:
            engine.shutdown()
        except Exception:
            pass


def worker_setup():
    """Per-worker startup hook (worker_process_setup_hook / actor init).

    Spreads each process's VLLM_PORT probe base so simultaneous engine starts
    on one node don't race to the same torch.distributed rendezvous port — at
    8 engines/node starting in the same second, two engines can otherwise grab
    the same port and die with EADDRINUSE. It also gives every worker a private
    vLLM assets cache. Run:ai's object-storage loader creates and removes a
    model-specific directory there; sharing it among eight engines on a node
    causes concurrent mkdir/rmtree failures during startup. Finally, it
    registers a graceful engine shutdown for actor exit so successful jobs
    don't end with EngineDeadError teardown noise in their logs.
    """
    import atexit
    import socket

    def _pick_free_port() -> int:
        # A blind PID-hash over 20k slots collides somewhere on a big fleet
        # (8 engines/node × 200 nodes booting in the same second ≈ 27% chance
        # of an EADDRINUSE that kills an engine — and one engine-creation
        # death is fatal to the whole actor pool). Claim a port with an
        # atomic O_EXCL lockfile so cooperating workers on a node can never
        # race each other, then probe-bind to dodge unrelated listeners.
        locks = "/tmp/vllm-port-locks"
        os.makedirs(locks, exist_ok=True)
        seed = os.getpid() * 211 + time.time_ns() % 104729
        for i in range(512):
            cand = 30000 + (seed + i * 1009) % 20000
            try:
                fd = os.open(os.path.join(locks, str(cand)),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except OSError:
                continue  # claimed by a sibling worker
            try:
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probe.bind(("", cand))
                probe.close()
                return cand
            except OSError:
                continue  # unrelated listener; lockfile stays, skip the port
        return 30000 + seed % 20000

    os.environ.setdefault("VLLM_PORT", str(_pick_free_port()))
    assets_root = os.environ.get("VLLM_ASSETS_CACHE_BASE", "/tmp/vllm-assets")
    os.environ["VLLM_ASSETS_CACHE"] = os.path.join(
        assets_root, f"worker-{os.getpid()}"
    )
    atexit.register(_graceful_vllm_shutdown)
    threading.Thread(target=_fix_vllm_shutdown_order, daemon=True).start()


def filesystem_and_path(uri: str) -> Tuple[Any, str]:
    """Resolve a CoreWeave S3 URI to (filesystem, relative path).

    CoreWeave AI Object Storage requires virtual-hosted addressing. Anyscale
    supplies credentials; the job YAML points clients at the in-cluster LOTA
    endpoint for the data path.
    """
    import pyarrow.fs as pafs

    uri = require_s3_uri(uri, "storage URI")
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get(
        "AWS_ENDPOINT_URL"
    )
    if endpoint:
        scheme, host = endpoint.split("://", 1)
        fs = pafs.S3FileSystem(
            endpoint_override=host,
            scheme=scheme,
            force_virtual_addressing=True,
            retry_strategy=pafs.AwsStandardS3RetryStrategy(
                max_attempts=int(os.environ.get("S3_MAX_ATTEMPTS", "8"))
            ),
            region=(
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or os.environ.get("ANYSCALE_CLOUD_STORAGE_BUCKET_REGION")
                or "US-EAST-14A"
            ),
        )
        return fs, uri.split("://", 1)[-1]
    return pafs.FileSystem.from_uri(uri)


def require_mirror_complete(uri: str, name: str) -> None:
    """Require the success marker written only after a full repository copy."""
    import pyarrow.fs as pafs

    fs, path = filesystem_and_path(uri)
    marker = fs.get_file_info(f"{path}/_SUCCESS")
    if marker.type != pafs.FileType.File:
        raise RuntimeError(
            f"{name} mirror is incomplete at {uri}: _SUCCESS is missing. "
            "Run job_mirror_to_ai_storage.yaml first."
        )


def default_output(pipeline: str) -> str:
    """Return a unique output prefix in CoreWeave AI Object Storage."""
    return ai_storage_uri(
        f"outputs/{pipeline}/{time.strftime('%Y%m%d-%H%M%S')}"
    )


def wait_for_gpus(min_gpus: int, timeout_s: int = 1800) -> int:
    """Block until at least `min_gpus` GPUs have joined the Ray cluster.

    The worker pool has a low minimum so the job starts cheaply. Submit an
    explicit resource request before polling; without demand, the autoscaler
    has no reason to grow the pool while the driver waits. The request remains
    active for the lifetime of this benchmark job.
    """
    if min_gpus > 0:
        from ray.autoscaler.sdk import request_resources

        request_resources(bundles=[{"GPU": 1}] * min_gpus)
        logger.info("Requested %d GPUs from the Ray autoscaler.", min_gpus)

    deadline = time.time() + timeout_s
    seen = 0
    while time.time() < deadline:
        seen = int(ray.cluster_resources().get("GPU", 0))
        if seen >= min_gpus:
            return seen
        logger.info("Waiting for GPUs to join the cluster (have %d, want %d)...", seen, min_gpus)
        time.sleep(5)
    return seen


# ---------------------------------------------------------------------------
# Utilization sampling + throughput report
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=0)
class _NodeSampler:
    """Samples GPU utilization (NVML) and CPU utilization (psutil) on one node
    on a fixed interval until result() is called."""

    def __init__(self, interval_s: float):
        self.gpu: List[float] = []
        self.cpu: List[float] = []
        self._stop = threading.Event()
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml, self._n_gpu = pynvml, pynvml.nvmlDeviceGetCount()
        except Exception:
            self._nvml, self._n_gpu = None, 0
        import psutil

        self._psutil = psutil
        psutil.cpu_percent(interval=None)  # prime the counter
        threading.Thread(target=self._loop, args=(interval_s,), daemon=True).start()

    def _loop(self, interval_s: float):
        while not self._stop.wait(interval_s):
            if self._nvml is not None:
                per_gpu = []
                for i in range(self._n_gpu):
                    try:
                        handle = self._nvml.nvmlDeviceGetHandleByIndex(i)
                        per_gpu.append(float(self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu))
                    except Exception:
                        pass
                if per_gpu:
                    self.gpu.append(sum(per_gpu) / len(per_gpu))
            self.cpu.append(float(self._psutil.cpu_percent(interval=None)))

    def result(self) -> Dict[str, List[float]]:
        self._stop.set()
        return {"gpu": self.gpu, "cpu": self.cpu}


def _summary(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"mean": 0.0, "p95": 0.0, "n": 0}
    s = sorted(samples)
    return {
        "mean": round(sum(s) / len(s), 1),
        "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 1),
        "n": len(s),
    }


class UtilizationMonitor:
    """Pins one _NodeSampler to every worker node; stop() returns cluster-wide
    GPU/CPU utilization (mean/p95). This answers the balance question the
    example poses: are the engines busy, and is decode keeping them fed?

    Also samples *held* GPUs (scheduled to actors/tasks, i.e. cluster GPUs
    minus available GPUs) from the driver. The integral of that curve —
    GPU-seconds held — is what distinguishes an autoscaled engine pool from a
    pre-provisioned fleet: both may finish in similar wall time, but they hold
    very different amounts of GPU while doing it."""

    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self._samplers: List[Any] = []
        self._sampled_nodes: set = set()
        self._held: List[Tuple[float, float]] = []  # (timestamp, GPUs held)
        self._provisioned: List[Tuple[float, float]] = []  # (timestamp, GPUs in cluster)
        self._stop_evt = threading.Event()
        self._held_thread: Optional[threading.Thread] = None

    def _attach_samplers(self):
        """Pin a sampler to every alive worker node not yet covered. Called
        periodically so nodes that join mid-run (production mode: the node
        autoscaler follows the engine pool) are sampled too."""
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        for node in ray.nodes():
            nid = node["NodeID"]
            if not node.get("Alive") or nid == self._head or nid in self._sampled_nodes:
                continue
            strategy = NodeAffinitySchedulingStrategy(node_id=nid, soft=False)
            self._samplers.append(
                _NodeSampler.options(scheduling_strategy=strategy).remote(self.interval_s)
            )
            self._sampled_nodes.add(nid)

    def _sample_held(self):
        n = 0
        while not self._stop_evt.wait(self.interval_s):
            try:
                total = float(ray.cluster_resources().get("GPU", 0.0))
                avail = float(ray.available_resources().get("GPU", 0.0))
                now = time.time()
                self._held.append((now, max(0.0, total - avail)))
                self._provisioned.append((now, total))
                self._attach_samplers()
                n += 1
                if n % 12 == 0:  # one machine-parseable heartbeat per minute
                    store_total = float(ray.cluster_resources().get("object_store_memory", 0.0))
                    store_avail = float(ray.available_resources().get("object_store_memory", 0.0))
                    print(
                        "PIPELINE_STAT "
                        + json.dumps({
                            "elapsed_s": round(now - self._held[0][0], 1),
                            "gpus_held": round(max(0.0, total - avail), 1),
                            "gpus_provisioned": round(total, 1),
                            "object_store_used_frac": round(
                                1 - store_avail / store_total, 3) if store_total else None,
                        }),
                        flush=True,
                    )
            except Exception:
                pass

    def start(self) -> "UtilizationMonitor":
        self._head = ray.get_runtime_context().get_node_id()
        self._attach_samplers()
        now = time.time()
        self._held.append((now, 0.0))
        self._provisioned.append((now, float(ray.cluster_resources().get("GPU", 0.0))))
        self._held_thread = threading.Thread(target=self._sample_held, daemon=True)
        self._held_thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        self._stop_evt.set()
        if self._held_thread is not None:
            self._held_thread.join(timeout=self.interval_s + 5)
        refs = [s.result.remote() for s in self._samplers]
        # Bounded wait: a sampler on a node that scaled down would hang forever.
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=60) if refs else ([], [])
        gpu, cpu = [], []
        for ref in ready:
            try:
                res = ray.get(ref)
                gpu += res["gpu"]
                cpu += res["cpu"]
            except Exception:
                pass

        def integrate(series):
            # Step-integrate the curve up to now.
            pts = series + [(time.time(), series[-1][1])] if series else []
            gpu_seconds = sum(
                v * (t_next - t) for (t, v), (t_next, _) in zip(pts, pts[1:])
            )
            values = [v for _, v in series]
            return {
                "mean": round(sum(values) / len(values), 1) if values else 0.0,
                "peak": round(max(values), 1) if values else 0.0,
                "gpu_seconds": round(gpu_seconds, 1),
            }

        return {
            "gpu_util": _summary(gpu),
            "cpu_util": _summary(cpu),
            "gpus_held": integrate(self._held),
            "gpus_provisioned": integrate(self._provisioned),
        }


def write_report(
    pipeline: str,
    input_uri: str,
    num_gpus: int,
    num_captions: int,
    video_seconds: float,
    wall_seconds: float,
    utilization: Dict[str, Any],
    output_uri: str,
):
    """Log the headline numbers and write report.json next to the captions."""
    wall = max(wall_seconds, 1e-9)
    # Echo the resolved knobs into the report: a benchmark number without its
    # configuration is unreproducible, and reconstructing which run had which
    # knobs from session memory is how comparisons go wrong.
    config_keys = [
        "EXPECTED_GPUS", "PREPROVISION_GPUS", "DATASET_REPEATS", "PASS_SHARDS",
        "POOL_FLOOR", "CAPTION_WINDOW_SEC", "NUM_KEYFRAMES", "FRAME_SIZE",
        "VLM_BATCH_SIZE", "MAX_MODEL_LEN", "WRITE_KEYFRAMES", "MAX_BUFFERED_ROWS",
    ]
    report = {
        "pipeline": pipeline,
        "config": {k: os.environ[k] for k in config_keys if k in os.environ},
        "model": model_source(),
        "dataset": input_uri,
        "output": output_uri,
        "num_gpus": num_gpus,
        "num_captions": num_captions,
        "wall_seconds": round(wall_seconds, 1),
        "captions_per_sec": round(num_captions / wall, 3),
        "captions_per_gpu_per_sec": round(num_captions / wall / max(num_gpus, 1), 4),
        "video_hours_per_wall_hour": round(video_seconds / wall, 2),
        "utilization": utilization,
    }
    gpu_seconds = utilization.get("gpus_held", {}).get("gpu_seconds", 0.0)
    if gpu_seconds > 0:
        report["gpu_seconds_held"] = gpu_seconds
        report["captions_per_gpu_hour"] = round(num_captions / (gpu_seconds / 3600.0), 1)
    provisioned = utilization.get("gpus_provisioned", {}).get("gpu_seconds", 0.0)
    if provisioned > 0:
        # The dedicated-cluster cost lens: GPU-seconds that existed in the
        # cluster, whether or not the pool held them. In production mode
        # (PREPROVISION_GPUS=0) nodes track the pool and this approaches
        # gpu_seconds_held; in benchmark mode it is ~fleet × wall.
        report["gpu_seconds_provisioned"] = provisioned
    logger.info("=" * 62)
    logger.info("THROUGHPUT REPORT  (%s, %d GPUs)", pipeline, num_gpus)
    logger.info("=" * 62)
    logger.info("  captions generated      : %d", num_captions)
    logger.info("  wall time               : %.1f s", wall_seconds)
    logger.info("  captions/sec            : %s", report["captions_per_sec"])
    logger.info("  captions/GPU/sec        : %s", report["captions_per_gpu_per_sec"])
    logger.info("  video-hours / wall-hour : %s", report["video_hours_per_wall_hour"])
    if "gpu_seconds_held" in report:
        held = utilization["gpus_held"]
        logger.info("  GPUs held mean/peak     : %s / %s", held["mean"], held["peak"])
        logger.info("  GPU-seconds held        : %s", report["gpu_seconds_held"])
        logger.info("  captions / GPU-hour     : %s", report["captions_per_gpu_hour"])
    logger.info("  GPU util mean/p95       : %s / %s",
                utilization["gpu_util"]["mean"], utilization["gpu_util"]["p95"])
    logger.info("  CPU util mean/p95       : %s / %s",
                utilization["cpu_util"]["mean"], utilization["cpu_util"]["p95"])
    logger.info("=" * 62)
    # Also print(): `anyscale job logs` shows driver stdout, not the log stream.
    print("THROUGHPUT_REPORT " + json.dumps(report), flush=True)

    body = json.dumps(report, indent=2).encode()
    fs, path = filesystem_and_path(output_uri)
    with fs.open_output_stream(f"{path}/report.json") as f:
        f.write(body)
    logger.info("Wrote report.json to %s", output_uri)

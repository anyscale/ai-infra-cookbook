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
# Model + workload knobs (env-overridable so the job YAMLs can tune without
# code edits).
# ---------------------------------------------------------------------------

# Qwen3-VL-8B matches much larger VLMs on video benchmarks while fitting one
# replica per GPU (no tensor parallelism) — the throughput sweet spot.
MODEL_SOURCE = os.environ.get("CAPTION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

# Frames sampled per video, fed to the VLM as an ordered image sequence.
NUM_KEYFRAMES = int(os.environ.get("NUM_KEYFRAMES", "6"))
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

# Schema of the output parquet, shared by both pipelines.
OUTPUT_COLUMNS = [
    "video_id",
    "duration_sec",
    "fps",
    "width",
    "height",
    "num_keyframes",
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
) -> Optional[Dict[str, Any]]:
    """Open one mp4 with decord, sample NUM_KEYFRAMES evenly, JPEG-encode them.

    Returns a dict with keyframe bytes + lightweight metadata, or None if the
    clip is undecodable / empty. The raw mp4 blob is never carried downstream.
    """
    from decord import VideoReader, cpu as decord_cpu

    if not video_bytes:
        return None
    vid = video_id or hashlib.sha1(video_bytes[:4096]).hexdigest()[:12]
    try:
        vr = VideoReader(BytesIO(video_bytes), ctx=decord_cpu(0))
        total_frames = len(vr)
        fps = float(vr.get_avg_fps() or 30.0)
        if total_frames == 0 or fps <= 0:
            return None

        n = min(NUM_KEYFRAMES, total_frames)
        indices = np.linspace(0, total_frames - 1, n).astype(int).tolist()
        frames = vr.get_batch(indices).asnumpy()
        return {
            "video_id": vid,
            "duration_sec": round(total_frames / fps, 3),
            "fps": round(fps, 3),
            "width": int(frames[0].shape[1]),
            "height": int(frames[0].shape[0]),
            "num_keyframes": n,
            "keyframes": [_encode_jpeg(f) for f in frames],
        }
    except Exception:
        logger.exception("decode_and_sample failed for video_id=%s", vid)
        return None


def decode_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ray Data flat_map adapter: {mp4: bytes} -> 0 or 1 decoded rows.

    flat_map (not map) so undecodable clips are dropped instead of failing the
    block. The mp4 column is projected out — only keyframes + metadata flow on.
    """
    decoded = decode_and_sample(row.get("mp4"), row.get("video_id"))
    return [decoded] if decoded is not None else []


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


def worker_setup():
    """Per-worker startup hook (worker_process_setup_hook / actor init).

    Spreads each process's VLLM_PORT probe base so simultaneous engine starts
    on one node don't race to the same torch.distributed rendezvous port — at
    8 engines/node starting in the same second, two engines can otherwise grab
    the same port and die with EADDRINUSE. setdefault so an explicit VLLM_PORT
    env wins.
    """
    os.environ.setdefault("VLLM_PORT", str(30000 + (os.getpid() * 211) % 20000))


def filesystem_and_path(uri: str) -> Tuple[Any, str]:
    """Resolve a URI to (pyarrow filesystem, filesystem-relative path).

    Clusters that point S3 clients at a custom endpoint (AWS_ENDPOINT_URL_S3 —
    e.g. CoreWeave object storage) need the filesystem built explicitly: those
    endpoints require virtual-hosted addressing, which pyarrow's default
    path-style requests don't use.
    """
    import pyarrow.fs as pafs

    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3")
    if uri.startswith("s3://") and endpoint:
        scheme, host = endpoint.split("://", 1)
        fs = pafs.S3FileSystem(
            endpoint_override=host,
            scheme=scheme,
            force_virtual_addressing=True,
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )
        return fs, uri.split("://", 1)[-1]
    return pafs.FileSystem.from_uri(uri)


def default_output(pipeline: str) -> str:
    """A unique output prefix for one run, preferring the cloud's object storage
    (ANYSCALE_ARTIFACT_STORAGE is a cluster-wide s3://... prefix with credentials
    already wired). Falls back to node-local /tmp outside Anyscale."""
    base = os.environ.get("ANYSCALE_ARTIFACT_STORAGE", "/tmp").rstrip("/")
    return f"{base}/video_captioning/{pipeline}/{time.strftime('%Y%m%d-%H%M%S')}"


def wait_for_gpus(min_gpus: int, timeout_s: int = 1800) -> int:
    """Block until at least `min_gpus` GPUs have joined the Ray cluster.

    GPU worker nodes provision after the driver starts, so reading
    `ray.cluster_resources()` immediately would report 0 GPUs. Poll until the
    pool is up (or timeout), then return the GPU count actually seen.
    """
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
    example poses: are the engines busy, and is decode keeping them fed?"""

    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self._samplers: List[Any] = []

    def start(self) -> "UtilizationMonitor":
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        head = ray.get_runtime_context().get_node_id()
        for node in ray.nodes():
            if not node.get("Alive") or node["NodeID"] == head:
                continue  # the head node is control-plane only
            strategy = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            self._samplers.append(
                _NodeSampler.options(scheduling_strategy=strategy).remote(self.interval_s)
            )
        return self

    def stop(self) -> Dict[str, Any]:
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
        return {"gpu_util": _summary(gpu), "cpu_util": _summary(cpu)}


def write_report(
    pipeline: str,
    num_gpus: int,
    num_captions: int,
    video_seconds: float,
    wall_seconds: float,
    utilization: Dict[str, Any],
    output_uri: str,
):
    """Log the headline numbers and write report.json next to the captions."""
    wall = max(wall_seconds, 1e-9)
    report = {
        "pipeline": pipeline,
        "model": MODEL_SOURCE,
        "num_gpus": num_gpus,
        "num_captions": num_captions,
        "wall_seconds": round(wall_seconds, 1),
        "captions_per_sec": round(num_captions / wall, 3),
        "captions_per_gpu_per_sec": round(num_captions / wall / max(num_gpus, 1), 4),
        "video_hours_per_wall_hour": round(video_seconds / wall, 2),
        "utilization": utilization,
    }
    logger.info("=" * 62)
    logger.info("THROUGHPUT REPORT  (%s, %d GPUs)", pipeline, num_gpus)
    logger.info("=" * 62)
    logger.info("  captions generated      : %d", num_captions)
    logger.info("  wall time               : %.1f s", wall_seconds)
    logger.info("  captions/sec            : %s", report["captions_per_sec"])
    logger.info("  captions/GPU/sec        : %s", report["captions_per_gpu_per_sec"])
    logger.info("  video-hours / wall-hour : %s", report["video_hours_per_wall_hour"])
    logger.info("  GPU util mean/p95       : %s / %s",
                utilization["gpu_util"]["mean"], utilization["gpu_util"]["p95"])
    logger.info("  CPU util mean/p95       : %s / %s",
                utilization["cpu_util"]["mean"], utilization["cpu_util"]["p95"])
    logger.info("=" * 62)
    # Also print(): `anyscale job logs` shows driver stdout, not the log stream.
    print("THROUGHPUT_REPORT " + json.dumps(report), flush=True)

    body = json.dumps(report, indent=2).encode()
    if "://" in output_uri:
        fs, path = filesystem_and_path(output_uri)
        with fs.open_output_stream(f"{path}/report.json") as f:
            f.write(body)
    else:
        os.makedirs(output_uri, exist_ok=True)
        with open(os.path.join(output_uri, "report.json"), "wb") as f:
            f.write(body)
    logger.info("Wrote report.json to %s", output_uri)

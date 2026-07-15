"""Stage functions for the streaming video curation pipeline.

Each function (or actor class) is one Ray Data operator. Operators are
assembled into the pipeline in `video_curation.py`. Data flow:

    {mp4: bytes}                     # from HF parquet stream
        -> flat_map(process_video_bytes)
    clip rows with keyframes (1 video -> ~10 clips)
        -> vLLM processor (vlm_preprocess + vlm_postprocess)
    same rows enriched with category / is_safe / vlm_description
        -> filter(is_safe)
        -> map_batches(CLIPEmbedder)
    rows with 512-d CLIP embedding, written to parquet
"""

import hashlib
import json
import logging
from io import BytesIO
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu as decord_cpu
from PIL import Image

logger = logging.getLogger("video_curation")


# ===========================================================================
# Stage 1 (fused): process_video_bytes
#
# INPUT  row: {mp4: bytes}
# OUTPUT 0..N rows per input (one per quality-passing scene clip):
#   {video_id, clip_index, start_frame, end_frame, start_sec, end_sec,
#    clip_duration_sec, clip_num_frames, fps, width, height,
#    brightness, contrast, sharpness, motion_score,
#    keyframe_bytes,          # 384x384 JPEG for VLM
#    keyframe_bytes_list}     # N 224x224 JPEGs for CLIP (N=3)
#
# Fan-out: 1 mp4 -> avg ~10 clips.
# ===========================================================================

HISTOGRAM_DIFF_THRESHOLD = 0.5   # Bhattacharyya distance for a scene cut
SAMPLE_INTERVAL = 20             # scene-detect sample stride (every 20th frame)
MIN_SCENE_FRAMES = 30            # reject clips shorter than ~1 sec @ 30fps

MIN_BRIGHTNESS = 30.0
MAX_BRIGHTNESS = 240.0
MIN_CONTRAST = 20.0
MIN_SHARPNESS = 50.0
MIN_MOTION_SCORE = 0.5
MIN_CLIP_DURATION_SEC = 1.0
NUM_SAMPLE_FRAMES = 5            # frames sampled per clip for quality signals

NUM_KEYFRAMES_PER_CLIP = 3       # keyframes emitted per clip for CLIP
IMAGE_SIZE = (384, 384)          # Qwen2.5-VL input resolution
CLIP_IMAGE_SIZE = (224, 224)     # CLIP ViT-B/32 input resolution


def _compute_histogram(frame_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _quality_signals(frames_rgb: np.ndarray) -> Dict[str, float]:
    brightnesses, contrasts, sharpnesses, flow_magnitudes = [], [], [], []
    prev_gray = None
    for frame_rgb in frames_rgb:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        brightnesses.append(float(np.mean(gray)))
        contrasts.append(float(np.std(gray)))
        sharpnesses.append(float(np.var(cv2.Laplacian(gray, cv2.CV_64F))))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 1, 15, 2, 5, 1.1, 0
            )
            flow_magnitudes.append(
                float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            )
        prev_gray = gray
    return {
        "brightness": round(float(np.mean(brightnesses)), 2),
        "contrast": round(float(np.mean(contrasts)), 2),
        "sharpness": round(float(np.mean(sharpnesses)), 2),
        "motion_score": round(
            float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0, 2
        ),
    }


def _passes_quality(sig: Dict[str, float], duration_sec: float) -> bool:
    if duration_sec < MIN_CLIP_DURATION_SEC:
        return False
    if sig["brightness"] < MIN_BRIGHTNESS or sig["brightness"] > MAX_BRIGHTNESS:
        return False
    if sig["contrast"] < MIN_CONTRAST:
        return False
    if sig["sharpness"] < MIN_SHARPNESS:
        return False
    if sig["motion_score"] < MIN_MOTION_SCORE:
        return False
    return True


def _encode_jpeg(frame_rgb: np.ndarray, size: tuple[int, int], quality: int = 85) -> bytes:
    img = Image.fromarray(frame_rgb).resize(size, Image.Resampling.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def process_video_bytes(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scene-detect, quality-filter and keyframe-encode one mp4 in a single
    decord open. It creates 0..N clip rows; the mp4 blob is not stored."""
    video_bytes = row["mp4"] 
    video_id = (
        row.get("video_id")
        or hashlib.sha1(video_bytes[:4096]).hexdigest()[:12]
    )

    results: List[Dict[str, Any]] = []
    try:
        vr = VideoReader(BytesIO(video_bytes), ctx=decord_cpu(0))
        total_frames = len(vr)
        fps = float(vr.get_avg_fps() or 30.0)
        if total_frames == 0 or fps <= 0:
            return results

        first = vr[0].asnumpy()
        height, width = int(first.shape[0]), int(first.shape[1])

        sample_indices = list(range(0, total_frames, SAMPLE_INTERVAL))
        if not sample_indices:
            return results
        sample_frames = vr.get_batch(sample_indices).asnumpy()

        scene_boundaries = [0]
        prev_hist = None
        for i, frame in enumerate(sample_frames):
            hist = _compute_histogram(frame)
            if prev_hist is not None:
                diff = cv2.compareHist(
                    prev_hist.reshape(-1, 1).astype(np.float32),
                    hist.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA,
                )
                if diff > HISTOGRAM_DIFF_THRESHOLD:
                    scene_boundaries.append(sample_indices[i])
            prev_hist = hist
        scene_boundaries.append(total_frames)

        for clip_index in range(len(scene_boundaries) - 1):
            start_frame = scene_boundaries[clip_index]
            end_frame = scene_boundaries[clip_index + 1]
            num_frames = end_frame - start_frame
            if num_frames < MIN_SCENE_FRAMES:
                continue

            start_sec = start_frame / fps
            end_sec = end_frame / fps
            duration_sec = end_sec - start_sec

            mid_frame = min((start_frame + end_frame) // 2, total_frames - 1)
            clip_keyframe_indices = np.linspace(
                start_frame,
                min(end_frame - 1, total_frames - 1),
                min(NUM_KEYFRAMES_PER_CLIP, num_frames),
            ).astype(int).tolist()
            quality_indices = np.linspace(
                start_frame,
                min(end_frame - 1, total_frames - 1),
                min(NUM_SAMPLE_FRAMES, num_frames),
            ).astype(int).tolist()

            all_idx = sorted(
                set([mid_frame] + clip_keyframe_indices + quality_indices)
            )
            frames = vr.get_batch(all_idx).asnumpy()
            idx_map = {idx: i for i, idx in enumerate(all_idx)}

            quality_frames = np.stack(
                [frames[idx_map[i]] for i in quality_indices]
            )
            sig = _quality_signals(quality_frames)
            if not _passes_quality(sig, duration_sec):
                continue

            keyframe_bytes = _encode_jpeg(frames[idx_map[mid_frame]], IMAGE_SIZE)
            keyframe_bytes_list = [
                _encode_jpeg(frames[idx_map[i]], CLIP_IMAGE_SIZE)
                for i in clip_keyframe_indices
            ]

            results.append(
                {
                    "video_id": video_id,
                    "clip_index": clip_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "clip_duration_sec": round(duration_sec, 3),
                    "clip_num_frames": num_frames,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "brightness": sig["brightness"],
                    "contrast": sig["contrast"],
                    "sharpness": sig["sharpness"],
                    "motion_score": sig["motion_score"],
                    "keyframe_bytes": keyframe_bytes,
                    "keyframe_bytes_list": keyframe_bytes_list,
                }
            )
    except Exception:
        logger.exception(f"process_video_bytes failed for {video_id}")
    return results


# ===========================================================================
# Stage 2: vlm_preprocess / vlm_postprocess (wraps around the vLLM operator)
#
# INPUT  clip row with {keyframe_bytes, ...passthrough}
# OUTPUT same columns + {category, is_safe, vlm_description, vlm_quality}
#
# Fan-out: 1:1 (no rows dropped here; the is_safe filter runs afterwards).
# ===========================================================================

MODEL_SOURCE = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_MODEL_LEN = 2048
VLM_BATCH_SIZE = 64

SYSTEM_PROMPT = (
    "You are a video content analyst. Analyze the provided keyframe from a "
    "video clip and respond ONLY with valid JSON (no markdown, no extra text)."
)
USER_PROMPT = (
    "Analyze this video keyframe. Respond with a JSON object:\n"
    '{"category": "<one of: education, entertainment, sports, nature, '
    'news, cooking, music, gaming, vlog, tutorial, other>", '
    '"is_safe": <true or false>, '
    '"description": "<one sentence describing the scene>", '
    '"visual_quality": "<one of: high, medium, low>"}'
)

# Columns propagated through the vLLM processor back into downstream stages.
# `keyframe_bytes_list` must survive the VLM stage because CLIP needs it.
_PASSTHROUGH_COLS = [
    "video_id",
    "clip_index",
    "start_frame",
    "end_frame",
    "start_sec",
    "end_sec",
    "clip_duration_sec",
    "clip_num_frames",
    "fps",
    "width",
    "height",
    "brightness",
    "contrast",
    "sharpness",
    "motion_score",
    "keyframe_bytes_list",
]


def vlm_preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
    """Clip row -> vLLM chat-completion request + passthrough columns."""
    image = Image.open(BytesIO(row["keyframe_bytes"]))
    result = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image", "image": image},
                ],
            },
        ],
        "sampling_params": {"temperature": 0.1, "max_tokens": 200},
    }
    for col in _PASSTHROUGH_COLS:
        if col in row:
            result[col] = row[col]
    return result


def vlm_postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
    """vLLM response -> clip row enriched with category/is_safe/description."""
    generated = row.get("generated_text", "")
    result = {col: row[col] for col in _PASSTHROUGH_COLS if col in row}
    result.update(
        {
            "category": "other",
            "is_safe": True,
            "vlm_description": "",
            "vlm_quality": "medium",
        }
    )
    try:
        text = generated.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        result["category"] = parsed.get("category", "other")
        result["is_safe"] = parsed.get("is_safe", True)
        result["vlm_description"] = parsed.get("description", "")
        result["vlm_quality"] = parsed.get("visual_quality", "medium")
    except (json.JSONDecodeError, KeyError):
        result["vlm_description"] = generated[:200]
    return result


# ===========================================================================
# Stage 3: CLIPEmbedder (stateful actor)
#
# INPUT  batch with {keyframe_bytes_list: List[List[bytes]], ...}
# OUTPUT same batch + {embedding: List[np.ndarray(512)]}; `keyframe_bytes_list`
#        is dropped after use so rows going to write carry only metadata + embed.
#
# Fan-out: 1:1 (batch-preserving).
# ===========================================================================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
CLIP_BATCH_SIZE = 32


class CLIPEmbedder:
    """Loads CLIP ViT-B/32 once per actor and reuses it for every batch."""

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor

        # Explicitly set device to "cpu" (even though this is the default) to avoid accidental GPU usage.
        self.device = "cpu"
        torch.set_num_threads(2)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    def __call__(self, batch: Dict[str, List]) -> Dict[str, List]:
        all_embeddings = []
        for keyframe_list in batch["keyframe_bytes_list"]:
            # Ray Data packs list columns as np.ndarray in batches, so use
            # an explicit length check (truthiness is ambiguous on arrays).
            if keyframe_list is None or len(keyframe_list) == 0:
                all_embeddings.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
                continue
            images = [
                Image.open(BytesIO(kb)).convert("RGB") for kb in keyframe_list
            ]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.mean(dim=0).cpu().numpy().astype(np.float32))
        batch["embedding"] = all_embeddings
        batch.pop("keyframe_bytes_list", None)
        return batch

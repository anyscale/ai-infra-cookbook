# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for downloading and preparing image datasets as WebDataset tar files."""

from __future__ import annotations

import io
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import requests
from loguru import logger
from PIL import Image
import webdataset as wds


def download_single_image(url: str, session: requests.Session) -> bytes | None:
    """Download a single image, returning bytes or None on failure."""
    try:
        response = session.get(url, timeout=5, stream=True)
        return response.content if response.status_code == 200 else None
    except Exception:
        return None


def image_download_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Download a batch of images using ThreadPoolExecutor for parallelism."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=100, pool_maxsize=100, max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Use ThreadPoolExecutor for parallel downloads within this batch
    # 50 threads means 50 concurrent downloads per Ray task
    with ThreadPoolExecutor(max_workers=100) as executor:
        batch["bytes"] = list(executor.map(lambda url: download_single_image(url, session), batch["url"]))

    return batch


def process_image(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate downloaded image bytes, convert to JPEG, and drop failures.

    Returns a single-element list on success or an empty list to drop the row.
    Designed for use with Ray Data's flat_map.
    """
    image_bytes = row.get("bytes")
    if not image_bytes:
        return []

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes))

        # Robust RGB conversion for ALL image modes (L, LA, P, PA, RGBA, CMYK, etc.)
        # This ensures CLIP gets 3-channel images
        if img.mode != "RGB":
            if img.mode == "P":
                img = img.convert("RGBA")
            # For any mode with alpha, composite onto white background
            if img.mode in ("RGBA", "LA", "PA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                # Use alpha channel as mask
                if img.mode == "LA":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert("RGB")

        if img.mode != "RGB" or img.size[0] < 3 or img.size[1] < 3:
            return []

        jpeg_buffer = io.BytesIO()
        img.save(jpeg_buffer, format="JPEG", quality=95)
        row["jpeg_bytes"] = jpeg_buffer.getvalue()
        return [row]
    except Exception:
        return []


def write_tar_batch(batch: dict[str, Any], output_dir: str) -> dict[str, Any]:
    """Write a batch of images to a WebDataset tar shard."""
    import ray

    node_id = ray.get_runtime_context().get_node_id()[:8]
    shard_id = f"{node_id}_{uuid.uuid4().hex[:8]}"
    tar_path = os.path.join(output_dir, f"{shard_id}.tar")

    urls = batch["url"]
    captions = batch["caption"]
    jpeg_list = batch["jpeg_bytes"]
    num_images = len(urls)

    with wds.TarWriter(tar_path) as sink:
        for i in range(num_images):
            sink.write({
                "__key__": f"{shard_id}_{i:06d}",
                "jpg": jpeg_list[i],
                "txt": str(captions[i]),
                "json": {"url": urls[i], "caption": captions[i]},
            })

    return {"shard_id": [shard_id], "success_count": [num_images], "total_count": [num_images]}


def parquet_to_webdataset_ray(
    hf_dataset_path: str,
    output_dir: str,
    entries_per_tar: int = 1000,
    max_entries: int | None = None,
    concurrency: int | None = None,
) -> dict[str, int]:
    """Convert HuggingFace parquet dataset to WebDataset tar files using Ray Data."""
    import ray
    import ray.data
    from functools import partial
    from huggingface_hub import HfFileSystem

    os.makedirs(output_dir, exist_ok=True)

    ds = ray.data.read_parquet(
        hf_dataset_path,
        file_extensions=["parquet"],
        columns=["url", "caption"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        concurrency=10,
    )

    if max_entries is not None:
        ds = ds.limit(max_entries)
        ds = ds.repartition(num_blocks=max(100, max_entries // 1000))

    if concurrency is None:
        cluster_resources = ray.cluster_resources()
        concurrency = max(4, int(cluster_resources.get("CPU", 4)))

    # Download images, validate, convert to JPEG
    ds = ds.map_batches(image_download_batch, batch_size=100, batch_format="numpy")
    ds = ds.flat_map(process_image)

    # Write tar shards
    results = ds.map_batches(
        partial(write_tar_batch, output_dir=output_dir),
        batch_size=entries_per_tar,
        batch_format="numpy",
        concurrency=concurrency,
    ).take_all()

    total_success = sum(r["success_count"] for r in results)
    num_shards = len(results)
    total_attempted = max_entries if max_entries is not None else total_success
    success_rate = (total_success / total_attempted * 100) if total_attempted > 0 else 0
    logger.info(f"Download complete: {total_success} images in {num_shards} shards ({success_rate:.1f}% success rate)")

    return {"total_success": total_success, "total_attempted": total_attempted, "num_shards": num_shards}

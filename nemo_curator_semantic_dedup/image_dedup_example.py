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

import os
import shutil
from functools import partial

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from helper import image_download_batch, process_image, write_tar_batch

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.deduplication.removal import ImageDuplicatesRemovalStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter


class Config(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    input_parquet: str
    input_wds_dir: str
    output_dir: str
    embeddings_dir: str
    removal_dir: str
    model_dir: str = "/home/ray/model_weights"
    entries_per_tar: int = 1000
    max_entries: int | None = None
    tar_files_per_partition: int = 1
    batch_size: int = 100
    embedding_batch_size: int = 32
    reader_cpus_per_task: int = 8


def _make_reader(config: Config) -> ImageReaderStage:
    """Create an ImageReaderStage with capped concurrency to prevent OOM.

    DALI reader tasks consume 10-55 GB each (full-resolution decode buffers).
    By default each task requests only 1 CPU, so Ray schedules ~48 per node,
    far exceeding the 184 GB node memory.  Requesting more CPUs per task
    limits concurrency (e.g. 16 CPUs → 3 readers per 48-CPU node).  This
    also prevents readers and embedding actors from running simultaneously,
    avoiding GPU-VRAM exhaustion from reader CUDA contexts.
    """
    reader = ImageReaderStage(batch_size=config.batch_size, num_gpus_per_worker=0)
    reader.resources.cpus = config.reader_cpus_per_task
    # Request a tiny GPU fraction to force placement on GPU nodes
    # (without this, readers land on small cpu-downloader nodes)
    reader.resources.gpus = 0.01
    return reader


def create_image_embedding_pipeline(config: Config) -> Pipeline:
    """Create pipeline: read images -> generate CLIP embeddings -> save to parquet."""
    pipeline = Pipeline(name="image_embedding")

    pipeline.add_stage(FilePartitioningStage(
        file_paths=config.input_wds_dir,
        files_per_partition=config.tar_files_per_partition,
        file_extensions=[".tar"],
    ))

    pipeline.add_stage(_make_reader(config))

    pipeline.add_stage(ImageEmbeddingStage(
        model_dir=config.model_dir,
        model_inference_batch_size=config.embedding_batch_size,
        remove_image_data=True,
    ))

    pipeline.add_stage(ConvertImageBatchToDocumentBatchStage(fields=["image_id", "embedding"]))

    pipeline.add_stage(ParquetWriter(path=config.embeddings_dir))

    return pipeline


def create_embedding_deduplication_workflow(config: Config) -> Pipeline:
    """Create semantic deduplication workflow using K-means + DBSCAN."""
    return SemanticDeduplicationWorkflow(
        input_path=config.embeddings_dir,
        output_path=config.removal_dir,
        # Must match column names written by ConvertImageBatchToDocumentBatchStage
        id_field="image_id",
        embedding_field="embedding",
        # Number of K-means clusters — controls the size of each pairwise
        # comparison group. More clusters = smaller groups = faster pairwise
        # but coarser semantic grouping. 100 is a reasonable default for ~10M images.
        n_clusters=100,
        # Cosine similarity threshold for marking duplicates. Pairs with
        # similarity >= (1 - eps) are considered duplicates. 0.01 means
        # images must be >= 99% similar to be flagged.
        eps=0.01,
    )


def create_image_deduplication_pipeline(config: Config) -> Pipeline:
    """Create pipeline: read images -> filter duplicates -> write deduplicated dataset."""
    pipeline = Pipeline(name="image_deduplication")

    pipeline.add_stage(FilePartitioningStage(
        file_paths=config.input_wds_dir,
        files_per_partition=config.tar_files_per_partition,
        file_extensions=[".tar"],
    ))

    pipeline.add_stage(_make_reader(config))

    pipeline.add_stage(ImageDuplicatesRemovalStage(
        removal_parquets_dir=config.removal_dir + "/duplicates",
        duplicate_id_field="id",
    ))

    pipeline.add_stage(ImageWriterStage(
        output_dir=config.output_dir,
        remove_image_data=True,
    ))

    return pipeline


def main(config: Config) -> None:
    """Main execution function for image semantic deduplication pipeline."""
    # Clean output directories from previous runs to avoid processing stale data
    for d in [config.input_wds_dir, config.embeddings_dir,
              config.removal_dir, config.output_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # Step 1: Download images and create WebDataset tar files
    import ray
    import ray.data
    from huggingface_hub import HfFileSystem

    os.makedirs(config.input_wds_dir, exist_ok=True)

    ds = ray.data.read_parquet(
        config.input_parquet,
        file_extensions=["parquet"],
        columns=["url", "caption"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        concurrency=10,
    )

    if config.max_entries is not None:
        ds = ds.limit(config.max_entries)
        ds = ds.repartition(num_blocks=max(100, config.max_entries // 1000))

    cluster_cpus = int(ray.cluster_resources().get("CPU", 4))
    concurrency = max(4, cluster_cpus)

    ds = ds.map_batches(image_download_batch, batch_size=100, batch_format="numpy")
    ds = ds.flat_map(process_image)

    # Write tar shards — these become input for the embedding pipeline below
    results = ds.map_batches(
        partial(write_tar_batch, output_dir=config.input_wds_dir),
        batch_size=config.entries_per_tar,
        batch_format="numpy",
        concurrency=concurrency,
    ).take_all()

    total_success = sum(r["success_count"] for r in results)
    num_shards = len(results)
    total_attempted = config.max_entries if config.max_entries is not None else total_success
    success_rate = (total_success / total_attempted * 100) if total_attempted > 0 else 0
    logger.info(f"Download complete: {total_success} images in {num_shards} shards ({success_rate:.1f}% success rate)")

    # Use executors that avoid scheduling on CPU-only head node
    streaming_executor = RayDataExecutor(ignore_head_node=True)
    actor_executor = RayActorPoolExecutor(ignore_head_node=True)

    # Step 2: Generate CLIP embeddings
    pipeline = create_image_embedding_pipeline(config)
    pipeline.run(executor=streaming_executor)

    # Step 3: Find semantic duplicates using K-means + DBSCAN
    workflow = create_embedding_deduplication_workflow(config)
    workflow.run(kmeans_executor=actor_executor, pairwise_executor=actor_executor)

    # Step 4: Write deduplicated dataset
    pipeline = create_image_deduplication_pipeline(config)
    pipeline.run(executor=streaming_executor)


if __name__ == "__main__":
    config = Config()
    main(config)
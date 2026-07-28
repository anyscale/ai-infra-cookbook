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

import argparse
import os
import time

import ray
from helper import download_webdataset

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.deduplication.removal import ImageDuplicatesRemovalStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter


def create_image_embedding_pipeline(args: argparse.Namespace) -> Pipeline:
    """Create image curation pipeline with file partitioning, image reading, embedding, deduplication."""

    # Define pipeline
    pipeline = Pipeline(name="image_curation", description="Curate images with embeddings and quality scoring")

    # Stage 0: Partition tar files for parallel processing
    pipeline.add_stage(FilePartitioningStage(
        file_paths=args.input_wds_dataset_dir,
        files_per_partition=args.tar_files_per_partition,
        file_extensions=[".tar"],
    ))

    # Stage 1: Read images from webdataset tar files (now runs in parallel)
    pipeline.add_stage(ImageReaderStage(
        task_batch_size=args.batch_size,
        verbose=args.verbose,
        num_threads=16,  # More threads for I/O
        num_gpus_per_worker=0.25,
    ))

    # Stage 2: Generate CLIP embeddings for images
    pipeline.add_stage(ImageEmbeddingStage(
        model_dir=args.model_dir,
        num_gpus_per_worker=args.embedding_gpus_per_worker,
        model_inference_batch_size=args.embedding_batch_size,
        verbose=args.verbose,
    ))

    # Stage 3: Convert embeddings to document batch
    pipeline.add_stage(ConvertImageBatchToDocumentBatchStage(fields=["image_id", "embedding"]))

    # Stage 4: Save embeddings to parquet file
    pipeline.add_stage(ParquetWriter(
        path=args.embeddings_dir,
    ))

    return pipeline

def create_embedding_deduplication_workflow(args: argparse.Namespace) -> Pipeline:
    """Create image deduplication pipeline with embedding deduplication."""
    return SemanticDeduplicationWorkflow(
        input_path=args.embeddings_dir,
        output_path=args.removal_parquets_dir,
        id_field="image_id",
        embedding_field="embedding",
        n_clusters=100,
        eps=0.01,
        read_kwargs={"storage_options": {}},
        write_kwargs={"storage_options": {}},
        verbose=args.verbose,
    )

def create_image_deduplication_pipeline(args: argparse.Namespace) -> Pipeline:
    """Create image deduplication pipeline with image deduplication."""
    # Define pipeline
    pipeline = Pipeline(name="image_deduplication", description="Deduplicate images with image deduplication")

    # Stage 0: Partition tar files for parallel processing
    pipeline.add_stage(FilePartitioningStage(
        file_paths=args.input_wds_dataset_dir,
        files_per_partition=args.tar_files_per_partition,
        file_extensions=[".tar"],
    ))

    # Stage 1: Read images from webdataset tar files (now runs in parallel)
    pipeline.add_stage(ImageReaderStage(
        task_batch_size=args.batch_size,
        verbose=args.verbose,
        num_threads=16,  # More threads for I/O
        num_gpus_per_worker=0.25,
    ))

    # Stage 2: Read removal list from parquet file and filter images
    pipeline.add_stage(ImageDuplicatesRemovalStage(
        removal_parquets_dir=args.removal_parquets_dir + "/duplicates",
        duplicate_id_field="id",
        verbose=args.verbose,
    ))

    # Stage 3: Write filtered images to disk
    pipeline.add_stage(ImageWriterStage(
        output_dir=args.output_dataset_dir,
        remove_image_data=True,
        verbose=args.verbose,
    ))

    return pipeline


def main(args: argparse.Namespace) -> None:
    """Main execution function for image curation pipeline."""

    ray_client = RayClient()
    ray_client.start()

    print("Starting image curation pipeline...")
    print(f"Input parquet file: {args.input_parquet}")
    print(f"Input webdataset directory: {args.input_wds_dataset_dir}")
    print(f"Output webdataset directory: {args.output_dataset_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Tar files per partition: {args.tar_files_per_partition}")
    print(f"Task batch size: {args.batch_size}")
    print("\n" + "=" * 50 + "\n")

    # Step 1: Download and prepare webdataset from parquet file
    if not args.skip_download:
        print("Step 1: Downloading webdataset from parquet file...")
        download_start = time.time()

        # Create output directory if it doesn't exist
        os.makedirs(args.input_wds_dataset_dir, exist_ok=True)

        # Download webdataset using helper function
        download_webdataset(
            parquet_path=args.input_parquet,
            output_dir=args.input_wds_dataset_dir,
            num_processes=args.download_processes,
            entries_per_tar=args.entries_per_tar,
        )

        download_time = time.time() - download_start
        print(f"✓ Dataset download completed in {download_time:.2f} seconds")
        print(f"✓ Webdataset saved to: {args.input_wds_dataset_dir}")
        print("\n" + "=" * 50 + "\n")
    else:
        print("Step 1: Skipping download (using existing dataset)")
        print(f"Using existing dataset at: {args.input_wds_dataset_dir}")
        print("\n" + "=" * 50 + "\n")

    # Step 2: Create and run curation pipelines
    # Step 2.1: Create image embedding pipeline
    print("Step 2.1: Running image embedding pipeline...")
    start_time = time.time()
    pipeline = create_image_embedding_pipeline(args)
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")
    pipeline.run()

    # Step 2.2: Create image deduplication pipeline (pairwise executor is XennaExecutor by default)
    print("Step 2.2: Running image deduplication pipeline...")
    start_time = time.time()
    pipeline = create_embedding_deduplication_workflow(args)
    print("\n" + "=" * 50 + "\n")
    pipeline.run()

    # Step 2.3: Create image deduplication pipeline
    print("Step 2.3: Running image deduplication pipeline...")
    start_time = time.time()
    pipeline = create_image_deduplication_pipeline(args)
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")
    pipeline.run()

    end_time = time.time()

    # Calculate and print execution time
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\nImage curation pipeline completed!")
    print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"\nProcessed dataset available at: {args.output_dataset_dir}")

    ray_client.stop()


def get_env_or_arg(env_var: str, arg_value, default=None):
    """Get value from environment variable or command-line argument."""
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value
    if arg_value is not None:
        return arg_value
    return default


def get_env_bool(env_var: str, arg_value: bool, default: bool = False) -> bool:
    """Get boolean value from environment variable or command-line argument."""
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value.lower() in ("true", "1", "yes")
    return arg_value if arg_value is not None else default


def get_env_int(env_var: str, arg_value: int, default: int) -> int:
    """Get integer value from environment variable or command-line argument."""
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return int(env_value)
    return arg_value if arg_value is not None else default


def get_env_float(env_var: str, arg_value: float, default: float) -> float:
    """Get float value from environment variable or command-line argument."""
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return float(env_value)
    return arg_value if arg_value is not None else default


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image curation pipeline with embedding generation and quality scoring. "
                    "Arguments can also be set via environment variables (see job.yaml)."
    )

    # Dataset arguments
    parser.add_argument(
        "--input-parquet",
        type=str,
        required=False,
        default=None,
        help="Path to input parquet file containing image URLs and metadata (env: INPUT_PARQUET)"
    )
    parser.add_argument(
        "--input-wds-dataset-dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save the downloaded webdataset (env: INPUT_WDS_DIR)"
    )
    parser.add_argument(
        "--output-dataset-dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save the resulting webdataset (env: OUTPUT_DIR)"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save the embeddings (env: EMBEDDINGS_DIR)"
    )
    parser.add_argument(
        "--removal-parquets-dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save the remove parquets (env: REMOVAL_DIR)"
    )
    parser.add_argument(
        "--download-processes",
        type=int,
        default=None,
        help="Number of parallel processes for downloading images (env: DOWNLOAD_PROCESSES)"
    )
    parser.add_argument(
        "--entries-per-tar",
        type=int,
        default=None,
        help="Number of entries per tar shard during download (env: ENTRIES_PER_TAR)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=None,
        help="Skip dataset download and use existing webdataset (env: SKIP_DOWNLOAD)"
    )

    # Image reader arguments
    parser.add_argument(
        "--tar-files-per-partition",
        type=int,
        default=None,
        help="Number of tar files to process per partition (env: TAR_FILES_PER_PARTITION)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of images per ImageBatch for the reader stage (env: BATCH_SIZE)"
    )

    # General arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        required=False,
        default=None,
        help="Path to model directory containing all model weights (env: MODEL_DIR)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging for all stages"
    )

    # Embedding stage arguments
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Batch size for embedding generation (env: EMBEDDING_BATCH_SIZE)"
    )
    parser.add_argument(
        "--embedding-gpus-per-worker",
        type=float,
        default=None,
        help="GPU allocation per worker for embedding generation"
    )

    cli_args = parser.parse_args()
    
    # Resolve arguments from environment variables or command-line args
    args = argparse.Namespace(
        input_parquet=get_env_or_arg("INPUT_PARQUET", cli_args.input_parquet),
        input_wds_dataset_dir=get_env_or_arg("INPUT_WDS_DIR", cli_args.input_wds_dataset_dir),
        output_dataset_dir=get_env_or_arg("OUTPUT_DIR", cli_args.output_dataset_dir),
        embeddings_dir=get_env_or_arg("EMBEDDINGS_DIR", cli_args.embeddings_dir),
        removal_parquets_dir=get_env_or_arg("REMOVAL_DIR", cli_args.removal_parquets_dir),
        model_dir=get_env_or_arg("MODEL_DIR", cli_args.model_dir, "/home/ray/model_weights"),
        download_processes=get_env_int("DOWNLOAD_PROCESSES", cli_args.download_processes, 8),
        entries_per_tar=get_env_int("ENTRIES_PER_TAR", cli_args.entries_per_tar, 1000),
        skip_download=get_env_bool("SKIP_DOWNLOAD", cli_args.skip_download, False),
        tar_files_per_partition=get_env_int("TAR_FILES_PER_PARTITION", cli_args.tar_files_per_partition, 1),
        batch_size=get_env_int("BATCH_SIZE", cli_args.batch_size, 100),
        embedding_batch_size=get_env_int("EMBEDDING_BATCH_SIZE", cli_args.embedding_batch_size, 32),
        embedding_gpus_per_worker=get_env_float("EMBEDDING_GPUS_PER_WORKER", cli_args.embedding_gpus_per_worker, 0.25),
        verbose=cli_args.verbose,
    )
    
    # Validate required arguments
    required_args = ["input_wds_dataset_dir", "output_dataset_dir", "embeddings_dir", "removal_parquets_dir"]
    missing = [arg for arg in required_args if getattr(args, arg) is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}. "
                     "Set them via command-line or environment variables.")
    
    main(args)
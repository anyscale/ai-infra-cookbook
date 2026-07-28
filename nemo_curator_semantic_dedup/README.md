# Image Semantic Deduplication with NeMo Curator

This example uses [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) to perform GPU-accelerated semantic deduplication on image datasets.

NeMo Curator is a scalable data curation library that leverages NVIDIA RAPIDS™ for GPU acceleration. This example downloads images from a parquet file, generates CLIP embeddings, and removes near-duplicate images based on semantic similarity.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Run the job

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/nemo_curator_semantic_dedup
```

Submit the job.

```bash
anyscale job submit -f job.yaml
```

## Understanding the example

- The [Dockerfile](./Dockerfile) builds a custom image with NeMo Curator CUDA dependencies (`nemo-curator[image_cuda12]`), downloads the MS COCO sample dataset from HuggingFace, and pre-downloads the CLIP model weights to speed up job startup.

- The entrypoint defined in [job.yaml](./job.yaml) runs `image_dedup_example.py` which executes a 3-step pipeline:
  1. **Download WebDataset**: Fetches images from URLs in the parquet file and saves them as WebDataset tar files to `/mnt/cluster_storage/nemo_curator/webdataset`
  2. **Generate CLIP embeddings**: Uses OpenAI's CLIP ViT-L/14 model to create 768-dimensional embeddings for each image
  3. **Semantic deduplication**: Clusters embeddings with k-means and removes near-duplicates based on cosine similarity

- The `/mnt/cluster_storage/` directory is an ephemeral shared filesystem attached to the cluster for the duration of the job. All outputs (embeddings, duplicate IDs, and deduplicated images) are saved here.

- To use your own data, prepare a parquet file with `URL` and `TEXT` columns, upload it to cluster storage, and override the `INPUT_PARQUET` environment variable:
  ```bash
  anyscale job submit -f job.yaml \
    --env INPUT_PARQUET=/mnt/cluster_storage/your_data.parquet \
    --env OUTPUT_DIR=/mnt/cluster_storage/your_results
  ```

- The [helper.py](./helper.py) module provides utilities for downloading images in parallel and converting them to [WebDataset](https://github.com/webdataset/webdataset) format, which is optimized for streaming large-scale image datasets.

## View the job

View the job in the [jobs tab](https://console.anyscale.com/jobs) of the Anyscale console.

## Learn more

- [NeMo Curator Documentation](https://docs.nvidia.com/nemo/curator/latest/)
- [NeMo Curator Image Tutorials](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/image/getting-started)
- [Anyscale Jobs Documentation](https://docs.anyscale.com/platform/jobs/)

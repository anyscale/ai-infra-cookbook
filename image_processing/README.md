# Large-Scale Image Processing with Vision Language Models

This example demonstrates how to build an image processing pipeline that scales to billions of images using Ray Data and vLLM on Anyscale. We process the [ReLAION-2B dataset](https://huggingface.co/datasets/laion/relaion2B-en-research-safe), which contains over 2 billion image URLs with associated metadata.

You'll need a HuggingFace token to access the ReLAION-2B dataset. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).


## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```


## Submit the job.

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/image_processing
```

Submit the job. Use `--env` to forward your Hugging Face token to authenticate with Hugging Face.

```bash
anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN
```

Results will be written to `/mnt/shared_storage/process_images_output/{timestamp}/` in the Parquet format.


## Understanding the example

- The pipeline performs three main stages on each image:
    - **Image Download**: Download images from URLs in a multi-threaded manner handling timeouts and invalid URLs.
    - **Image Preprocessing**: Validate, resize, and standardize images, filtering out corrupted or invalid images.
    - **Vision Model Inference**: Run the [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) vision-language model using vLLM to generate a caption for each image.
- The entire pipeline is orchestrated by [Ray Data](https://docs.ray.io/en/latest/data/data.html), which handles distributed execution, fault tolerance, and resource management across your cluster.
- This example uses [Ray Data's native vLLM integration](https://docs.ray.io/en/latest/data/working-with-llms.html) to optimize vLLM for throughput and perform batch inference in the overall pipeline.
- Some notes on configuration.
    - This example passes `concurrency=10` into `ray.data.read_parquet` in order to reduce the likelihood of hitting Hugging Face rate limits.
    - This example calls `repartition(target_num_rows_per_block=1000)` after the `read_parquet` call. The blocks created by `read_parquet` can consist of millions of rows because each row consists of small URL data. Ray Data processes a single block sequentially (one batch at a time). The `repartition` call creates smaller blocks from the larger block which is important both to increase the degree of parallelism and to reduce the memory required to process each block.

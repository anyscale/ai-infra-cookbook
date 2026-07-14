# Video captioning at scale: raw Ray Core vs Ray Data

This example captions a video corpus with a vision-language model twice — once
hand-built on [Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
and once on [Ray Data](https://docs.ray.io/en/latest/data/data.html) — so you can
compare throughput, GPU/CPU utilization, and code complexity on identical
hardware. Both read [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo),
decode and sample keyframes on CPU, and caption each clip with
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) served by
[vLLM](https://github.com/vllm-project/vllm), one replica per GPU.

## The stack

| Layer | Library | Role in this example |
|---|---|---|
| Orchestration (baseline) | [Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html) | hand-rolled tasks + actors + a manual backpressure loop |
| Orchestration (framework) | [Ray Data](https://docs.ray.io/en/latest/data/data.html) | the same DAG as a streaming pipeline with automatic backpressure |
| Inference | [vLLM](https://github.com/vllm-project/vllm) | high-throughput batched VLM inference, one engine per GPU |
| Model | [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | captions an ordered sequence of keyframes per clip |
| Dataset | [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) | source video corpus, streamed from Hugging Face or object storage |
| Platform | [Anyscale](https://www.anyscale.com) | image build, compute provisioning, job/service management |

## Pipeline

The example is three files. Both implementations run exactly the same two
compute-heavy stages — defined once in
[`utils.py`](https://github.com/anyscale/ai-infra-cookbook/blob/main/video_captioning/utils.py)
— so only the orchestration differs.

```
FineVideo parquet (mp4 bytes)
    |
    +-- decode + uniform keyframe sample   # CPU: decord open, N frames per clip
    |
    +-- Qwen3-VL caption                    # GPU: vLLM, one replica per GPU
    |
    +-- write parquet + report.json         # captions + throughput/utilization
```

The **baseline** ([`caption_ray_core.py`](https://github.com/anyscale/ai-infra-cookbook/blob/main/video_captioning/caption_ray_core.py))
builds this out of Ray Core primitives: shard-reader/decode tasks, a pool of
GPU caption actors, and a hand-written scheduling loop that caps in-flight work
at each stage, batches decoded rows, and round-robins them across the actors.
The **rebuild** ([`caption_ray_data.py`](https://github.com/anyscale/ai-infra-cookbook/blob/main/video_captioning/caption_ray_data.py))
expresses the same DAG in a handful of Ray Data operators and gets streaming,
backpressure, fault tolerance, and autoscaling for free.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Submit the job

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/ai-infra-cookbook.git
cd ai-infra-cookbook/video_captioning
```

[FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) is a gated
Hugging Face dataset, so you **must** pass an `HF_TOKEN`. Run the Ray Data
pipeline on a small slice first:

```bash
export HF_TOKEN=hf_...

anyscale job submit -f job_ray_data.yaml --env HF_TOKEN=$HF_TOKEN --env NUM_VIDEOS=500
```

Then run the raw Ray Core baseline on the same slice to compare:

```bash
anyscale job submit -f job_ray_core.yaml --env HF_TOKEN=$HF_TOKEN --env NUM_VIDEOS=500
```

Omit `NUM_VIDEOS` to process the whole dataset. Pass `--env INPUT=<uri>` to
read from object storage instead of Hugging Face and `--env OUTPUT=<uri>` to
choose where captions land (default: a timestamped prefix under
`$ANYSCALE_ARTIFACT_STORAGE/video_captioning/`).

## Understanding the example

- [`utils.py`](https://github.com/anyscale/ai-infra-cookbook/blob/main/video_captioning/utils.py)
  holds the *work*, shared by both pipelines: `decode_and_sample` does the CPU
  stage (decord open, uniform keyframe sample, JPEG encode) and the message
  builders construct the Qwen3-VL prompt. Because the work is identical, any
  throughput difference comes from orchestration.
- [`caption_ray_core.py`](https://github.com/anyscale/ai-infra-cookbook/blob/main/video_captioning/caption_ray_core.py)
  is intentionally explicit: it shows every piece of scheduling a framework
  normally hides — in-flight caps per stage, manual batching, round-robin
  dispatch, and partial-batch draining — and it hand-rolls no fault tolerance,
  so a dead actor fails the run.
- [`caption_ray_data.py`](https://github.com/anyscale/ai-infra-cookbook/blob/main/video_captioning/caption_ray_data.py)
  uses the native [`ray.data.llm`](https://docs.ray.io/en/latest/data/api/llm.html)
  vLLM integration. The read stage is pinned to `cpu_only`-labeled nodes so
  multi-MB mp4 blobs never share RAM with the GPU engines.
- Qwen3-VL-8B runs one replica per GPU with no tensor parallelism — it fits
  comfortably on a 96 GB RTX PRO 6000, which maximizes throughput.

## Measuring throughput and utilization

Every run writes `report.json` next to its captions with:

- **Throughput** — captions/sec, captions/GPU/sec, and video-hours processed
  per wall-clock hour.
- **GPU utilization** (NVML) and **CPU utilization** (psutil), mean and p95, so
  you can see whether decode is keeping the GPUs fed or starving them.

## Scaling the run

The committed job YAMLs default to an 8-GPU smoke test (one GPU worker node).
To scale, edit the GPU worker pool's `min_nodes`/`max_nodes` in the YAML and
pass a matching `--env EXPECTED_GPUS=<n>` so the driver waits for the full
fleet before starting the clock: 8 nodes for 64 GPUs, 32 nodes for 256 GPUs.

## Position in the stack

**Stage:** Curate

- **Related:** [Streaming video curation with Ray Data](../video_curation/) — same video modality; that example curates, this one benchmarks captioning throughput
- **Downstream:** [SFT with Megatron-Bridge and Ray Train](../megatron_training/) — captioned clip datasets feed multimodal training
- **Journeys:** [Pretraining data factory](../README.md#journeys)

Part of the [Open-Source Frontier Infra Stack](../README.md) — explore the
map in the [interactive explorer](../README.md#interactive-explorer).

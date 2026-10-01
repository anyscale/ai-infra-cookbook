# Cosmos Curate

This repository is an example of running NVIDIA [cosmos-curate](https://github.com/nvidia-cosmos/cosmos-curate) pipelines on Anyscale. Examples include the Hello World and Reference Video Pipelines.

## Prerequisites

- An [Anyscale account](https://console.anyscale.com/) with the `anyscale` CLI installed (`pip install anyscale`)
- AWS account with ECR access (for pushing the Docker image) and S3 permissions on the nodes
- A local clone (or symlink, e.g. `ln -sf /path/to/cosmos-curate ./cosmos-curate`) of the [`cosmos-curate`](https://github.com/NVIDIA/cosmos-curate) repo (see [Runtime Environment](#3-runtime-environment) for details)

Your directory layout should look like:
```
cosmos_curate/          # this directory
├── cosmos-curate/      # clone of the cosmos-curate repo
├── docker/
├── hello_world.yaml
├── reference_pipeline.yaml
├── all_nodes_init_script.py
├── cosmos_curate_tokens.yaml
└── ...
```

## Setup

### 1. Docker image

Update the ECR information in `./push_anyscale.sh` with your own repo then:
```
TAG=1
./build_anyscale.sh $TAG && ./push_anyscale.sh $TAG
```

The `anyscale-cosmos-curate.Dockerfile` adds [Anyscale requirements](https://docs.anyscale.com/container-image/image-requirement) prior to building the `pixi` layers as `chown`'ing these layers later almost doubles the image size. This image used `./generate_dockerfile.sh` from `cosmos-curate` repo to generate the `cosmos-curate.Dockerfile` without `cuml` env then added the Anyscale portion to that generated Dockerfile.

Can update the jobs `image_uri:` with your image once it is built and pushed.

### 2. cosmos_curate.yaml (API auth)

`cosmos-curate` expects `/cosmos_curate/config/cosmos_curate.yaml` to control the authentication to APIs and model registrys. `huggingface` is all that is required to run the two examples in this repo. Can add your credentials locally and when the job runs there `entrypoint:` will distributed to all nodes at that path with `all_nodes_init_script.py`.

```
μ cat cosmos_curate_tokens.yaml
huggingface:
    user: ""
    api_key: ""
```

### 3. s3_creds_file.yaml (S3 auth)

`cosmos-curate` expects an S3 credential file at `/dev/shm/s3_creds_file`. This is configurable by `COSMOS_S3_PROFILE_PATH`. For this examplet the jobs run on AWS where the IAM has S3 permissions so we use the `aws` cli to write out temporary crednetials for the job to this path. 

If you need to authenticate in a different way need to ensure this file is written and distributed to all nodes at the expected filepath.

## Run

The Hello World Pipeline runs in a few minutes and only requires 1 T4 GPU node. 

```
anyscale job submit -f hello_world.yaml
```

The Reference Video Pipeline will take ~45m with the default setup of 4 L40S GPUs on ~3h of video. 
```
anyscale job submit -f reference_pipeline.yaml
```

## How It Works

### Cosmos Curate on Anyscale

Let's breakdown the the `reference_video_pipeline.yaml` to get a sense for how the setup comes together, starting from defining the hardware we want to use up to the user code defining the pipeline.

### 1. Compute Config

This defines the nodes we will require to run the pipeline. Typically the head nodes in Ray clusters should be set to have zero resources, but the `cosmos-curate` library expects it. 
```
compute_config:
  head_node:
    instance_type: m5.2xlarge
    resources:
      CPU: 8
      GPU: 0
    flags: {}
  worker_nodes:
    - instance_type: g6e.4xlarge
      flags: {}
      min_nodes: 4
      max_nodes: 4
      market_type: ON_DEMAND
```

The reference video pipeline defaults to 4 1xL40S instances. The logs at end of pipeline will report on runtimes. Here is 4 GPUs compared to 16 GPUs for ~1k videos, about 3h of video:

4 GPUs takes 44m
```
2026-02-28 19:12:46.030 | INFO     | cosmos_curate.pipelines.video.splitting_pipeline:split:703 - Split-Transcode-Filter-Annotate pipeline: input_build_time=0.01 / pipeline_run_time=44.26 / summary_run_time=0.02 mins processing time for total_video_length=3.191 hours of raw videos
```

16 GPUs took 13m
```
2026-03-01 05:56:58.599 | INFO     | cosmos_curate.pipelines.video.splitting_pipeline:split:703 - Split-Transcode-Filter-Annotate pipeline: input_build_time=0.01 / pipeline_run_time=12.71 / summary_run_time=0.01 mins processing time for total_video_length=3.191 hours of raw videos
```

### 2. Image

This block defines name of job, the image all the nodes will start and clarifies for a custom image built the expected Ray version we will be running.
```
image_uri: 367974485317.dkr.ecr.us-west-2.amazonaws.com/anyscale-cosmos-curate:6
ray_version: 2.48.0
```

When the job runs it will acquire all the nodes and use our image which handles a few things for us:
* All `pixi` environments are built into the container
* While on Anyscale we typically just use `working_dir` or `py_modules` to ship code for use at runtime, `cosmos-curate` expects code at `/opt/cosmos-curate/cosmos_curate` so there is a copy of the code in there as well for referencing the `all_models.json` file and some other configurations.
* We set `PIXI_PROJECT_MANIFEST` in the image so that runtime `pixi run` calls (whether by the `entrypoint:` or in the pipeline model classes `py_executable` to enable switching between different envs for specific models) all know where these environments are built and cached. The `default` `pixi` environment is the default `python` on `PATH` if you call `python` directly outside of `pixi run`.

### 3. Runtime Environment

Anyscale will ship your `working_dir` which should be the `examples/cosmos-curate/` directory. This allows us to access files for setting up the nodes, addition python scripts to run, python packages, etc. This allows us to generally update code running on the image without requiring rebuild.

`py_modules` packages a local clone of the `cosmos-curate` repo (the `./cosmos-curate` directory listed in [Prerequisites](#prerequisites)) and ships it to all nodes at runtime. This lets you iterate on `cosmos-curate` source code without rebuilding the Docker image, overriding the copy baked into the image at `/opt/cosmos-curate/cosmos_curate/`.

```
py_modules: ["./cosmos-curate"]
working_dir: "."
```

### 4. Entrypoint

The `entrypoint:` will be executed on the head node only. Typically this might be as simple as `entrypoint: python main.py`, but for `cosmos-curate` we want to coordinate some startup logic so use Ray to distribute initialization logic before executing the main entrypoint from the `cosmos_curate` library.

```
entrypoint: >
  python all_nodes_init_script.py qwen2.5_vl,transnetv2,internvideo2_mm,bert
  && pixi run python -m cosmos_curate.pipelines.video.run_pipeline split
  --input-video-path "s3://ray-example-data/videos/Hollywood2-actions-videos/Hollywood2/AVIClips/"
  --output-clip-path "/mnt/user_storage/output_clips/"
```

#### python all_nodes_init_script.py

The `all_nodes_init_script.py` handles a few initialization steps for the cluster:

1. Use `write_s3_creds_file.sh` to put an S3 credential file where it is expected on each node.
2. Copy our local `cosmos_curate_tokens.yaml` to the expected location on each not for API and model registry auth.
3. Use the `model-download` `pixi` env to run `python -m cosmos_curate.core.managers.model_cli download` and pass the list of models needed for the pipeline we are going to run (if you do not specify the models it will download all models which takes a while and 500GB+ of space).

#### python -m cosmos_curate.pipelines.video.run_pipeline split

Now the actual pipelne uses the default `pixi` env to run `python -m cosmos_curate.pipelines.video.run_pipeline split`. There are many cli options you can pass to the pipelines `cosmos-curate` provides, but here we just set the minimal input and output paths and accept the rest as default. 

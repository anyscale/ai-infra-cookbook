# Train a model with Jax on GPUs

This example uses Jax to train a small model on 16 T4 GPUs.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Submit the job

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/jax_training
```

Submit the job with

```bash
anyscale job submit -f job.yaml
```

## Understanding the example

- This example installs a nightly version of Ray in the [Dockerfile](https://github.com/anyscale/examples/blob/main/jax_training/Dockerfile) because Ray Train GPU support for Jax is very recent.
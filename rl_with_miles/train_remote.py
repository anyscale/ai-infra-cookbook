"""Ray remote wrapper for training - ensures it runs on GPU workers."""
import sys
import subprocess
import os
import ray


@ray.remote(label_selector={"ray.io/accelerator-type": "H100"})
def run_training(cmd_args):
    """Run training on GPU workers.

    Uses label selector to ensure placement on H100 GPU nodes.
    The label must match the accelerator-type in job.yaml compute_config.

    Explicitly sets CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 so the subprocess
    can access all GPUs on the worker node. Does not reserve GPUs (num_gpus)
    to avoid conflicts with the MILES training script's internal GPU allocation.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    result = subprocess.run(
        ["python", "/tmp/miles/train_async.py"] + cmd_args,
        capture_output=False,  # Stream output directly
        text=True,
        env=env
    )
    return result.returncode


sys.exit(ray.get(run_training.remote(sys.argv[1:])))

"""Ray remote wrapper for weight conversion - ensures it runs on a GPU worker."""
import sys
import subprocess
import os
import ray


@ray.remote(label_selector={"ray.io/accelerator-type": "H100"})
def convert_weights(cmd_args):
    """Run weight conversion on a GPU worker.

    Uses label selector to ensure placement on H100 GPU nodes.
    The label must match the accelerator-type in job.yaml compute_config.

    Explicitly sets CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 so the subprocess
    can access all GPUs on the worker node. Does not reserve GPUs (num_gpus)
    to allow flexible GPU allocation.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    result = subprocess.run(
        ["python", "/tmp/miles/tools/convert_hf_to_torch_dist.py"] + cmd_args,
        capture_output=True,
        text=True,
        env=env
    )
    return result.returncode, result.stdout, result.stderr


returncode, stdout, stderr = ray.get(convert_weights.remote(sys.argv[1:]))
if stdout:
    print(stdout, end="")
if stderr:
    print(stderr, end="", file=sys.stderr)
sys.exit(returncode)

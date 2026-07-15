import sys
import ray
import subprocess
from time import perf_counter as pc

SCRIPT = """
set -e
bash write_s3_creds_file.sh
cp cosmos_curate_tokens.yaml /cosmos_curate/config/cosmos_curate.yaml
pixi run -e model-download python -m cosmos_curate.core.managers.model_cli download --models {models}
"""

@ray.remote(num_cpus=0)
def run_init(script):
    try:
        subprocess.check_call(script, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Init script failed (exit code {e.returncode})") from None

if __name__ == "__main__":
    models = sys.argv[1]
    script = SCRIPT.format(models=models)
    t = pc()
    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n["Alive"]]
    tasks = [
        run_init.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=n["NodeID"], soft=False
            )
        ).remote(script)
        for n in nodes
    ]
    print(f"Downloading models on {len(tasks)} nodes...")
    ray.get(tasks)
    dur = pc() - t
    print(f"Done. ({dur:0.1f}s)")

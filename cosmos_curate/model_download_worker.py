import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import cosmos_curate

@ray.remote(runtime_env={"py_executable": "pixi run -e model-download python", "excludes": ["./pixi/"], "env_vars": {"PIXI_PROJECT_MANIFEST": "/opt/cosmos-curate/pixi.toml"}})
def download_model():
    # Only works from /opt/cosmos-curate. Not importable otherwise from package.
    import cosmos_curate.core.managers.model_cli as cli
    cli.main(["download", "--models", "qwen2.5_vl,transnetv2,internvideo2_mm,bert"])

if __name__ == "__main__":
    ray.init(runtime_env={"env_vars": {"PIXI_PROJECT_MANIFEST": "/opt/cosmos-curate/pixi.toml"}, "py_modules": [cosmos_curate]})
    refs = []
    for n in ray.nodes():
        if not n["Alive"]:
            continue
        ref = (
            download_model
            .options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False))
            .remote()
         )
        refs.append(ref)
    ray.get(refs)


"""
RDT Weight Sync Correctness Test

Verifies that model weights can be correctly transferred from a "trainer"
Ray actor to an SGLang SchedulerActor using Ray Direct Transport (RDT)
with NCCL.

Key idea: SchedulerActor.__init__ loads the model.  By *not* calling
run_event_loop() the actor is free to receive regular Ray method calls,
which makes the test trivially sequential.

Requires 2 GPUs on the same node (one for the producer, one for the
scheduler).

Usage:
    python test_rdt_weight_sync.py --model-path Qwen/Qwen3-0.6B
"""

import argparse
import json
import sys
import time

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


# ---------------------------------------------------------------------------
# Weight-producer actor (simulates a trainer holding updated weights)
# ---------------------------------------------------------------------------
@ray.remote(num_gpus=1)
class WeightProducerActor:
    """Loads a HuggingFace model and exposes its weights via RDT."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).cuda()
        self._build_bucket()

    # -- internal ---------------------------------------------------------
    def _build_bucket(self):
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        named_tensors = [
            (name, param.data) for name, param in self.model.named_parameters()
        ]
        self._bucket = FlattenedTensorBucket(named_tensors=named_tensors)

    # -- public API -------------------------------------------------------
    @ray.method(tensor_transport="nccl")
    def get_weights(self) -> torch.Tensor:
        """Return the flattened weight tensor (transferred via NCCL/RDT)."""
        return self._bucket.get_flattened_tensor()

    def get_metadata_json(self) -> str:
        """Return bucket metadata as a JSON string (object-store path)."""
        metadata = self._bucket.get_metadata()
        return json.dumps(
            [
                {
                    "name": m.name,
                    "shape": list(m.shape),
                    "dtype": str(m.dtype).replace("torch.", ""),
                    "start_idx": m.start_idx,
                    "end_idx": m.end_idx,
                    "numel": m.numel,
                }
                for m in metadata
            ]
        )

    def get_param(self, name: str) -> torch.Tensor:
        """Return a single parameter tensor (on CPU) for verification."""
        return self.model.state_dict()[name].cpu()

    def get_param_names(self) -> list:
        return list(self.model.state_dict().keys())


# ---------------------------------------------------------------------------
# Main test driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RDT Weight Sync Correctness Test")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("RDT Weight Sync Correctness Test")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")

    # ------------------------------------------------------------------
    # 1. Placement group: 2 GPUs on the same node
    # ------------------------------------------------------------------
    pg = placement_group(
        bundles=[{"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}],
        strategy="STRICT_PACK",
    )
    ray.get(pg.ready())
    print("Placement group ready (2 GPUs).\n")

    # ------------------------------------------------------------------
    # 2. Create the weight producer (bundle 0)
    # ------------------------------------------------------------------
    print("Creating WeightProducerActor …")
    producer = WeightProducerActor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        ),
    ).remote(args.model_path)

    # ------------------------------------------------------------------
    # 3. Create the SchedulerActor (bundle 1) — don't start event loop
    # ------------------------------------------------------------------
    print("Creating SchedulerActor …")
    from sglang.srt.managers.scheduler_actor import SchedulerActor
    from sglang.srt.server_args import PortArgs, ServerArgs

    server_args = ServerArgs(
        model_path=args.model_path,
        tp_size=1,
        port=args.port,
    )
    port_args = PortArgs.init_new(server_args)

    scheduler = SchedulerActor.options(
        num_gpus=1,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=1,
        ),
    ).remote(
        server_args=server_args,
        port_args=port_args,
        gpu_id=0,
        tp_rank=0,
        moe_ep_rank=0,
        pp_rank=0,
        dp_rank=0,
    )

    # Wait for both actors to be ready
    print("Waiting for actors to initialise …")
    ray.get(scheduler.get_info.remote())
    param_names = ray.get(producer.get_param_names.remote())
    print(f"Both actors ready.  Model has {len(param_names)} state-dict entries.\n")

    # ------------------------------------------------------------------
    # 4. Transfer weights: metadata via object store, tensor via RDT
    # ------------------------------------------------------------------
    print("Step 1: Fetch metadata (object store) …")
    t0 = time.time()
    metadata_json = ray.get(producer.get_metadata_json.remote())
    print(f"  Done ({time.time() - t0:.2f}s)")

    print("Step 2: Transfer weights (RDT / NCCL) …")
    t0 = time.time()
    weights_ref = producer.get_weights.remote()  # RDT-enabled ObjectRef
    ok = ray.get(scheduler.receive_weights_rdt.remote(weights_ref, metadata_json))
    elapsed = time.time() - t0
    print(f"  Done — success={ok} ({elapsed:.2f}s)")

    # ------------------------------------------------------------------
    # 5. Verify: compare every parameter between producer and scheduler
    # ------------------------------------------------------------------
    print(f"\nStep 3: Verifying {len(param_names)} parameters …")
    mismatches = []
    for i, name in enumerate(param_names):
        p_tensor = ray.get(producer.get_param.remote(name))
        s_tensor = ray.get(scheduler.get_param.remote(name))
        if not torch.allclose(p_tensor, s_tensor, atol=1e-6):
            mismatches.append(name)
            print(f"  MISMATCH [{i}] {name}")
        elif i % 20 == 0:
            print(f"  [{i}/{len(param_names)}] {name} ✓")

    print(f"\nChecked {len(param_names)} parameters, {len(mismatches)} mismatches.")
    if mismatches:
        print("FAILURE — mismatched parameters:")
        for n in mismatches:
            print(f"  - {n}")
        return 1

    print("SUCCESS: All parameters match!")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    ray.util.remove_placement_group(pg)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

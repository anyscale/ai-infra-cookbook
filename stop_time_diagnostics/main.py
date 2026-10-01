"""
Standalone entry point for running stop-time diagnostics as a Ray job.

This script runs GPU health checks and communication tests on specified GPUs
to diagnose failures in distributed training systems.
"""

import argparse
from dataclasses import dataclass

import ray
from ray.util.placement_group import placement_group

from diagnostics import DiagnosticRunner, DiagnosticConfig


@dataclass
class Config:
    """Configuration for the diagnostics job."""
    num_gpus: int = 0  # Number of GPUs (from max_resources or specified)
    timeout: float = 120.0  # Timeout for each test phase


def get_max_gpus_from_cluster() -> int:
    """
    Get the maximum number of GPUs that can be provisioned in the cluster.

    This looks at max_resources which defines the cluster's scaling limits.
    For auto-scaling clusters, this tells us how many GPUs we can request.

    Returns:
        Maximum number of GPUs available, or 0 if none configured
    """
    # Check cluster resources first (already provisioned)
    resources = ray.cluster_resources()
    current_gpus = int(resources.get("GPU", 0))

    if current_gpus > 0:
        print(f"Found {current_gpus} GPUs already provisioned")
        return current_gpus

    # For auto-scaling clusters, we need to look at available node types
    # The cluster will scale up when we create a placement group
    # Default to a reasonable number that matches job.yaml max_resources
    print("No GPUs currently provisioned - cluster will auto-scale when placement group is created")
    return 0


def main():
    config = Config()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run stop-time GPU diagnostics")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to request (required for auto-scaling clusters)")
    parser.add_argument("--timeout", type=float, default=config.timeout,
                        help="Timeout in seconds for each test phase")
    args = parser.parse_args()

    config.timeout = args.timeout

    # Initialize Ray (connects to existing cluster if running as a job)
    ray.init()

    # Determine number of GPUs
    if args.num_gpus is not None:
        config.num_gpus = args.num_gpus
    else:
        # Try to get from already-provisioned resources
        config.num_gpus = get_max_gpus_from_cluster()

    if config.num_gpus == 0:
        raise RuntimeError(
            "No GPUs found and --num-gpus not specified. "
            "For auto-scaling clusters, you must specify --num-gpus to trigger scaling."
        )

    print(f"Starting stop-time diagnostics on {config.num_gpus} GPUs...")

    # Create placement group for the diagnostic actors
    # This triggers auto-scaling in clusters with min_nodes=0
    print("Creating placement group (this will trigger auto-scaling if needed)...")
    pg = placement_group(
        [{"GPU": 1, "CPU": 1}] * config.num_gpus,
        strategy="PACK",
    )
    ray.get(pg.ready(), timeout=600)  # 10 min timeout for auto-scaling
    print("Placement group ready")

    # Run diagnostics
    runner = DiagnosticRunner(DiagnosticConfig(timeout=config.timeout))
    bundle_indices = list(range(config.num_gpus))
    result = runner.run_on_placement_group(pg, bundle_indices)

    # Print results
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Summary: {result.summary()}")
    print(f"Stop Reason: {result.stop_reason.value}")
    print(f"Is Graceful Stop: {result.is_graceful_stop}")

    if result.faulty_gpus:
        print(f"Faulty GPUs: {result.faulty_gpus}")
    if result.faulty_nodes:
        print(f"Faulty Nodes: {result.faulty_nodes}")

    print("\nDetailed Test Results:")
    print("-" * 60)
    for test_result in result.test_results:
        phase = test_result.get("phase", "unknown")
        rank = test_result.get("rank", "?")
        passed = test_result.get("passed", False)
        status = "PASS" if passed else "FAIL"

        print(f"[{status}] Phase: {phase}, Rank: {rank}")

        if test_result.get("node_ip"):
            print(f"       Node IP: {test_result['node_ip']}")
        if test_result.get("ray_gpu_ids"):
            print(f"       GPU IDs: {test_result['ray_gpu_ids']}")

        if test_result.get("errors"):
            print("       Errors:")
            for error in test_result["errors"]:
                for line in error.split("\n")[:5]:  # Limit traceback lines
                    print(f"         {line}")

        if test_result.get("metrics"):
            metrics = test_result["metrics"]
            if metrics.get("cuda_device_name"):
                print(f"       Device: {metrics['cuda_device_name']}")
            if metrics.get("cuda_compute_test"):
                print(f"       CUDA Compute: {metrics['cuda_compute_test']}")

    print("=" * 60)

    # Exit with appropriate code
    if result.is_graceful_stop:
        print("\nAll diagnostics passed - system is healthy")
        return 0
    else:
        print(f"\nDiagnostics detected issues - stop reason: {result.stop_reason.value}")
        return 1


if __name__ == "__main__":
    exit(main())

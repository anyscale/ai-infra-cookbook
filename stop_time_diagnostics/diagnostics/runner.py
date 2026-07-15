"""
Stop-time diagnostic runner for hierarchical failure diagnosis.

This module provides:
- Failure classification (StopReason)
- DiagnosticRunner to orchestrate independent DiagnosticActors
- Hierarchical test execution (GPU health, intra-node comm, inter-node comm)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import logging
import ray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor import DiagnosticActor


class StopReason(Enum):
    """Classification of stop/failure reasons."""

    UNKNOWN = "unknown"
    GRACEFUL_STOP = "graceful_stop"  # User-initiated or planned stop
    GPU_ERROR = "gpu_error"  # GPU hardware error
    NCCL_ERROR = "nccl_error"  # NCCL communication error


@dataclass
class DiagnosticConfig:
    """Configuration for the diagnostic module."""

    # Timeouts (in seconds)
    timeout: float = 120.0


@dataclass
class DiagnosticResult:
    """Aggregated diagnostic results."""

    stop_reason: StopReason
    is_graceful_stop: bool = False
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    faulty_gpus: List[int] = field(default_factory=list)
    faulty_nodes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a summary string of the diagnostic result."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.get("passed"))
        return (
            f"StopReason: {self.stop_reason.value}, "
            f"Tests: {passed}/{total} passed, "
            f"GracefulStop: {self.is_graceful_stop}"
        )


class DiagnosticRunner:
    """
    Orchestrates diagnostic tests using independent DiagnosticActors.

    This runner spawns DiagnosticActors on specified GPU bundles and runs
    hierarchical diagnostic tests. It is completely independent of any
    application-specific actors (like MegatronActor).

    Usage:
        runner = DiagnosticRunner()
        result = runner.run_on_placement_group(pg, bundle_indices)
    """

    def __init__(self, config: Optional[DiagnosticConfig] = None):
        self.config = config or DiagnosticConfig()

    def run_on_placement_group(
        self,
        pg: PlacementGroup,
        bundle_indices: List[int],
    ) -> DiagnosticResult:
        """
        Spawn DiagnosticActors on a placement group and run all tests.

        This method:
        1. Spawns independent DiagnosticActors on the specified bundles
        2. Initializes NCCL process groups for communication tests
        3. Runs hierarchical tests (GPU → intra-node → inter-node)
        4. Cleans up actors after tests complete
        5. Returns aggregated results

        Args:
            pg: Ray PlacementGroup to schedule actors on
            bundle_indices: List of bundle indices to use for diagnostic actors

        Returns:
            DiagnosticResult with test outcomes and failure classification
        """
        world_size = len(bundle_indices)
        if world_size == 0:
            return DiagnosticResult(
                stop_reason=StopReason.UNKNOWN,
                is_graceful_stop=False,
            )

        logger.info(f"Starting diagnostics on {world_size} GPUs")

        # Spawn first actor to get master addr/port from a GPU worker node
        # (Ray head node doesn't have GPUs, so we need a worker node's IP for NCCL)
        first_actor = DiagnosticActor.options(
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_indices[0],
            ),
        ).remote(rank=0, world_size=world_size, master_addr="", master_port=0)

        master_addr = ray.get(first_actor.get_node_ip.remote())
        master_port = ray.get(first_actor.get_free_port.remote())

        # Update first actor with correct master info
        ray.get(first_actor.set_master_info.remote(master_addr, master_port))

        logger.info(f"Using master_addr={master_addr}, master_port={master_port}")

        # Spawn remaining diagnostic actors
        actors = [first_actor]
        for rank, bundle_idx in enumerate(bundle_indices[1:], start=1):
            actor = DiagnosticActor.options(
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                ),
            ).remote(rank, world_size, master_addr, master_port)
            actors.append(actor)

        try:
            # Collect node topology from all actors
            logger.info("Collecting node topology...")
            node_ids = ray.get(
                [a.get_node_id.remote() for a in actors],
                timeout=30,
            )

            # Group ranks by node
            node_to_ranks: Dict[str, List[int]] = {}
            for rank, node_id in enumerate(node_ids):
                node_to_ranks.setdefault(node_id, []).append(rank)

            logger.info(f"Node topology: {node_to_ranks}")

            # Initialize global process groups on all actors
            logger.info("Initializing process groups...")
            ray.get(
                [a.init_process_group.remote() for a in actors],
                timeout=self.config.timeout,
            )

            # Create intra-node sub-groups for nodes with multiple GPUs
            # IMPORTANT: dist.new_group() must be called by ALL ranks, even those not in the group
            for node_id, ranks in node_to_ranks.items():
                if len(ranks) > 1:
                    logger.info(f"Creating intra-node group for node {node_id[:8]}... with ranks {ranks}")
                    # All actors must call init_intra_node_group with the same ranks list
                    ray.get(
                        [a.init_intra_node_group.remote(ranks) for a in actors],
                        timeout=30,
                    )

            # Run hierarchical tests
            result = self._run_all_phases(actors)

            # Cleanup process groups
            logger.info("Cleaning up process groups...")
            ray.get(
                [a.destroy_process_group.remote() for a in actors],
                timeout=30,
            )

        except Exception as e:
            logger.error(f"Diagnostic run failed: {e}")
            result = DiagnosticResult(
                stop_reason=StopReason.UNKNOWN,
                is_graceful_stop=False,
                test_results=[{"passed": False, "phase": "setup", "error": str(e)}],
            )

        finally:
            # Kill diagnostic actors
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass

        logger.info(f"Diagnostics complete: {result.summary()}")
        return result

    def _run_all_phases(self, actors: List) -> DiagnosticResult:
        """Run all diagnostic phases hierarchically."""
        result = DiagnosticResult(stop_reason=StopReason.UNKNOWN)

        # Phase 1: GPU health
        logger.info("Phase 1: Running GPU health checks...")
        gpu_results = ray.get(
            [a.run_gpu_health_check.remote() for a in actors],
            timeout=self.config.timeout,
        )
        result.test_results.extend(gpu_results)

        gpu_failures = [r for r in gpu_results if not r.get("passed")]
        if gpu_failures:
            logger.warning(f"GPU health check failed on {len(gpu_failures)} actors")
            result.faulty_gpus = [r.get("rank", -1) for r in gpu_failures]
            result.stop_reason = StopReason.GPU_ERROR
            return result

        logger.info("Phase 1 passed: All GPUs healthy")

        # Phase 2: Intra-node communication
        logger.info("Phase 2: Running intra-node communication tests...")
        intra_results = ray.get(
            [a.run_intra_node_comm_test.remote() for a in actors],
            timeout=self.config.timeout,
        )
        result.test_results.extend(intra_results)

        intra_failures = [r for r in intra_results if not r.get("passed")]
        if intra_failures:
            logger.warning(
                f"Intra-node comm test failed on {len(intra_failures)} actors"
            )
            result.stop_reason = StopReason.NCCL_ERROR
            return result

        logger.info("Phase 2 passed: Intra-node communication healthy")

        # Phase 3: Inter-node communication
        logger.info("Phase 3: Running inter-node communication tests...")
        inter_results = ray.get(
            [a.run_inter_node_comm_test.remote() for a in actors],
            timeout=self.config.timeout,
        )
        result.test_results.extend(inter_results)

        inter_failures = [r for r in inter_results if not r.get("passed")]
        if inter_failures:
            logger.warning(
                f"Inter-node comm test failed on {len(inter_failures)} actors"
            )
            result.stop_reason = StopReason.NCCL_ERROR
            return result

        logger.info("Phase 3 passed: Inter-node communication healthy")

        # All tests passed - classify as graceful stop
        result.is_graceful_stop = True
        result.stop_reason = StopReason.GRACEFUL_STOP
        logger.info("All diagnostic phases passed - classified as graceful stop")

        return result

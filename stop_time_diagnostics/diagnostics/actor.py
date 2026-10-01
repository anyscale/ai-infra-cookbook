"""
Independent diagnostic Ray actor for GPU and communication tests.

This can be spawned on any GPU to run diagnostic tests and manages its own NCCL process group.
"""

import os
import socket
import subprocess
import traceback
import logging
from typing import Dict, Any, List, Optional
import torch
import torch.distributed as dist
import ray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class DiagnosticActor:
    """
    Independent Ray actor for running diagnostic tests.

    Can be spawned on any GPU without depending on application actors.
    Manages its own NCCL process group for communication tests.
    """

    def __init__(
        self, rank: int, world_size: int, master_addr: str, master_port: int
    ):
        self._rank = rank
        self._world_size = world_size
        self._master_addr = master_addr
        self._master_port = master_port
        self._process_group_initialized = False
        self._intra_node_group: Optional[dist.ProcessGroup] = None
        self._intra_node_ranks: Optional[List[int]] = None

    def set_master_info(self, master_addr: str, master_port: int):
        """Update master address and port (used when first actor discovers these values)."""
        self._master_addr = master_addr
        self._master_port = master_port

    def init_process_group(self):
        """Initialize NCCL process group for communication tests."""
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = "0"

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        self._process_group_initialized = True
        logger.info(f"DiagnosticActor rank {self._rank}: process group initialized")

    def destroy_process_group(self):
        """Clean up process group."""
        # Destroy intra-node group first if it exists
        if self._intra_node_group is not None:
            dist.destroy_process_group(self._intra_node_group)
            self._intra_node_group = None
            self._intra_node_ranks = None
        if dist.is_initialized():
            dist.destroy_process_group()
        self._process_group_initialized = False

    def init_intra_node_group(self, ranks: List[int]):
        """
        Create a sub-group for intra-node communication.

        Args:
            ranks: List of global ranks that are on the same node as this actor
        """
        self._intra_node_ranks = ranks
        # All ranks must call new_group, but only ranks in the list will be part of the group
        self._intra_node_group = dist.new_group(ranks=ranks)
        logger.info(f"DiagnosticActor rank {self._rank}: intra-node group created with ranks {ranks}")

    def get_node_id(self) -> str:
        """Get Ray node ID."""
        return ray.get_runtime_context().get_node_id()

    def get_gpu_id(self):
        """Get assigned GPU ID."""
        return ray.get_gpu_ids()[0] if ray.get_gpu_ids() else -1

    @staticmethod
    def get_node_ip() -> str:
        """Get current node IP address."""
        return ray._private.services.get_node_ip_address().strip("[]")

    @staticmethod
    def get_free_port() -> int:
        """Get a free port on the current node."""
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def run_gpu_health_check(self) -> Dict[str, Any]:
        """
        Phase 1: Check GPU health via nvidia-smi and simple CUDA operation.

        Returns:
            Dict with 'passed', 'phase', 'rank', and optional 'metrics' or 'error'
        """
        # Collect base info for all results
        base_info = {
            "phase": "gpu_health",
            "rank": self._rank,
            "node_id": self.get_node_id(),
            "node_ip": self.get_node_ip(),
            "ray_gpu_ids": ray.get_gpu_ids(),
        }

        metrics = {}
        errors = []

        # Step 1: Query GPU info via nvidia-smi
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu,ecc.errors.uncorrected.volatile.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            metrics["nvidia_smi_stdout"] = result.stdout.strip()
            metrics["nvidia_smi_stderr"] = result.stderr.strip()
            metrics["nvidia_smi_returncode"] = result.returncode

            # Parse nvidia-smi output
            if result.returncode == 0 and result.stdout.strip():
                # nvidia-smi returns info for all GPUs, parse each line
                gpu_infos = []
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(", ")]
                    if len(parts) >= 5:
                        gpu_info = {
                            "index": parts[0],
                            "name": parts[1],
                            "temperature_c": parts[2],
                            "memory_used_mb": parts[3],
                            "memory_total_mb": parts[4],
                        }
                        if len(parts) >= 6:
                            gpu_info["utilization_percent"] = parts[5]
                        if len(parts) >= 7 and parts[6] not in ["[N/A]", "N/A"]:
                            gpu_info["ecc_errors"] = parts[6]
                        gpu_infos.append(gpu_info)
                metrics["all_gpus"] = gpu_infos
            elif result.returncode != 0:
                errors.append(f"nvidia-smi failed with code {result.returncode}: {result.stderr}")
        except subprocess.TimeoutExpired:
            errors.append("nvidia-smi timed out after 30 seconds")
        except FileNotFoundError:
            errors.append("nvidia-smi not found in PATH")
        except Exception as e:
            errors.append(f"nvidia-smi error: {str(e)}")

        # Step 2: Check CUDA visibility
        try:
            metrics["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
            metrics["cuda_available"] = torch.cuda.is_available()
            metrics["cuda_device_count"] = torch.cuda.device_count()
            if torch.cuda.is_available():
                metrics["cuda_current_device"] = torch.cuda.current_device()
                metrics["cuda_device_name"] = torch.cuda.get_device_name()
                metrics["cuda_device_capability"] = torch.cuda.get_device_capability()
        except Exception as e:
            errors.append(f"CUDA info error: {str(e)}")

        # Step 3: Simple CUDA compute test
        try:
            if not torch.cuda.is_available():
                errors.append("CUDA not available for compute test")
            else:
                device = torch.cuda.current_device()
                test_tensor = torch.zeros(1000, 1000, device=device)
                test_tensor = test_tensor + 1
                torch.cuda.synchronize(device)
                del test_tensor
                metrics["cuda_compute_test"] = "passed"
        except Exception as e:
            errors.append(f"CUDA compute test failed: {str(e)}\n{traceback.format_exc()}")
            metrics["cuda_compute_test"] = "failed"

        # Determine overall pass/fail
        passed = len(errors) == 0

        result = {
            **base_info,
            "passed": passed,
            "metrics": metrics,
        }

        if errors:
            result["errors"] = errors

        return result

    def run_intra_node_comm_test(self) -> Dict[str, Any]:
        """
        Phase 2: Test intra-node GPU communication via all-to-all.

        Uses the intra-node sub-group (GPUs on same node only) for communication test.

        Returns:
            Dict with 'passed', 'phase', 'rank', and optional 'error'
        """
        base_info = {
            "phase": "intra_node_comm",
            "rank": self._rank,
            "node_id": self.get_node_id(),
            "node_ip": self.get_node_ip(),
            "ray_gpu_ids": ray.get_gpu_ids(),
        }

        if not self._process_group_initialized:
            return {
                **base_info,
                "passed": False,
                "errors": ["Process group not initialized"],
            }

        # If no intra-node group, this is a single GPU on the node - skip test
        if self._intra_node_group is None or self._intra_node_ranks is None:
            return {
                **base_info,
                "passed": True,
                "metrics": {"skipped": "single_gpu_on_node", "intra_node_ranks": [self._rank]},
            }

        try:
            device = torch.cuda.current_device()
            group_size = len(self._intra_node_ranks)

            # Create test tensors sized for the intra-node group
            input_tensor = torch.randn(group_size, 1024, device=device)
            output_tensor = torch.empty_like(input_tensor)

            # Run all-to-all within intra-node sub-group only
            dist.all_to_all_single(output_tensor, input_tensor, group=self._intra_node_group)
            torch.cuda.synchronize(device)

            return {
                **base_info,
                "passed": True,
                "metrics": {
                    "group_size": group_size,
                    "intra_node_ranks": self._intra_node_ranks,
                    "data_elements": group_size * 1024,
                },
            }

        except Exception as e:
            return {
                **base_info,
                "passed": False,
                "errors": [f"{str(e)}\n{traceback.format_exc()}"],
            }

    def run_inter_node_comm_test(self) -> Dict[str, Any]:
        """
        Phase 3: Test inter-node communication via all-gather.

        Uses the diagnostic actor's NCCL process group to test cross-node communication.

        Returns:
            Dict with 'passed', 'phase', 'rank', and optional 'error'
        """
        base_info = {
            "phase": "inter_node_comm",
            "rank": self._rank,
            "node_id": self.get_node_id(),
            "node_ip": self.get_node_ip(),
            "ray_gpu_ids": ray.get_gpu_ids(),
        }

        if not self._process_group_initialized:
            return {
                **base_info,
                "passed": False,
                "errors": ["Process group not initialized"],
            }

        try:
            device = torch.cuda.current_device()
            world_size = dist.get_world_size()

            # Create test tensor for all-gather
            input_tensor = torch.randn(1024, device=device)
            output_tensors = [torch.empty_like(input_tensor) for _ in range(world_size)]

            # Run all-gather across all nodes
            dist.all_gather(output_tensors, input_tensor)
            torch.cuda.synchronize(device)

            return {
                **base_info,
                "passed": True,
                "metrics": {"world_size": world_size, "data_elements": 1024},
            }

        except Exception as e:
            return {
                **base_info,
                "passed": False,
                "errors": [f"{str(e)}\n{traceback.format_exc()}"],
            }

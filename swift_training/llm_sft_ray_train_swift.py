"""
Ray Train + Megatron-SWIFT Tutorial
====================================

Distributed LLM finetuning combining:
- Ray Train: Orchestrates distributed workers across GPUs/nodes
- Megatron-SWIFT: Efficient tensor and pipeline parallelism with SWIFT's training interface

Quick Start:
    python llm_sft_ray_train_swift.py --num_workers 4

Architecture:
    main() -> TorchTrainer -> train_loop() on each GPU
                              |
                         megatron_sft_main()
                              |
                         pretrain()

Default Configuration:
    - Tensor Parallelism (TP) = 2
    - Pipeline Parallelism (PP) = 1
    - Data Parallelism (DP) = num_workers / (TP * PP)
    - Model: Qwen/Qwen2.5-1.5B-Instruct
    - Dataset: AI-ModelScope/alpaca-gpt4-data-en (subset for tutorial)
"""

import argparse
import json
import logging
import os
import uuid
from typing import Any, Dict

# ============================================================
# SECTION 1: ENVIRONMENT SETUP
# Required environment variables for Megatron + Ray Train
# ============================================================

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # Required for sequence parallelism
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import ray
import ray.train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

logger = logging.getLogger(__name__)

# Tutorial defaults
DEFAULT_TENSOR_PARALLEL_SIZE = 2
DEFAULT_PIPELINE_PARALLEL_SIZE = 1
DEFAULT_MICRO_BATCH_SIZE = 2
DEFAULT_SEQ_LENGTH = 512
DEFAULT_LEARNING_RATE = 1e-5


# ============================================================
# SECTION 2: TRAINING LOOP
# This function runs on each Ray worker (one per GPU)
# ============================================================


def _detect_attention_backend() -> str:
    """Auto-detect the best attention backend for the current GPU.

    - If flash-attn is installed: use "flash" (works on all SM 8.0+ GPUs)
    - If no flash-attn, prefer "fused" on SM 8.0+ GPUs (cuDNN/TE backend)
    - Fallback: use "auto" and let Megatron decide

    Note: Do NOT set NVTE_FUSED_ATTN / NVTE_UNFUSED_ATTN env vars here.
    Megatron-LM's _set_attention_backend() manages them and will assert-fail
    if they conflict with the --attention-backend flag.
    """
    # Prefer flash-attn only when available.
    try:
        import flash_attn  # noqa: F401
        return "flash"
    except ImportError:
        pass

    import torch
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "fused"
    return "auto"


def train_loop(config: Dict[str, Any]) -> None:
    """Per-worker training loop for Megatron-SWIFT.

    Ray Train calls this on each worker. It:
    1. Synchronizes workers before initialization
    2. Calls SWIFT's megatron_sft_main with appropriate arguments

    SWIFT's megatron module handles:
    - Megatron-LM environment setup (git clone if needed)
    - Model loading and conversion to Megatron format
    - Distributed training with tensor/pipeline parallelism
    - Checkpoint saving in safetensors format
    """
    import torch.distributed as dist

    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    # Get parallelism config from training config
    tp_size = config.get("tensor_parallel_size", DEFAULT_TENSOR_PARALLEL_SIZE)
    pp_size = config.get("pipeline_parallel_size", DEFAULT_PIPELINE_PARALLEL_SIZE)
    dp_size = world_size // (tp_size * pp_size)

    # Adjust batch sizes based on actual DP size
    micro_batch = config.get("micro_batch_size", DEFAULT_MICRO_BATCH_SIZE)
    global_batch = dp_size * micro_batch

    # Auto-detect attention backend based on GPU architecture unless explicitly set.
    attn_backend = config.get("attention_backend") or _detect_attention_backend()

    if world_rank == 0:
        import torch
        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"
        logger.info(f"GPU: {gpu_name}, attention_backend: {attn_backend}")
        logger.info(f"Training with {world_size} GPUs: TP={tp_size}, PP={pp_size}, DP={dp_size}")
        logger.info(f"Batch sizes: micro={micro_batch}, global={global_batch}")

    # Synchronize workers before SWIFT initialization
    # SWIFT's init_process_group checks if dist is already initialized
    if dist.is_initialized():
        dist.barrier()

    # Tell SWIFT's RayHelper that we're in the 'default' worker group.
    # SWIFT detects Ray workers and uses RAY_SWIFT_GROUP for dispatch routing.
    # Without this, _prepare_template and other @RayHelper.function-decorated
    # methods fail with KeyError: 'RAY_SWIFT_GROUP'.
    os.environ["RAY_SWIFT_GROUP"] = "default"

    # Import SWIFT's megatron module
    from swift.megatron import megatron_sft_main

    # Build CLI arguments for SWIFT megatron training
    args_list = [
        "--model", config["model"],
        "--dataset", config["dataset"],
        "--tensor_model_parallel_size", str(tp_size),
        "--pipeline_model_parallel_size", str(pp_size),
        "--micro_batch_size", str(micro_batch),
        "--global_batch_size", str(global_batch),
        "--seq_length", str(config.get("seq_length", DEFAULT_SEQ_LENGTH)),
        "--lr", str(config.get("learning_rate", DEFAULT_LEARNING_RATE)),
        "--train_iters", str(config["train_iters"]),
        "--save", config["output_dir"],
        "--save_interval", str(config.get("save_interval", 100)),
        "--log_interval", str(config.get("log_interval", 10)),
        # Performance optimizations
        "--use_distributed_optimizer", "true",
        "--bf16", "true",
        # Enable sequence parallelism when TP > 1
        "--sequence_parallel", "true" if tp_size > 1 else "false",
        # Dataset settings
        "--dataset_shuffle", "true",
        # Disable version suffix in output dir for predictable paths
        "--add_version", "false",
        # Use non-THD data layout to avoid TE backend filtering issues on some setups.
        "--padding_free", config.get("padding_free", "false"),
        # Attention backend: auto-detected based on GPU compute capability
        # "fused" (cuDNN) requires Hopper+ (SM >= 9.0), "unfused" works on all GPUs
        "--attention_backend", attn_backend,
    ]

    # Add optional LoRA settings if specified
    if config.get("use_lora", False):
        args_list.extend([
            "--tuner_type", "lora",
            "--lora_rank", str(config.get("lora_rank", 8)),
            "--lora_alpha", str(config.get("lora_alpha", 32)),
        ])

    if world_rank == 0:
        logger.info(f"SWIFT CLI args: {' '.join(args_list)}")

    # SWIFT checkpoint save path expects args.json to exist at --save on every worker.
    # With node-local storage paths (e.g., /mnt/local_storage), each worker must
    # create its own copy to avoid FileNotFoundError during save_checkpoint.
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    args_json_path = os.path.join(output_dir, "args.json")
    if not os.path.exists(args_json_path):
        with open(args_json_path, "w", encoding="utf-8") as f:
            json.dump({"args_list": args_list, "config": config}, f, indent=2, sort_keys=True)

    # Run SWIFT megatron training
    megatron_sft_main(args_list)

    if world_rank == 0:
        logger.info("Training completed successfully")


# ============================================================
# SECTION 3: MAIN ENTRY POINT
# Sets up Ray Train and launches distributed training
# ============================================================


def main():
    """Launch Ray Train distributed training."""
    args = parse_args()

    # Validate: num_workers must be divisible by TP * PP
    total_parallel = args.tensor_parallel_size * args.pipeline_parallel_size
    if args.num_workers % total_parallel != 0:
        raise ValueError(
            f"num_workers ({args.num_workers}) must be divisible by "
            f"TP * PP ({args.tensor_parallel_size} * {args.pipeline_parallel_size} = {total_parallel})"
        )

    dp = args.num_workers // total_parallel
    print(f"Configuration: TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}, DP={dp}")
    print(f"Total workers: {args.num_workers}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")

    # Ray Train scaling: PACK strategy keeps TP workers on same node.
    # accelerator_type ensures workers land on nodes with the specified GPU.
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": 1},
        placement_strategy="PACK",
        accelerator_type=args.accelerator_type,
    )

    output_dir = os.path.join(args.storage_path, "swift_outputs")

    train_loop_config = {
        "model": args.model,
        "dataset": args.dataset,
        "output_dir": output_dir,
        "train_iters": args.train_iters,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "micro_batch_size": args.micro_batch_size,
        "seq_length": args.seq_length,
        "learning_rate": args.learning_rate,
        "use_lora": args.use_lora,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "attention_backend": args.attention_backend,
        "padding_free": args.padding_free,
    }

    experiment_name = f"swift_ray_{uuid.uuid4().hex[:8]}"
    print(f"Experiment: {experiment_name}")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=RunConfig(storage_path=args.storage_path, name=experiment_name),
    )

    print("Starting Ray Train...")
    result = trainer.fit()
    print(f"Training finished. Result: {result}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ray Train + Megatron-SWIFT Tutorial",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="AI-ModelScope/alpaca-gpt4-data-en#500",
        help="Dataset name with optional sample count (default: AI-ModelScope/alpaca-gpt4-data-en#500)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of GPUs (default: 4, must be divisible by TP*PP)",
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="/mnt/local_storage",
        help="Storage path for checkpoints (default: /mnt/local_storage)",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=100,
        help="Training iterations (default: 100)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Checkpoint save interval (default: 100)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Logging interval (default: 10)",
    )
    # Parallelism settings
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help=f"Tensor parallelism size (default: {DEFAULT_TENSOR_PARALLEL_SIZE})",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=DEFAULT_PIPELINE_PARALLEL_SIZE,
        help=f"Pipeline parallelism size (default: {DEFAULT_PIPELINE_PARALLEL_SIZE})",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=DEFAULT_MICRO_BATCH_SIZE,
        help=f"Micro batch size per GPU (default: {DEFAULT_MICRO_BATCH_SIZE})",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=DEFAULT_SEQ_LENGTH,
        help=f"Sequence length (default: {DEFAULT_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    # Accelerator / resource settings
    parser.add_argument(
        "--accelerator_type",
        type=str,
        default=None,
        help="GPU accelerator type (e.g. L40S, A10G, A100, H100).\n"
             "Ensures workers are placed on nodes with the specified GPU.\n"
             "See: https://docs.ray.io/en/latest/ray-core/accelerator-types.html",
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        default=None,
        choices=["auto", "flash", "fused", "unfused"],
        help="Attention backend override. If omitted, auto-detects based on installed packages and GPU.",
    )
    parser.add_argument(
        "--padding_free",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether to enable Megatron-SWIFT padding_free (THD layout).",
    )

    # LoRA settings
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA fine-tuning",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()

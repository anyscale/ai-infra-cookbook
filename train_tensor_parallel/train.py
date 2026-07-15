"""
Ray Train + FSDP2 + DTensor Tensor Parallelism Training Tutorial.

This script demonstrates how to train large language models with tensor parallelism
using PyTorch native FSDP2 + DTensor and Ray Train for distributed execution.

Key concepts:
- Tensor Parallelism (TP): Shards model weights across GPUs within a TP group
- Data Parallelism (DP): Replicates the model across DP groups, each processing different data
- 2D Parallelism: Combines TP and DP for scaling to many GPUs

Example usage:
    # 8 GPUs: 4-way tensor parallelism, 2-way data parallelism
    python train_fsdp.py \
        --model_name Qwen/Qwen2-7B \
        --tp_size 4 \
        --dp_size 2 \
        --num_workers 8 \
        --dataset_name wikitext \
        --batch_size 2 \
        --seq_length 2048 \
        --num_epochs 3

    # 4 GPUs: 4-way tensor parallelism only
    python train_fsdp.py \
        --model_name Qwen/Qwen2-7B \
        --tp_size 4 \
        --dp_size 1 \
        --num_workers 4 \
        --dataset_name wikitext \
        --num_epochs 3
"""

import json
import logging
import os
import tempfile
import uuid
from typing import Any, Dict

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import torch
import torch.distributed as dist
from datasets import DownloadConfig, load_dataset
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ray.train
import ray.train.torch
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from args import get_args

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================


def create_dataloader(
    model_name: str,
    dataset_name: str,
    seq_length: int,
    batch_size: int,
    dp_rank: int,
    dp_size: int,
    seed: int = 42,
    dataset_percentage: float = 10.0,
) -> DataLoader:
    """
    Create dataloader with TP-aware sharding.

    IMPORTANT: Uses dp_rank/dp_size for sharding (NOT world_rank/world_size).
    This ensures all TP ranks in the same DP group see identical batches.
    """
    world_rank = ray.train.get_context().get_world_rank()

    # Handle datasets that require a config name
    dataset_config = "wikitext-2-raw-v1" if dataset_name == "wikitext" else None
    split_spec = f"train[:{int(dataset_percentage)}%]"

    # Rank 0 downloads first to avoid conflicts
    if world_rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dataset = load_dataset(
            dataset_name, dataset_config, split=split_spec,
            download_config=DownloadConfig(disable_tqdm=True),
        )
    dist.barrier()

    # Other ranks load from cache
    if world_rank != 0:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dataset = load_dataset(
            dataset_name, dataset_config, split=split_spec,
            download_config=DownloadConfig(disable_tqdm=True),
        )

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], padding="max_length", max_length=seq_length, truncation=True
        )

    tokenized = dataset.map(
        tokenize_fn, batched=True, num_proc=1, keep_in_memory=True,
        remove_columns=dataset.column_names,
    )

    # Add labels (same as input_ids for causal LM)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized = tokenized.map(add_labels, batched=True, num_proc=1, keep_in_memory=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Use DP rank/size for sharding (ensures TP ranks get same data)
    sampler = DistributedSampler(
        tokenized, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=seed
    )

    return DataLoader(tokenized, batch_size=batch_size, sampler=sampler, drop_last=True)


# =============================================================================
# Training Loop
# =============================================================================


def train_loop_per_worker(config: Dict[str, Any]) -> None:
    """
    Main training loop executed by each Ray Train worker.

    This function:
    1. Sets up the 2D device mesh for TP + DP
    2. Creates and shards the model with DTensor (TP) and FSDP2 (DP)
    3. Runs the training loop with checkpointing
    """
    # Get Ray Train context
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    device = ray.train.torch.get_device()

    tp_size = config["tp_size"]
    dp_size = config["dp_size"]

    if world_rank == 0:
        logger.info(f"Worker started: world_rank={world_rank}, world_size={world_size}")

    # -------------------------------------------------------------------------
    # Step 1: Create 2D Device Mesh
    # -------------------------------------------------------------------------
    # The mesh is organized as (dp, tp) where:
    # - dp dimension: FSDP2 shards optimizer states and gradients
    # - tp dimension: DTensor shards model weights for tensor parallelism

    # Calculate TP and DP rank
    tp_rank = world_rank % tp_size
    dp_rank = world_rank // tp_size

    # Validate configuration
    if dp_size * tp_size != world_size:
        raise ValueError(
            f"dp_size ({dp_size}) * tp_size ({tp_size}) must equal "
            f"world_size ({world_size})"
        )

    # Validate TP size divides num_key_value_heads (required for Qwen/Llama models)
    hf_config = AutoConfig.from_pretrained(config["model_name"], trust_remote_code=True)
    if hf_config.num_key_value_heads % tp_size != 0:
        raise ValueError(
            f"TP size {tp_size} must divide num_key_value_heads "
            f"{hf_config.num_key_value_heads}"
        )

    if world_rank == 0:
        logger.info(f"Setting up 2D mesh: dp_size={dp_size}, tp_size={tp_size}")

    # Create 2D device mesh: (dp, tp)
    device_mesh = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    if world_rank == 0:
        logger.info(f"Device mesh created: {device_mesh}")

    # -------------------------------------------------------------------------
    # Step 2: Create and Shard Model
    # -------------------------------------------------------------------------

    dtype = torch.bfloat16

    # Create model with random initialization on the target device
    prev_device = torch.get_default_device()
    torch.set_default_device(device)
    model = AutoModelForCausalLM.from_config(hf_config).to(dtype=dtype)
    torch.set_default_device(prev_device)

    # Get transformer layers for parallelization (Qwen model structure)
    layers = model.model.layers

    # TP mapping for transformer layers (Qwen/Llama-style models)
    # ColwiseParallel: splits output features across TP ranks
    # RowwiseParallel: splits input features across TP ranks
    tp_mapping = {
        # Attention projections
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        # MLP projections
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }

    if world_rank == 0:
        logger.info(f"Applying DTensor TP to {len(layers)} layers")

    # Apply DTensor TP to transformer layers
    for layer in layers:
        parallelize_module(layer, tp_mesh, tp_mapping)

    # Apply FSDP2 (fully_shard) for data parallelism
    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

    if dp_size > 1:
        if world_rank == 0:
            logger.info("Applying FSDP2 to transformer layers")

        for layer in layers:
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)

        # Apply to the whole model
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
    else:
        if world_rank == 0:
            logger.info("dp_size=1, skipping FSDP sharding (TP only)")

    # Create optimizer
    # Note: Use foreach=False because DTensor doesn't support fused optimizer ops
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-5),
        weight_decay=config.get("weight_decay", 0.01),
        foreach=False,
    )

    if world_rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {num_params:,} parameters")
        if dp_size > 1:
            logger.info(f"2D parallelism: {dp_size} DP x {tp_size} TP")
        logger.info("torch.autocast enabled with dtype=bfloat16")

    # -------------------------------------------------------------------------
    # Step 3: Create Dataloader
    # -------------------------------------------------------------------------
    # IMPORTANT: Use dp_rank/dp_size for sharding, NOT world_rank/world_size
    # This ensures all TP ranks in the same DP group see identical batches

    dataloader = create_dataloader(
        model_name=config["model_name"],
        dataset_name=config["dataset_name"],
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        dp_rank=dp_rank,
        dp_size=dp_size,
        seed=config.get("seed", 42),
        dataset_percentage=config.get("dataset_percentage", 10.0),
    )

    steps_per_epoch = len(dataloader)
    if world_rank == 0:
        logger.info(f"Dataloader created: {steps_per_epoch} steps per epoch")

    # -------------------------------------------------------------------------
    # Step 4: Training Loop
    # -------------------------------------------------------------------------

    model.train()

    for epoch in range(config["num_epochs"]):
        dataloader.sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with autocast
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = outputs.loss

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Track loss
            loss_value = loss.item()
            running_loss += loss_value
            num_batches += 1

            # Log progress
            if world_rank == 0 and step % config.get("log_interval", 10) == 0:
                logger.info(
                    f"Epoch: {epoch} Step: {step + 1}/{steps_per_epoch} Loss: {loss_value:.4f}"
                )

            # Debug mode: stop early for testing
            if config.get("debug_steps", 0) > 0 and step + 1 >= config["debug_steps"]:
                if world_rank == 0:
                    logger.info(f"Debug steps finished. Stopping epoch {epoch}.")
                break

        # Calculate average loss for epoch
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

        # Save checkpoint at end of epoch
        _save_checkpoint(model, optimizer, world_rank, epoch, step, avg_loss)

        if world_rank == 0:
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    world_rank: int,
    epoch: int,
    step: int,
    avg_loss: float,
) -> None:
    """Save checkpoint and report to Ray Train."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Each rank saves its model/optimizer shard
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, f"model_rank{world_rank}.pt"),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(checkpoint_dir, f"optimizer_rank{world_rank}.pt"),
        )

        # Save metadata (from rank 0 only)
        if world_rank == 0:
            with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
                json.dump({"epoch": epoch, "step": step}, f)

        # All workers must call report() with their checkpoint
        checkpoint = Checkpoint.from_directory(tmp_dir)
        ray.train.report({"loss": avg_loss, "epoch": epoch}, checkpoint=checkpoint)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point."""
    args = get_args()

    # Validate parallelism configuration
    if args.tp_size * args.dp_size != args.num_workers:
        raise ValueError(
            f"tp_size ({args.tp_size}) * dp_size ({args.dp_size}) "
            f"must equal num_workers ({args.num_workers})"
        )

    print(f"Configuration: {args}")

    # Build train_loop_config
    train_loop_config = {
        "model_name": args.model_name,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "dataset_name": args.dataset_name,
        "dataset_percentage": args.dataset_percentage,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "log_interval": args.log_interval,
        "debug_steps": args.debug_steps,
        "seed": args.seed,
    }

    # Configure Ray Train
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
    )

    # Generate experiment name
    name = args.experiment_name
    if name is None:
        name = f"fsdp_tp{args.tp_size}_dp{args.dp_size}_{uuid.uuid4().hex[:8]}"

    print(f"Experiment name: {name}")

    run_config = RunConfig(
        storage_path=args.storage_path,
        name=name,
    )

    # Create and run trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    result = trainer.fit()
    print(f"Training finished. Result: {result}")


if __name__ == "__main__":
    main()

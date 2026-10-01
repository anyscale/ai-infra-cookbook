"""Command line argument parsing for tensor parallelism training."""

import argparse


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ray Train + FSDP2 + DTensor Tensor Parallelism Training"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-7B",
        help="HuggingFace model name or path",
    )

    # Parallelism configuration
    parser.add_argument(
        "--tp_size",
        type=int,
        required=True,
        help="Tensor parallel degree",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Data parallel degree",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="Total number of workers (must equal tp_size * dp_size)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_percentage",
        type=float,
        default=10.0,
        help="Percentage of dataset to use (0-100)",
    )

    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-GPU micro batch size",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )

    # Checkpointing configuration
    parser.add_argument(
        "--storage_path",
        type=str,
        default="/mnt/cluster_storage",
        help="Storage path for checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )

    # Logging and debugging
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Logging interval (steps)",
    )
    parser.add_argument(
        "--debug_steps",
        type=int,
        default=0,
        help="Stop after this many steps per epoch (0 = run full epoch)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()

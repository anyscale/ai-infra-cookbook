"""
Training Utilities for VLA Fine-Tuning
=======================================

Helper functions used by vla.py. These are separated out to keep the main
script focused on the Ray Data + Ray Train pipeline, while housing the
model-specific plumbing (freezing layers, checkpointing, collation, etc.)
in one place.
"""

import os
import tempfile
import warnings

import numpy as np
import torch


# ============================================================================
# PI0.5 Attention Mask Patch
# ============================================================================
#
# The PI0.5 model's make_att_2d_masks function assumes pad_masks and
# att_masks always have matching sequence lengths. In practice, the
# preprocessor can produce mismatched lengths (e.g. after truncation or
# when the tokenizer pads differently). This monkey-patch detects the
# mismatch and truncates both masks to the shorter length, preventing a
# hard crash during training.
# ============================================================================


def apply_pi05_attention_mask_patch():
    """Monkey-patch make_att_2d_masks to tolerate pad/attention mask length mismatches."""
    import lerobot.policies.pi05.modeling_pi05 as mp

    if getattr(mp, "_PI05_MASK_PATCH_APPLIED", False):
        return
    _orig = mp.make_att_2d_masks

    def _patched(pad_masks, att_masks):
        pad_len, att_len = pad_masks.shape[-1], att_masks.shape[-1]
        if pad_len != att_len:
            warnings.warn(
                f"PI0.5 mask length mismatch: pad_masks={pad_len}, att_masks={att_len}; "
                f"truncating to {min(pad_len, att_len)}"
            )
            L = min(pad_len, att_len)
            return _orig(pad_masks[..., :L], att_masks[..., :L])
        return _orig(pad_masks, att_masks)

    mp.make_att_2d_masks = _patched
    mp._PI05_MASK_PATCH_APPLIED = True


# ============================================================================
# Dataset Helpers
# ============================================================================


def extract_stats(source):
    """Extract normalization stats (mean/std) from a LeRobot datasource.

    Returns a dict with keys like "action" and "observation.state", each
    containing "mean" and "std" arrays from the dataset metadata.
    """
    raw = source.meta.stats
    stats = {}
    for key in ("action", "observation.state"):
        if key in raw:
            stats[key] = {"mean": raw[key]["mean"], "std": raw[key]["std"]}
    return stats


def renamed_image_keys(source, camera_rename):
    """Get camera column names after applying the rename map."""
    return [camera_rename.get(k, k) for k in source.meta.video_keys]


# ============================================================================
# Model Loading
# ============================================================================


def load_pi05_policy(pretrained_path="lerobot/pi05_base"):
    """Load PI0.5, apply the attention mask patch, and freeze the backbone.

    Returns the policy with only the action/time projection heads unfrozen
    (action_in_proj, action_out_proj, time_mlp_in, time_mlp_out). The large
    pretrained vision-language backbone stays frozen, dramatically reducing
    memory and compute while still adapting the model to new tasks.

    Also applies the attention mask monkey-patch (see above) so training
    doesn't crash on sequence-length mismatches.
    """
    from lerobot.policies.pi05 import PI05Policy

    apply_pi05_attention_mask_patch()

    policy = PI05Policy.from_pretrained(
        pretrained_path, device="cuda", dtype=torch.bfloat16, train_expert_only=True,
    )

    # Freeze everything, then unfreeze the small trainable heads.
    # train_expert_only=True sets a config flag but doesn't actually toggle
    # requires_grad -- we do that manually here.
    for p in policy.parameters():
        p.requires_grad = False
    for name, module in policy.model.named_children():
        if name in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}:
            for p in module.parameters():
                p.requires_grad = True

    return policy


# ============================================================================
# Checkpoint Save / Load
# ============================================================================


def load_checkpoint(checkpoint, policy, optimizer) -> tuple[int, int]:
    """Restore model/optimizer state from a Ray Train checkpoint.

    Returns (start_epoch, start_step).
    """
    import ray.cloudpickle as pickle

    with checkpoint.as_directory() as d:
        with open(os.path.join(d, "state.pkl"), "rb") as f:
            state = pickle.load(f)
    policy.module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optim"])
    return state["epoch"] + 1, state.get("step", 0)


def make_checkpoint(policy, optimizer, epoch, step):
    """Serialize model + optimizer state into a Ray Train Checkpoint.

    This captures everything needed to resume training: model weights,
    optimizer state, and the current epoch/step.
    The checkpoint is written to a temp directory and returned as a
    ray.train.Checkpoint for use with ray.train.report().
    """
    import ray.cloudpickle as pickle
    import ray.train

    ckpt_dir = tempfile.mkdtemp(prefix="pi05_ckpt_")
    with open(os.path.join(ckpt_dir, "state.pkl"), "wb") as f:
        pickle.dump(
            {"model": policy.module.state_dict(), "optim": optimizer.state_dict(),
             "epoch": epoch, "step": step},
            f,
        )
    return ray.train.Checkpoint.from_directory(ckpt_dir)


# ============================================================================
# Sequence Truncation
# ============================================================================


def truncate_batch(batch: dict, max_len: int) -> dict:
    """Clip sequence-length tensors to max_len tokens.

    PI0.5 can produce very long token sequences depending on the number of
    cameras and action horizon. Truncating caps GPU memory usage at the cost
    of losing some context. Set max_len=None to disable.
    """
    if not max_len:
        return batch
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 2:
            batch[k] = v[..., :max_len]
    return batch


# ============================================================================
# Collation: numpy dicts -> torch tensors on GPU
# ============================================================================


def make_collate_fn(device: torch.device):
    """Return a collate function that converts numpy batch dicts to torch tensors.

    Ray Data delivers batches as numpy arrays. The returned function moves
    them to GPU as torch tensors, preserving dtype semantics: integer arrays
    become torch.long, booleans become torch.bool, and everything else
    becomes torch.float32.

    The ``task`` column is kept as a Python list of strings (the model's
    language conditioning input).
    """
    def collate(batch: dict) -> dict:
        task = list(batch.pop("task"))
        result = {}
        for k, v in batch.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.integer):
                result[k] = torch.tensor(arr, dtype=torch.long, device=device)
            elif np.issubdtype(arr.dtype, np.bool_):
                result[k] = torch.tensor(arr, dtype=torch.bool, device=device)
            else:
                result[k] = torch.tensor(arr, dtype=torch.float32, device=device)
        result["task"] = task
        return result
    return collate


# ============================================================================
# Training Step Helpers
# ============================================================================


def train_step(policy, batch, preprocessor, max_len, grad_accum):
    """Run one forward + backward pass. Returns the scalar loss value.

    This is vanilla PyTorch training: autocast for mixed precision, scale the
    loss for gradient accumulation, and call backward. Nothing Ray-specific
    happens here -- it's the same code you'd write for single-GPU training.
    """
    batch = preprocessor(batch)
    batch = truncate_batch(batch, max_len)
    batch.pop("task", None)
    batch.pop("task_index", None)

    with torch.autocast("cuda", torch.bfloat16):
        out = policy(batch)
        loss = out.loss if hasattr(out, "loss") else out[0]

    (loss / grad_accum).backward()
    return float(loss.detach())


def optimizer_step(policy, optimizer):
    """Clip gradients, step the optimizer, and zero gradients.

    Called every ``grad_accum`` micro-batches.
    """
    torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad], max_norm=1.0,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
